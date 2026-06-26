import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import jsonify, request, send_file
from flask_jwt_extended import get_jwt

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.node_pm import NodeProcessManager

from ..config import config
from ..helpers.auth_helpers import admin_required
from ..utils import error, response
from .api import api

node_process_manager = NodeProcessManager(config.node_config)

_APPLICATION_LOG_BASENAME = "application.log"
_LOG_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def _actor_from_request() -> dict:
    """Build audit metadata for the user or process that called the route.

    The node process manager records this actor information for start, stop,
    and restart requests. This helper only reads JWT claims already available
    on the request; it does not enforce authorization. Route decorators remain
    responsible for access control.

    Returns:
        Dictionary containing the GUI source marker and available identity
        claims for the current request.
    """

    claims = get_jwt()

    return {
        "source": "gui",
        "user_id": claims.get("sub"),
        "email": claims.get("email"),
        "role": claims.get("role"),
        "name": claims.get("name"),
        "surname": claims.get("surname"),
    }


def _config_field_schema(
    section: str,
    key: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the GUI schema for a single node configuration field.

    Args:
        section: Name of the `config.ini` section that owns the field.
        key: Name of the option inside the section.
        schema: GUI configuration descriptor returned by `get_gui_config_sections()`.

    Returns:
        The field descriptor used for validation, normalization, and UI
        rendering. The descriptor includes values such as `type`, `editable`,
        `options`, and `min` when they apply.

    Raises:
        ValueError: If the section or key is not present in the GUI
            configuration descriptor.
    """

    if section not in schema:
        raise ValueError(f"Unsupported node configuration section '{section}'")

    fields = schema[section]["fields"]
    if key not in fields:
        raise ValueError(
            f"Unsupported node configuration key '{key}' for section '{section}'"
        )

    return fields[key]


def _normalize_config_value(
    section: str,
    key: str,
    value: Any,
    schema: Dict[str, Any],
) -> bool | int | str:
    """Normalize a raw request or file value according to the field schema.

    The GUI and config file do not share the same runtime types: JSON requests
    can contain booleans and numbers, while `config.ini` stores strings. This
    function converts either representation into the typed value used by the
    API response and conflict checks.

    Args:
        section: Name of the `config.ini` section that owns the value.
        key: Name of the option inside the section.
        value: Raw value from the request body or current config file.
        schema: GUI configuration descriptor returned by `get_gui_config_sections()`.

    Returns:
        A normalized boolean, integer, or string value.

    Raises:
        ValueError: If the key is unsupported, the field type is unsupported,
            or the value does not satisfy the field type constraints.
    """

    field = _config_field_schema(section, key, schema)
    field_type = field["type"]

    if field_type == "boolean":
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False

        raise ValueError(f"Invalid boolean value for '{key}'")

    if field_type == "integer":
        if isinstance(value, bool):
            raise ValueError(f"Invalid integer value for '{key}'")

        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer value for '{key}'") from exc

        min_value = field.get("min", 0)
        if parsed < min_value:
            raise ValueError(f"'{key}' must be greater than or equal to {min_value}")

        return parsed

    if field_type == "enum":
        normalized = str(value).strip().upper()
        options = field.get("options", [])
        if normalized not in options:
            raise ValueError(
                f"Invalid value '{value}' for '{key}'. "
                f"Expected one of: {', '.join(options)}"
            )
        return normalized

    if field_type == "string":
        return "" if value is None else str(value)

    raise ValueError(f"Unsupported field type '{field_type}' for '{key}'")


def _node_config_response_payload() -> Dict[str, Any]:
    """Build the payload returned by `GET /api/node/config`.

    The payload is intentionally derived from the latest on-disk config after
    `config.node_config.read()` has been called by the route. Each field is
    returned with its GUI descriptor and normalized current value so the
    frontend can render sections dynamically without hardcoded key lists.

    Returns:
        Dictionary containing all node configuration sections, their fields,
        normalized field values, editability flags, and current node process
        state.
    """

    schema = config.node_config.get_gui_config_sections()

    modification_status = _config_modification_status()

    return {
        "sections": {
            section: {
                "label": section_schema.get("label"),
                "fields": {
                    key: {
                        name: value
                        for name, value in {
                            **field_schema,
                            "value": _normalize_config_value(
                                section,
                                key,
                                config.node_config.get(section, key),
                                schema,
                            ),
                        }.items()
                        if name != "env"
                    }
                    for key, field_schema in section_schema["fields"].items()
                },
            }
            for section, section_schema in schema.items()
        },
        "node_state": node_process_manager.get_status().value,
        **modification_status,
    }


def _config_updates_from_request(
    payload: Any,
) -> tuple[str, Dict[str, bool | int | str], Dict[str, bool | int | str], bool]:
    """Validate and normalize a node configuration PATCH request body.

    The request must target one section and provide every editable value in
    that section. Unless `force` is true, the request must also include
    matching `base_values`, which are the values that were last shown to the
    user. Those base values are later compared against the current file values
    to detect concurrent edits.

    Args:
        payload: Parsed JSON request body.

    Returns:
        Tuple containing the target section, normalized requested updates,
        normalized base values, and the overwrite flag.

    Raises:
        ValueError: If the payload shape is invalid, the section/key is
            unsupported, a read-only key is requested, or any value fails
            type validation.
    """

    if not isinstance(payload, dict):
        raise ValueError("Request body must be an object")

    schema = config.node_config.get_gui_config_sections()
    section = payload.get("section")
    if not isinstance(section, str) or not section:
        raise ValueError("'section' must be a non-empty string")
    if section not in schema:
        raise ValueError(f"Unsupported node configuration section '{section}'")

    values = payload.get("values")
    if not isinstance(values, dict):
        raise ValueError("'values' must be an object")

    if not values:
        raise ValueError("No configuration values provided")

    editable_keys = {
        key
        for key, field in schema[section]["fields"].items()
        if field.get("editable", True)
    }
    value_keys = set(values.keys())
    if value_keys != editable_keys:
        missing = sorted(editable_keys - value_keys)
        unsupported = sorted(value_keys - editable_keys)
        message_parts = []
        if missing:
            message_parts.append(
                f"Missing configuration value(s): {', '.join(missing)}"
            )
        if unsupported:
            message_parts.append(
                f"Unsupported or read-only configuration value(s): "
                f"{', '.join(unsupported)}"
            )
        raise ValueError("; ".join(message_parts))

    base_values = payload.get("base_values")
    force = bool(payload.get("force", False))
    if not force and not isinstance(base_values, dict):
        raise ValueError("'base_values' must be an object")
    if not force and set(base_values.keys()) != editable_keys:
        missing = sorted(editable_keys - set(base_values.keys()))
        unsupported = sorted(set(base_values.keys()) - editable_keys)
        message_parts = []
        if missing:
            message_parts.append(f"Missing base value(s): {', '.join(missing)}")
        if unsupported:
            message_parts.append(
                f"Unsupported or read-only base value(s): {', '.join(unsupported)}"
            )
        raise ValueError("; ".join(message_parts))

    updates = {}
    normalized_base_values = {}
    for key, value in values.items():
        field = _config_field_schema(section, key, schema)
        if not field.get("editable", True):
            raise ValueError(
                f"Node configuration key '{key}' in section '{section}' is read-only"
            )

        updates[key] = _normalize_config_value(section, key, value, schema)
        if not force:
            if key not in base_values:
                raise ValueError(f"Missing base value for '{key}'")
            normalized_base_values[key] = _normalize_config_value(
                section,
                key,
                base_values[key],
                schema,
            )

    return section, updates, normalized_base_values, force


def _config_update_conflicts(
    section: str,
    updates: Dict[str, bool | int | str],
    base_values: Dict[str, bool | int | str],
) -> tuple[Dict[str, Any], Dict[str, bool | int | str]]:
    """Compare requested updates with the latest file values.

    Conflict detection prevents silently overwriting edits made directly in
    `config.ini` or by another GUI session after the current user loaded the
    form. A field conflicts when its current file value differs from the
    `base_values` submitted by the frontend.

    Args:
        section: Target config section.
        updates: Normalized values requested by the user.
        base_values: Normalized values last shown to the user.

    Returns:
        A tuple containing conflict details and current file values for all
        editable keys in the section. Conflict details include the base value,
        current file value, and requested value for each conflicted key.
    """

    schema = config.node_config.get_gui_config_sections()
    conflicts = {}
    current_values = {}

    for key, requested in updates.items():
        current = _normalize_config_value(
            section,
            key,
            config.node_config.get(section, key),
            schema,
        )
        current_values[key] = current

        if current != base_values.get(key):
            conflicts[key] = {
                "base": base_values.get(key),
                "current": current,
                "requested": requested,
            }

    return conflicts, current_values


def _conflict_response(data: Dict[str, Any], message: str):
    """Create a standard API error response for config write conflicts.

    Args:
        data: Conflict payload returned under the `result` key.
        message: User-facing explanation of the conflict.

    Returns:
        Flask JSON response matching the GUI API error envelope.
    """

    return jsonify({"success": False, "result": data, "message": message})


def _parse_process_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse a process-state timestamp as a timezone-aware UTC datetime.

    Args:
        value: Timestamp stored in the node process-state table.

    Returns:
        Parsed UTC datetime, or `None` when the timestamp is missing or invalid.
    """

    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _config_modification_status(
    process_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool | Optional[str]]:
    """Check whether `config.ini` changed after the current node startup.

    The check intentionally uses the simple file modification time approach:
    compare the current `config.ini` mtime with the `started_at` timestamp of
    the running node process. The warning is only meaningful for a running node,
    because stopped nodes do not have currently effective in-memory values.

    Args:
        process_state: Optional process-state dictionary. When omitted, the
            latest state is read from the node process manager.

    Returns:
        Dictionary containing the boolean modification status and an optional
        message. The message is populated when the node is running but the
        startup timestamp cannot be read or parsed.
    """

    try:
        state = process_state or node_process_manager.get_process_state().to_dict()
        if str(state.get("state", "")).lower() != "running":
            return {
                "config_modified_after_startup": False,
                "config_startup_check_message": None,
            }

        started_at = _parse_process_timestamp(state.get("started_at"))
        if started_at is None:
            return {
                "config_modified_after_startup": False,
                "config_startup_check_message": (
                    "Could not determine node startup time; cannot check whether "
                    "config.ini was modified after node startup."
                ),
            }

        config_mtime = datetime.fromtimestamp(
            os.path.getmtime(config.node_config.config_path),
            timezone.utc,
        )
        return {
            "config_modified_after_startup": config_mtime > started_at,
            "config_startup_check_message": None,
        }
    except Exception:
        return {
            "config_modified_after_startup": False,
            "config_startup_check_message": (
                "Could not check whether config.ini was modified after node startup."
            ),
        }


def _node_args_from_request() -> tuple[dict, bool]:
    """Build safe node_args from GUI request body.

    Expected frontend body:
    {
        "gpu": false,
        "gpu_num": 1,
        "gpu_only": false,
        "debug": false
    }

    `background` is forced to True for GUI API calls so the HTTP request
    does not block.

    Returns:
        Tuple containing sanitized node process arguments and the background
        execution flag.
    """

    req = request.get_json(silent=True) or {}

    background = True

    gpu_only = bool(req.get("gpu_only", False))
    gpu = bool(req.get("gpu", False)) or gpu_only
    gpu_num = int(req.get("gpu_num", 0))
    debug = bool(req.get("debug", False))

    if gpu_num < 0:
        gpu_num = 0

    node_args = {
        "gpu": gpu,
        "gpu_num": gpu_num,
        "gpu_only": gpu_only,
        "debug": debug,
    }

    return node_args, background


@api.route("/node/config", methods=["GET"])
def get_node_config():
    """Return the latest node configuration fields editable from the GUI.

    The route re-reads `config.ini` before building the response so manual file
    edits are reflected when the user refreshes the page. Field descriptors and
    current values are returned together to let the frontend render the form
    dynamically for every supported section.

    Returns:
        HTTP 200 with the current section/field descriptors and node state, or
        HTTP 500 if the configuration file cannot be read.
    """

    try:
        config.node_config.read()
        return response(_node_config_response_payload()), 200
    except Exception as e:
        return error(f"Could not get node configuration: {e}"), 500


@api.route("/node/config", methods=["PATCH"])
@admin_required
def update_node_config():
    """Write GUI-submitted node configuration changes to `config.ini`.

    The route accepts one section per request. Before writing, it re-reads the
    file and compares the current file values with the `base_values` submitted
    by the frontend. If the file changed since the user loaded the form, the
    route returns a conflict instead of overwriting those external edits. A
    request with `force` set to true bypasses that conflict check and writes
    the requested values.

    Returns:
        HTTP 200 with the written fields and restart metadata, HTTP 400 for
        invalid request data, HTTP 409 for concurrent file edits, or HTTP 500
        for unexpected read/write failures.
    """

    req = request.get_json(silent=True) or {}

    try:
        config.node_config.read()
        section, updates, base_values, force = _config_updates_from_request(req)
    except ValueError as e:
        return error(str(e)), 400
    except Exception as e:
        return error(f"Could not update node configuration: {e}"), 500

    try:
        if not force:
            conflicts, current_values = _config_update_conflicts(
                section,
                updates,
                base_values,
            )
            if conflicts:
                return _conflict_response(
                    {
                        "section": section,
                        "conflicts": conflicts,
                        "current_values": current_values,
                    },
                    (
                        "Configuration file has been modified. Refresh the "
                        "latest values or confirm overwrite."
                    ),
                ), 409

        for key, value in updates.items():
            config.node_config.set(section, key, str(value))
        config.node_config.write()

        node_state = node_process_manager.get_status().value
        modification_status = _config_modification_status()
        return response(
            {
                "section": section,
                "values": updates,
                "node_state": node_state,
                "requires_restart": node_state == "running",
                **modification_status,
            },
            "Node configuration has been updated.",
        ), 200
    except Exception as e:
        return error(f"Could not update node configuration: {e}"), 500


def _application_log_dir() -> str:
    """Return the directory that stores node application log files.

    Returns:
        Absolute path to the node log directory.
    """

    return os.path.join(config["NODE_FEDBIOMED_ROOT"], "log")


def _is_application_log_file(name: str) -> bool:
    """Check whether a file name matches an application log file.

    Args:
        name: File name to validate.

    Returns:
        True if the name is the current application log or a rotated
        application log file, False otherwise.
    """

    return name == _APPLICATION_LOG_BASENAME or name.startswith(
        f"{_APPLICATION_LOG_BASENAME}."
    )


def _application_log_path(name: Optional[str] = None) -> str:
    """Resolve and validate an application log file path.

    Args:
        name: Application log file name. Defaults to the current log file.

    Returns:
        Absolute real path to the requested application log file.

    Raises:
        FedbiomedError: If the requested name is not an application log file
            or would resolve outside the node log directory.
    """

    name = name or _APPLICATION_LOG_BASENAME
    if os.path.basename(name) != name or not _is_application_log_file(name):
        raise FedbiomedError("Invalid application log file")

    log_dir = os.path.realpath(_application_log_dir())
    path = os.path.realpath(os.path.join(log_dir, name))
    if os.path.commonpath([log_dir, path]) != log_dir:
        raise FedbiomedError("Invalid application log file")

    return path


def _application_log_files() -> List[Dict[str, Any]]:
    """List application log files available for display and download.

    Returns:
        List of log file metadata dictionaries with name, size, and mtime keys.
        The current application log is listed first, followed by rotated files
        from newest to oldest.
    """

    try:
        entries = os.listdir(_application_log_dir())
    except FileNotFoundError:
        return []

    files = []
    for name in entries:
        if not _is_application_log_file(name):
            continue

        try:
            path = _application_log_path(name)
        except FedbiomedError:
            continue

        if not os.path.isfile(path):
            continue

        try:
            stat = os.stat(path)
        except OSError:
            continue

        files.append(
            {
                "name": name,
                "size": int(stat.st_size),
                "mtime": int(stat.st_mtime),
            }
        )

    files.sort(
        key=lambda item: (
            item["name"] != _APPLICATION_LOG_BASENAME,
            -item["mtime"],
            item["name"],
        )
    )
    return files


def _parse_positive_int(value: Any, default: int, maximum: int) -> Tuple[int, bool]:
    """Parse a positive integer query argument with bounds.

    Used by `node_logs` to validate the `page_size` query parameter.

    Args:
        value: Raw value to parse.
        default: Value returned when the raw value is missing or empty.
        maximum: Maximum accepted value.

    Returns:
        Tuple containing the parsed value and whether parsing succeeded.
    """

    if value in (None, ""):
        return default, True

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default, False

    return max(1, min(parsed, maximum)), True


def _parse_cursor(value: Any, file_size: int) -> Tuple[int, bool]:
    """Parse a log pagination cursor/file pointer.

    Used by `_read_application_log_page` to resolve the byte offset in the
    selected log file before which older lines should be read.

    Args:
        value: Raw cursor/file pointer value to parse.
        file_size: Size of the log file in bytes.

    Returns:
        Tuple containing the clamped file byte offset and whether parsing
        succeeded. Missing values resolve to the end of the file.
    """

    if value in (None, ""):
        return file_size, True

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return file_size, False

    return max(0, min(parsed, file_size)), True


def _extract_log_timestamp(line: str) -> Optional[str]:
    """Extract the leading timestamp from a raw application log line.

    Args:
        line: Raw log line.

    Returns:
        Timestamp string when the line starts with the expected timestamp
        format, or None otherwise.
    """

    match = _LOG_TIMESTAMP_RE.match(line)
    return match.group(1) if match else None


def _read_application_log_page(
    path: str,
    *,
    cursor: Optional[int] = None,
    page_size: int = 100,
    block_size: int = 8192,
) -> Dict[str, Any]:
    """Read a page of raw application log lines before a byte cursor.

    Args:
        path: Application log file path to read.
        cursor: Byte offset before which older lines should be read. Defaults
            to the end of the file.
        page_size: Maximum number of lines to return.
        block_size: Number of bytes to read per backwards file scan.

    Returns:
        Dictionary containing log items, next cursor, whether older lines are
        available, and file size.

    Raises:
        FedbiomedError: If the cursor value cannot be parsed.
    """

    file_size = os.path.getsize(path)
    end, valid_cursor = _parse_cursor(cursor, file_size)
    if not valid_cursor:
        raise FedbiomedError("Invalid 'cursor'")

    # If we have reached the beginning of the file, return an empty page
    if end <= 0:
        return {
            "items": [],
            "next_cursor": 0,
            "has_more": False,
            "file_size": file_size,
        }

    with open(path, "rb") as fp:
        pos = end
        chunks: List[bytes] = []
        newline_count = 0

        # Read from the end of the file backwards in blocks until we have enough lines for the page
        while pos > 0 and newline_count <= page_size:
            read_size = min(block_size, pos)
            pos -= read_size
            fp.seek(pos)
            chunk = fp.read(read_size)
            chunks.insert(0, chunk)
            newline_count += chunk.count(b"\n")

    data_start = pos
    segment = b"".join(chunks)

    # Drop the first line in case it is incomplete
    if data_start > 0:
        first_newline = segment.find(b"\n")
        # If there is no newline in the segment,
        # it means the entire segment is a single line that is incomplete, so we return an empty page
        if first_newline == -1:
            return {
                "items": [],
                "next_cursor": 0,
                "has_more": False,
                "file_size": file_size,
            }
        data_start += first_newline + 1
        segment = segment[first_newline + 1 :]

    # If the segment ends with a newline,
    # remove it to avoid returning an empty line at the end of the page
    if segment.endswith(b"\n"):
        segment = segment[:-1]

    # If the segment is empty after removing the first line and the last newline,
    # it means there are no complete lines to return, so we return an empty page
    if not segment:
        return {
            "items": [],
            "next_cursor": 0,
            "has_more": False,
            "file_size": file_size,
        }

    # Get the line offsets for each line in the segment, relative to the start of the file
    line_bytes = segment.split(b"\n")
    offsets: List[int] = []
    offset = data_start
    for raw_line in line_bytes:
        offsets.append(offset)
        offset += len(raw_line) + 1

    # Get page_size number of lines
    # Starts from the end of the segment (last line)
    selected_start = max(0, len(line_bytes) - page_size)
    selected_lines = line_bytes[selected_start:]
    selected_offsets = offsets[selected_start:]
    next_cursor = selected_offsets[0] if selected_offsets else 0

    # Decode the selected lines to get the actual log lines and build the items list with metadata
    items = []
    for line_offset, raw_line in zip(selected_offsets, selected_lines, strict=True):
        line = raw_line.decode("utf-8", errors="replace")
        items.append(
            {
                "id": str(line_offset),
                "offset": line_offset,
                "timestamp": _extract_log_timestamp(line),
                "raw": line,
            }
        )

    # Return the page of log lines along with the next cursor and file size
    return {
        "items": items,
        "next_cursor": next_cursor,
        "has_more": next_cursor > 0,
        "file_size": file_size,
    }


@api.route("/node/start", methods=["POST"])
def start_node():
    try:
        node_args, background = _node_args_from_request()

        node_process_manager.start(
            node_args=node_args,
            background=background,
            actor=_actor_from_request(),
            reason="start_requested_from_gui",
        )

        return response(
            {
                "state": node_process_manager.get_status().value,
                "node_args": node_args,
            }
        ), 200

    except FedbiomedError as e:
        return error(f"Could not start node process: {e}"), 500


@api.route("/node/stop", methods=["POST"])
def stop_node():
    """Stop the node process from GUI."""
    try:
        node_process_manager.stop(
            actor=_actor_from_request(),
            reason="stop_requested_from_gui",
        )

        return response(
            {
                "state": node_process_manager.get_status().value,
            }
        ), 200

    except FedbiomedError as e:
        return error(f"Could not stop node process: {e}"), 500


@api.route("/node/restart", methods=["POST"])
def restart_node():
    try:
        node_args, background = _node_args_from_request()

        node_process_manager.restart(
            node_args=node_args,
            background=background,
            actor=_actor_from_request(),
            reason="restart_requested_from_gui",
        )

        return response(
            {
                "state": node_process_manager.get_status().value,
                "node_args": node_args,
            }
        ), 200

    except Exception as e:
        return error(f"Could not restart node process: {e}"), 500


@api.route("/node/status", methods=["GET"])
def node_status():
    """Return current node process status."""
    try:
        return response(
            {
                "state": node_process_manager.get_status().value,
                "node_id": config.node_config.get("default", "id"),
                "node_name": config.node_config.get("default", "name"),
            }
        ), 200

    except FedbiomedError as e:
        return error(f"Could not get node process status: {e}"), 500


@api.route("/node/process-state", methods=["GET"])
def node_process_state():
    """Return current persisted node process state."""
    try:
        process_state = node_process_manager.get_process_state().to_dict()
        process_state.update(_config_modification_status(process_state))
        return response(process_state), 200

    except FedbiomedError as e:
        return error(f"Could not get node process state: {e}"), 500


@api.route("/node/logs", methods=["GET"])
def node_logs():
    """Return raw application log lines for the active node."""
    page_size, valid_page_size = _parse_positive_int(
        request.args.get("page_size"),
        default=100,
        maximum=2000,
    )
    if not valid_page_size:
        return error("Invalid 'page_size'"), 400

    log_name = request.args.get("file") or _APPLICATION_LOG_BASENAME
    try:
        path = _application_log_path(log_name)
    except FedbiomedError as e:
        return error(str(e)), 400

    if not os.path.isfile(path):
        return (
            response(
                {
                    "items": [],
                    "next_cursor": 0,
                    "has_more": False,
                    "file_size": 0,
                    "page_size": page_size,
                    "file": log_name,
                }
            ),
            200,
        )

    try:
        page = _read_application_log_page(
            path,
            cursor=request.args.get("cursor"),
            page_size=page_size,
        )
    except FedbiomedError as e:
        return error(str(e)), 400

    return response(
        {
            **page,
            "page_size": page_size,
            "file": log_name,
        }
    ), 200


@api.route("/node/logs/files", methods=["GET"])
def node_log_files():
    """Return application log files available for display or download."""
    return response({"files": _application_log_files()}), 200


@api.route("/node/logs/download", methods=["GET"])
def download_node_logs():
    """Download the raw application log file for the active node."""
    log_name = request.args.get("file") or _APPLICATION_LOG_BASENAME
    try:
        path = _application_log_path(log_name)
    except FedbiomedError as e:
        return error(str(e)), 400

    if not os.path.isfile(path):
        return error("Application log file does not exist"), 404

    return send_file(
        path,
        as_attachment=True,
        download_name=log_name,
        mimetype="text/plain",
    )

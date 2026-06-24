import os
import re
from typing import Any, Dict, List, Optional, Tuple

from flask import request, send_file
from flask_jwt_extended import get_jwt

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.node_pm import NodeProcessManager

from ..config import config
from ..utils import error, response
from .api import api

node_process_manager = NodeProcessManager(config.node_config)

_APPLICATION_LOG_BASENAME = "application.log"
_LOG_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def _actor_from_request() -> dict:
    """Build GUI actor metadata without enforcing role authorization.

    Returns:
        Dictionary containing actor metadata extracted from JWT claims.
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
        return response(node_process_manager.get_process_state().to_dict()), 200

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

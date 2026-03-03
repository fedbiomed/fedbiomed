import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from flask import request

from ..config import config
from ..helpers.auth_helpers import admin_required
from ..utils import error, response
from . import api

_SECURITY_LOG_BASENAME = "security_audit.log"


def _security_log_dir() -> str:
    return os.path.join(config["NODE_FEDBIOMED_ROOT"], "log")


def _list_security_log_files() -> List[Dict[str, Any]]:
    log_dir = _security_log_dir()
    try:
        entries = os.listdir(log_dir)
    except FileNotFoundError:
        return []

    files = []
    for name in entries:
        if not name.startswith(_SECURITY_LOG_BASENAME):
            continue
        full = os.path.join(log_dir, name)
        if not os.path.isfile(full):
            continue
        try:
            st = os.stat(full)
            files.append(
                {
                    "name": name,
                    "size": int(st.st_size),
                    "mtime": int(st.st_mtime),
                }
            )
        except OSError:
            continue

    files.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return files


def _file_date_from_name(name: str) -> Optional[datetime]:
    """Extract a file's logical date (UTC midnight) from its rotated filename.

    Supports: security_audit.log.YYYY-MM-DD
    Returns a timezone-aware datetime at 00:00:00 in *local timezone* (matching
    typical rotation behavior) or None if not parseable.
    """

    prefix = _SECURITY_LOG_BASENAME + "."
    if not name.startswith(prefix):
        return None

    suffix = name[len(prefix) :]
    # Some logrotate setups may append extra segments; we only care about the date.
    date_part = suffix.split(".", 1)[0]
    try:
        dt = datetime.strptime(date_part, "%Y-%m-%d")
        local_tz = datetime.now().astimezone().tzinfo
        return dt.replace(tzinfo=local_tz)
    except Exception:
        return None


def _resolve_security_log_path(file_name: Optional[str]) -> str:
    file_name = file_name or _SECURITY_LOG_BASENAME

    # Prevent path traversal: only accept basenames that appear in the log dir listing.
    allowed = {f["name"] for f in _list_security_log_files()}
    if file_name not in allowed:
        raise ValueError("Invalid log file")

    return os.path.join(_security_log_dir(), file_name)


def _iter_security_log_paths(
    *, start_dt: Optional[datetime] = None, end_dt: Optional[datetime] = None
) -> List[str]:
    """Returns relevant security log file paths in newest-first order.

    When a date range is provided, it selects rotated files by the date embedded
    in the filename (security_audit.log.YYYY-MM-DD) and avoids scanning unrelated
    days, which also fixes empty results caused by early stopping on newer files.
    """

    files = _list_security_log_files()
    if not files:
        return []

    # Convert requested UTC instants into local dates to match rotated filename
    # date semantics (typically local-time based).
    start_date = (
        start_dt.astimezone().date() if isinstance(start_dt, datetime) else None
    )
    end_date = end_dt.astimezone().date() if isinstance(end_dt, datetime) else None

    today_local = datetime.now().astimezone().date()
    include_current = True
    if start_date and today_local < start_date:
        include_current = False
    if end_date and today_local > end_date:
        include_current = False

    selected: List[Dict[str, Any]] = []
    for f in files:
        name = f.get("name")
        if not name:
            continue

        if name == _SECURITY_LOG_BASENAME:
            if include_current:
                selected.append(f)
            continue

        file_dt = _file_date_from_name(name)
        if file_dt is None:
            # Unknown naming; include conservatively.
            selected.append(f)
            continue

        file_date = file_dt.date()
        if start_date and file_date < start_date:
            continue
        if end_date and file_date > end_date:
            continue

        selected.append(f)

    # Keep newest-first based on mtime (already sorted), but filtered.
    return [os.path.join(_security_log_dir(), f["name"]) for f in selected]


def _tail_lines(path: str, max_lines: int, block_size: int = 8192) -> List[str]:
    if max_lines <= 0:
        return []

    try:
        with open(path, "rb") as fp:
            fp.seek(0, os.SEEK_END)
            pos = fp.tell()

            buffer = b""
            lines: List[bytes] = []

            while pos > 0 and len(lines) <= max_lines:
                read_size = block_size if pos >= block_size else pos
                pos -= read_size
                fp.seek(pos)
                chunk = fp.read(read_size)

                buffer = chunk + buffer
                parts = buffer.split(b"\n")

                # Keep the first part as the new buffer (may be a partial line)
                buffer = parts[0]
                if len(parts) > 1:
                    # parts[1:] are complete lines
                    lines = parts[1:] + lines

            if buffer:
                lines = [buffer] + lines

            decoded = [ln.decode("utf-8", errors="replace").strip() for ln in lines]
            decoded = [ln for ln in decoded if ln]
            return decoded[-max_lines:]
    except FileNotFoundError:
        return []


def _parse_json_lines(lines: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for line in lines:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            items.append(obj)
    return items


FilterValue = Union[str, datetime, None]


def _item_to_search_text(item: Dict[str, Any]) -> str:
    """Build a search string for 'contains' filtering.

    Includes both JSON formatting ("k": "v") and a human-oriented
    key=value formatting (k=v) that matches the GUI Details column.
    """

    try:
        as_json = json.dumps(item, ensure_ascii=False, sort_keys=True)
    except Exception:
        as_json = str(item)

    parts: List[str] = []
    try:
        for k, v in item.items():
            if v is None:
                parts.append(f"{k}=")
            elif isinstance(v, str):
                parts.append(f"{k}={v}")
            else:
                try:
                    parts.append(
                        f"{k}={json.dumps(v, ensure_ascii=False, sort_keys=True)}"
                    )
                except Exception:
                    parts.append(f"{k}={v}")
    except Exception:
        parts = []

    as_kv = " | ".join(parts)
    return f"{as_json}\n{as_kv}" if as_kv else as_json


def _matches_filters(item: Dict[str, Any], filters: Dict[str, FilterValue]) -> bool:
    operation = filters.get("operation")
    if operation and str(item.get("operation")) != operation:
        return False

    status = filters.get("status")
    if status and str(item.get("status")) != status:
        return False

    researcher_id = filters.get("researcher_id")
    if researcher_id:
        if researcher_id == "__none__":
            val = item.get("researcher_id")
            # JSONL may encode missing researcher_id as null, empty string, or string-like null.
            if val is not None and str(val) not in ("", "None", "null"):
                return False
        elif str(item.get("researcher_id")) != researcher_id:
            return False

    contains = filters.get("contains")
    if contains:
        needle = str(contains).strip().lower()
        if not needle:
            # Treat whitespace-only queries as "no filter".
            needle = ""

        candidate = _item_to_search_text(item)
        if needle and needle not in candidate.lower():
            return False

    start_dt = filters.get("start_dt")
    end_dt = filters.get("end_dt")
    if start_dt or end_dt:
        item_dt = _parse_timestamp(item.get("timestamp"))
        if item_dt is None:
            return False

        if isinstance(start_dt, datetime):
            if item_dt < start_dt:
                return False

        if isinstance(end_dt, datetime):
            if item_dt > end_dt:
                return False

    return True


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse timestamps from security audit logs.

    Supported inputs:
      - ISO 8601 strings, with or without timezone ("Z" supported)
      - epoch seconds or milliseconds (int/float or numeric string)

    Returns timezone-aware datetime in UTC, or None when unparseable.
    """

    if value is None:
        return None

    # Epoch number (seconds or milliseconds)
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            # Heuristic: >1e12 is likely ms
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    s = str(value).strip()
    if not s:
        return None

    # Numeric string epoch
    if s.isdigit():
        try:
            ts = float(s)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    # ISO8601 string
    try:
        # Handle trailing Z (UTC)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            # Treat naive datetimes as UTC to keep semantics stable.
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@api.route("/admin/security/log-files", methods=["GET"])
@admin_required
def list_security_log_files():
    """Lists available security audit log files under <node_root>/log."""
    return response(_list_security_log_files()), 200


@api.route("/admin/security/logs", methods=["GET"])
@admin_required
def get_security_logs():
    """Returns recent security log entries from security audit logs (JSONL).

    Query args:
        file: optional file name from /log-files (advanced/debug)
        limit: number of items to return (default 200, max 2000)
        skip: skip newest N matching items (for pagination)
        operation/status/researcher_id: exact match filters
        contains: substring match on entry content (any field)
        start_ts/end_ts: ISO 8601 or epoch seconds/ms (inclusive range)

    Returns:
        {"items": [...], "next_skip": int, "files": [str, ...]}
    """

    file_name = request.args.get("file")

    try:
        limit = int(request.args.get("limit", 200))
    except Exception:
        return error("Invalid 'limit'"), 400

    max_total_arg = request.args.get("max_total")
    max_total: Optional[int] = None
    if max_total_arg not in (None, ""):
        try:
            max_total = int(max_total_arg)
        except Exception:
            return error("Invalid 'max_total'"), 400
        if max_total <= 0:
            max_total = None

    try:
        skip = int(request.args.get("skip", 0))
    except Exception:
        return error("Invalid 'skip'"), 400

    limit = max(1, min(limit, 2000))
    skip = max(0, skip)

    filters: Dict[str, FilterValue] = {
        "operation": request.args.get("operation"),
        "status": request.args.get("status"),
        "researcher_id": request.args.get("researcher_id"),
        "contains": request.args.get("contains"),
        "start_dt": None,
        "end_dt": None,
    }

    start_ts = request.args.get("start_ts")
    end_ts = request.args.get("end_ts")

    # Validate date range args early (bad inputs should be 400) and parse once.
    if start_ts:
        parsed_start = _parse_timestamp(start_ts)
        if parsed_start is None:
            return error("Invalid 'start_ts'"), 400
        filters["start_dt"] = parsed_start

    if end_ts:
        parsed_end = _parse_timestamp(end_ts)
        if parsed_end is None:
            return error("Invalid 'end_ts'"), 400
        filters["end_dt"] = parsed_end

    if isinstance(filters.get("start_dt"), datetime) and isinstance(
        filters.get("end_dt"), datetime
    ):
        if filters["start_dt"] > filters["end_dt"]:
            return error("Invalid date range: start_ts is after end_ts"), 400

    # If a date range is requested and no max_total was provided, cap by default.
    # This keeps the UI responsive for large intervals but remains adjustable.
    has_date_filter = isinstance(filters.get("start_dt"), datetime) or isinstance(
        filters.get("end_dt"), datetime
    )
    if has_date_filter and max_total is None:
        max_total = 5000

    # Enforce max_total for pagination.
    if max_total is not None:
        if skip >= max_total:
            return response({"items": [], "next_skip": max_total, "files": []}), 200
        # Do not return more than remaining budget.
        limit = min(limit, max_total - skip)

    paths: List[str] = []
    if file_name:
        try:
            paths = [_resolve_security_log_path(file_name)]
        except ValueError:
            return error("Invalid log file"), 400
    else:
        if not has_date_filter:
            # Default: only current file (fast path).
            try:
                paths = [_resolve_security_log_path(None)]
            except ValueError:
                paths = []
        else:
            paths = _iter_security_log_paths(
                start_dt=filters.get("start_dt")
                if isinstance(filters.get("start_dt"), datetime)
                else None,
                end_dt=filters.get("end_dt")
                if isinstance(filters.get("end_dt"), datetime)
                else None,
            )

    if not paths:
        return response({"items": [], "next_skip": 0, "files": []}), 200

    # Read tail windows big enough for basic filtering + pagination without DB.
    # If filters are restrictive, users can increase limit or use a smaller range.
    window = min(50000, max(5000, skip + (limit * 5)))
    items: List[Dict[str, Any]] = []
    scanned_files: List[str] = []
    for path in paths:
        scanned_files.append(os.path.basename(path))
        lines = _tail_lines(path, window)
        items.extend(_parse_json_lines(lines))

        # Stop early once we have a decent pool for pagination.
        if len(items) >= (skip + (limit * 10)):
            break

    # Sort newest-first by parsed timestamp; unparseable timestamps go last.
    min_dt = datetime.min.replace(tzinfo=timezone.utc)
    decorated = [(_parse_timestamp(it.get("timestamp")) or min_dt, it) for it in items]
    decorated.sort(key=lambda x: x[0], reverse=True)
    items = [it for _, it in decorated]
    items = [it for it in items if _matches_filters(it, filters)]

    page = items[skip : skip + limit]
    next_skip = skip + len(page)

    return response(
        {"items": page, "next_skip": next_skip, "files": scanned_files}
    ), 200

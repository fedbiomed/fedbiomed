import json
import os
from typing import Any, Dict, List, Optional

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


def _resolve_security_log_path(file_name: Optional[str]) -> str:
    file_name = file_name or _SECURITY_LOG_BASENAME

    # Prevent path traversal: only accept basenames that appear in the log dir listing.
    allowed = {f["name"] for f in _list_security_log_files()}
    if file_name not in allowed:
        raise ValueError("Invalid log file")

    return os.path.join(_security_log_dir(), file_name)


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


def _matches_filters(item: Dict[str, Any], filters: Dict[str, Optional[str]]) -> bool:
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
        msg = str(item.get("message", ""))
        if contains.lower() not in msg.lower():
            return False

    return True


@api.route("/admin/security/log-files", methods=["GET"])
@admin_required
def list_security_log_files():
    """Lists available security audit log files under <node_root>/log."""
    return response(_list_security_log_files()), 200


@api.route("/admin/security/logs", methods=["GET"])
@admin_required
def get_security_logs():
    """Returns recent security log entries from a selected file (JSONL).

    Query args:
        file: file name from /log-files (defaults to current security_audit.log)
        limit: number of items to return (default 200, max 2000)
        skip: skip newest N matching items (for pagination)
        operation/status/researcher_id: exact match filters
        contains: substring match on message

    Returns:
        {"items": [...], "next_skip": int, "file": str}
    """

    file_name = request.args.get("file")

    try:
        limit = int(request.args.get("limit", 200))
    except Exception:
        return error("Invalid 'limit'"), 400

    try:
        skip = int(request.args.get("skip", 0))
    except Exception:
        return error("Invalid 'skip'"), 400

    limit = max(1, min(limit, 2000))
    skip = max(0, skip)

    filters = {
        "operation": request.args.get("operation"),
        "status": request.args.get("status"),
        "researcher_id": request.args.get("researcher_id"),
        "contains": request.args.get("contains"),
    }

    try:
        path = _resolve_security_log_path(file_name)
    except ValueError:
        return error("Invalid log file"), 400

    # Read a tail window big enough for basic filtering + pagination without DB.
    # If filters are restrictive, users can increase limit or use a smaller file/day.
    window = min(50000, max(5000, skip + (limit * 5)))
    lines = _tail_lines(path, window)
    items = _parse_json_lines(lines)

    # Sort newest-first by timestamp if available; fallback to original order.
    def _ts(it: Dict[str, Any]) -> str:
        return str(it.get("timestamp", ""))

    items.sort(key=_ts, reverse=True)
    items = [it for it in items if _matches_filters(it, filters)]

    page = items[skip : skip + limit]
    next_skip = skip + len(page)

    return response(
        {"items": page, "next_skip": next_skip, "file": os.path.basename(path)}
    ), 200

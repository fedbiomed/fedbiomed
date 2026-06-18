import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import request
from flask_jwt_extended import get_jwt

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.node_pm import NodeProcessManager

from ..config import config
from ..utils import error, response
from .api import api
from .log_utils import (
    FilterValue,
    parse_timestamp,
)

node_process_manager = NodeProcessManager(config.node_config)

_APPLICATION_LOG_BASENAME = "application.log"
_APPLICATION_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
_LOG_LINE_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+"
    r"(?P<logger>\S+)\s+"
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+"
    r"(?:\[(?P<caller>[^\]]+)\]\s+)?-\s"
    r"(?P<message>.*)$"
)


def _actor_from_request() -> dict:
    """Build GUI actor metadata without enforcing role authorization."""
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
    return os.path.join(config["NODE_FEDBIOMED_ROOT"], "log")


def _parse_application_log_lines(lines: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for line in lines:
        match = _LOG_LINE_RE.match(line)
        if match:
            parsed_dt = parse_timestamp(
                match.group("timestamp"),
                extra_formats=(_APPLICATION_LOG_TIMESTAMP_FORMAT,),
            )
            current = {
                "timestamp": parsed_dt.isoformat() if parsed_dt else None,
                "level": match.group("level"),
                "logger": match.group("logger"),
                "caller": match.group("caller") or "",
                "message": match.group("message"),
                "raw": line,
            }
            items.append(current)
            continue

        if current is not None:
            current["message"] = f"{current['message']}\n{line}"
            current["raw"] = f"{current['raw']}\n{line}"
        elif line:
            items.append(
                {
                    "timestamp": None,
                    "level": "",
                    "logger": "",
                    "caller": "",
                    "message": line,
                    "raw": line,
                }
            )

    return items


def _matches_log_filters(item: Dict[str, Any], filters: Dict[str, FilterValue]) -> bool:
    level = filters.get("level")
    if level and str(item.get("level")) != level:
        return False

    contains = filters.get("contains")
    if contains:
        needle = str(contains).strip().lower()
        candidate = "\n".join(
            str(item.get(k) or "")
            for k in ("timestamp", "level", "logger", "caller", "message", "raw")
        )
        if needle and needle not in candidate.lower():
            return False

    start_dt = filters.get("start_dt")
    end_dt = filters.get("end_dt")
    if start_dt or end_dt:
        item_dt = parse_timestamp(item.get("timestamp"))
        if item_dt is None:
            return False

        if isinstance(start_dt, datetime) and item_dt < start_dt:
            return False
        if isinstance(end_dt, datetime) and item_dt > end_dt:
            return False

    return True


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

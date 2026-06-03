from flask import request
from flask_jwt_extended import get_jwt

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.node_pm import NodeProcessManager

from ..config import config
from ..utils import error, response
from .api import api

node_process_manager = NodeProcessManager(config.node_config)


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


@api.route("/v1/node/start", methods=["POST"])
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


@api.route("/v1/node/stop", methods=["POST"])
def stop_node():
    """Stop the node process from GUI."""
    try:
        node_args, background = _node_args_from_request()

        node_process_manager.stop(
            node_args=node_args,
            background=background,
            actor=_actor_from_request(),
            reason="stop_requested_from_gui",
        )

        return response(
            {
                "state": node_process_manager.get_status().value,
                "node_args": node_args,
            }
        ), 200

    except FedbiomedError as e:
        return error(f"Could not stop node process: {e}"), 500


@api.route("/v1/node/restart", methods=["POST"])
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


@api.route("/v1/node/status", methods=["GET"])
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


@api.route("/v1/node/process-state", methods=["GET"])
def node_process_state():
    """Return current persisted node process state."""
    try:
        return response(node_process_manager._get_process_state().to_dict()), 200

    except FedbiomedError as e:
        return error(f"Could not get node process state: {e}"), 500

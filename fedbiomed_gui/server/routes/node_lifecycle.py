from flask import request
from flask_jwt_extended import get_jwt

from fedbiomed.node.node_pm import NodeProcessManager

from ..config import config
from ..helpers.auth_helpers import admin_required
from ..schemas import NodeLifecycleRequest
from ..utils import error, response, validate_request_data
from .api import api

node_process_manager = NodeProcessManager(config.node_config)


def _actor_from_jwt() -> dict:
    """Build node lifecycle actor metadata from the current JWT claims."""
    claims = get_jwt()
    return {
        "source": "gui",
        "user_id": claims.get("sub"),
        "email": claims.get("email"),
        "role": claims.get("role"),
        "name": claims.get("name"),
        "surname": claims.get("surname"),
    }


def _node_args_from_request() -> dict:
    """Build node startup arguments from a validated request body."""
    req = request.json or {}
    return {
        "gpu": req.get("gpu", False) is True or req.get("gpu_only", False) is True,
        "gpu_num": req.get("gpu_num", 1),
        "gpu_only": req.get("gpu_only", False) is True,
        "debug": req.get("debug", False) is True,
    }


@api.route("/node/lifecycle/status", methods=["GET"])
@admin_required
def node_lifecycle_status():
    """Return the current persisted node lifecycle status."""
    try:
        return response(node_process_manager.get_process_state()), 200
    except Exception as e:
        return error(f"Can not get node lifecycle status: {e}"), 400


@api.route("/node/lifecycle/start", methods=["POST"])
@admin_required
@validate_request_data(schema=NodeLifecycleRequest)
def node_lifecycle_start():
    """Start the managed node process."""
    try:
        actor = _actor_from_jwt()
        node_process_manager.start(_node_args_from_request(), actor=actor)
        return response(
            node_process_manager.get_process_state(),
            "Node start request has been processed",
        ), 200
    except Exception as e:
        return error(f"Can not start node: {e}"), 400


@api.route("/node/lifecycle/stop", methods=["POST"])
@admin_required
def node_lifecycle_stop():
    """Stop the managed node process."""
    try:
        actor = _actor_from_jwt()
        node_process_manager.stop(actor=actor)
        return response(
            node_process_manager.get_process_state(),
            "Node stop request has been processed",
        ), 200
    except Exception as e:
        return error(f"Can not stop node: {e}"), 400


@api.route("/node/lifecycle/restart", methods=["POST"])
@admin_required
@validate_request_data(schema=NodeLifecycleRequest)
def node_lifecycle_restart():
    """Restart the managed node process."""
    try:
        actor = _actor_from_jwt()
        node_process_manager.restart(_node_args_from_request(), actor=actor)
        return response(
            node_process_manager.get_process_state(),
            "Node restart request has been processed",
        ), 200
    except Exception as e:
        return error(f"Can not restart node: {e}"), 400

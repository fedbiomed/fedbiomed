from fedbiomed.common.logger import logger
from fedbiomed.node.node_process_controller import NodeStatus, nodeProcessController

from ..config import config
from ..utils import error, response
from .api import api


def _default_node_args() -> dict:
    """Default node arguments used by the draft GUI management routes."""

    return {
        "gpu": False,
        "gpu_num": 1,
        "gpu_only": False,
        "debug": False,
    }


@api.route("/node/start", methods=["POST"])
def start_node_backend():
    """Draft endpoint for starting the node backend."""

    try:
        nodeProcessController.start(config.node_config, _default_node_args())
        return response({}, "Node start command sent"), 200
    except Exception as exc:
        return error(f"Unable to start node: {exc}"), 500


@api.route("/node/stop", methods=["POST"])
def stop_node_backend():
    """Draft endpoint for stopping the node backend."""

    try:
        nodeProcessController.stop()
        return response({}, "Node stop command sent"), 200
    except Exception as exc:
        return error(f"Unable to stop node: {exc}"), 500


@api.route("/node/restart", methods=["POST"])
def restart_node_backend():
    """Draft endpoint for restarting the node backend."""

    try:
        nodeProcessController.restart(config.node_config, _default_node_args())
        return response({}, "Node restart command sent"), 200
    except Exception as exc:
        return error(f"Unable to restart node: {exc}"), 500


@api.route("/node/status", methods=["GET"])
def node_backend_status():
    """Draft endpoint for retrieving the node backend status."""

    try:
        status = nodeProcessController.get_status()
        if not isinstance(status, NodeStatus):
            logger.warning(
                f"nodeProcessController.get_status() returned unexpected node status type: {type(status)}. Defaulting to UNKNOWN."
            )
            status = NodeStatus.UNKNOWN

        return response({"status": status.value}, "Node status retrieved"), 200
    except Exception as exc:
        return error(f"Unable to get node status: {exc}"), 500

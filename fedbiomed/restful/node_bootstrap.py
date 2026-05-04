"""Helpers for starting and stopping a managed node from the GUI process."""

import json
import os

from fedbiomed.common.logger import logger


def _default_node_args() -> dict:
    """Return default node startup arguments."""
    return {
        "gpu": False,
        "gpu_num": None,
        "gpu_only": False,
        "debug": False,
    }


def _get_node_process_manager():
    """Return the shared node process manager."""
    from fedbiomed.node.node_pm import node_process_manager

    return node_process_manager


def _should_start_node_with_gui() -> bool:
    """Return whether the GUI process should manage the node lifecycle."""
    return os.getenv("FBM_START_NODE_WITH_GUI", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def load_node_args_from_env() -> dict:
    """Load node startup arguments from the environment."""
    raw_node_args = os.getenv("FBM_NODE_START_ARGS")
    if not raw_node_args:
        return _default_node_args()

    try:
        node_args = json.loads(raw_node_args)
    except json.JSONDecodeError:
        logger.warning("Could not decode FBM_NODE_START_ARGS. Using default node args.")
        return _default_node_args()

    if not isinstance(node_args, dict):
        logger.warning(
            "FBM_NODE_START_ARGS must decode to a dict. Using default node args."
        )
        return _default_node_args()

    default_args = _default_node_args()
    default_args.update(
        {key: value for key, value in node_args.items() if key in default_args}
    )
    return default_args


def start_node_for_gui(node_config) -> None:
    """Start the managed node when the GUI server starts."""
    if not _should_start_node_with_gui():
        return

    _get_node_process_manager().start(
        node_config,
        load_node_args_from_env(),
        actor={"source": "gui"},
    )


def stop_node_for_gui() -> None:
    """Stop the managed node when the GUI server stops."""
    if not _should_start_node_with_gui():
        return

    _get_node_process_manager().stop(actor={"source": "gui"}, reason="gui_stopped")

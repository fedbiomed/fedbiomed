# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Node process lifecycle manager."""

import enum
import getpass
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from types import FrameType
from typing import Any, Dict, Optional, Union

import psutil

from fedbiomed.common.constants import CONFIG_FOLDER_NAME, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import DEFAULT_APPLICATION_LOG_FILE, logger
from fedbiomed.node.config import NodeConfig
from fedbiomed.node.dataset_manager._db_dataclasses import NodeProcessStateEntry
from fedbiomed.node.dataset_manager._db_tables import (
    NodeProcessStateHistoryTable,
    NodeProcessStateTable,
)
from fedbiomed.node.node import Node

DEFAULT_NODE_ARGS = {
    "gpu": False,
    "gpu_num": 1,
    "gpu_only": False,
    "debug": False,
}


class NodeState(enum.Enum):
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    UNKNOWN = "unknown"  # used only in route layer as a fallback


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _node_signal_trigger_term() -> None:
    """Triggers a TERM signal to the current process."""
    os.kill(os.getpid(), signal.SIGTERM)


def _start_node_process(config_path: str, node_args: Union[str, dict]) -> None:
    """Start the node runtime inside the managed subprocess.

    Args:
        config_path: Path to the node configuration directory.
        node_args: Arguments for the node.
    """
    config = NodeConfig(root=config_path)
    node_args = json.loads(node_args) if isinstance(node_args, str) else node_args

    _node = Node(config, node_args)

    logger.info(str(_node))
    logger.info(f"Node name: {_node.config.get('default', 'name')}")

    def _node_signal_handler(signum: int, frame: Union[FrameType, None]):
        """Signal handler that terminates the process.

        Args:
            signum: Signal number received.
            frame: Frame object received. Currently unused

        Raises:
           SystemExit: Always raised.
        """
        try:
            if _node and _node.is_connected():
                logger.security_event(
                    operation="node_stopped",
                    status="success",
                    researcher_id=None,
                    node_name=_node.node_name,
                    reason="signal_received",
                    signal_number=signum,
                )

                _node.send_error(
                    ErrorNumbers.FB312, extra_msg="Node is stopped", broadcast=True
                )
                time.sleep(2)
                logger.critical(
                    "Node stopped in signal_handler, probably node exit on error or user decision (Ctrl C)"
                )
            else:
                logger.info(
                    "Cannot send error message to researcher (node not initialized yet)"
                )
                logger.info(
                    "Node stopped in signal_handler, probably node exit on error or user decision (Ctrl C)"
                )
        finally:
            time.sleep(0.5)
            sys.exit(signum)

    if getattr(_node, "_debug", False):
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    try:
        signal.signal(signal.SIGTERM, _node_signal_handler)
        signal.signal(signal.SIGINT, _node_signal_handler)
        logger.info("Launching node...")

        if _node.config.getbool("security", "training_plan_approval"):
            _node.tp_security_manager.check_hashes_for_registered_training_plans()
            if _node.config.getbool("security", "allow_default_training_plans"):
                logger.info("Loading default training plans")
                _node.tp_security_manager.register_update_default_training_plans()
        else:
            logger.warning(
                "Training plan approval for train request is not activated. "
                + "This might cause security problems. Please, consider to enable training plan approval."
            )

        logger.info("Starting communication channel with network")

        _node.start_messaging(_node_signal_trigger_term)
        logger.info("Starting node to node router")
        _node.start_protocol()
        logger.info("Starting task manager")
        _node.task_manager()

    except FedbiomedError as exp:
        logger.security_event(
            operation="node_stopped",
            status="error",
            researcher_id=None,
            node_name=_node.node_name if _node else None,
            reason="fedbiomed_error",
            error_message=str(exp),
        )
        logger.critical(f"Node stopped. {exp}")

    except Exception as exp:
        logger.security_event(
            operation="node_stopped",
            status="error",
            researcher_id=None,
            node_name=_node.node_name if _node else None,
            reason="unexpected_exception",
            error_message=str(exp),
            exception_type=type(exp).__name__,
        )
        _node.send_error(ErrorNumbers.FB300, extra_msg="Error = " + str(exp))
        logger.critical(f"Node stopped. {exp}")


class NodeProcessManager:
    """Manages a single node subprocess. This node subprocess is the one where whose config is passed during initialization."""

    def __init__(self, config: NodeConfig) -> None:
        """Initialize the NodeProcessManager with the given configuration.

        Args:
            config: Node configuration object that the manager will use to spawn the node subprocess and manage its state.
        """
        self._config = config
        self._node_id: str | None = config.get("default", "id")
        self._node_name: str | None = config.get("default", "name")
        self._cleanup_process_state_history(
            days=30
        )  # Clean up old history entries on initialization

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state_table(self) -> NodeProcessStateTable:
        """Get the state table instance."""
        db_path = os.path.abspath(
            os.path.join(
                self._config.root,
                CONFIG_FOLDER_NAME,
                self._config.get("default", "db"),
            )
        )
        return NodeProcessStateTable(db_path)

    def _get_history_table(self) -> NodeProcessStateHistoryTable:
        """Get the history table instance."""
        db_path = os.path.abspath(
            os.path.join(
                self._config.root,
                CONFIG_FOLDER_NAME,
                self._config.get("default", "db"),
            )
        )
        return NodeProcessStateHistoryTable(db_path)

    def _cleanup_process_state_history(self, days: int = 30) -> None:
        """Remove process-state history entries older than the given number of days.

        Args:
            days: Number of days to retain. Defaults to 30.
        """
        cutoff = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
            days=days
        )

        try:
            entries = self._get_history_table().all()
            for entry in entries:
                updated_at = entry.get("updated_at")
                if not updated_at:
                    continue

                try:
                    entry_date = datetime.fromisoformat(
                        updated_at.replace("Z", "+00:00")
                    )
                except ValueError:
                    logger.warning(
                        f"Could not parse process-state history timestamp: {updated_at}"
                    )
                    continue

                if entry_date < cutoff:
                    self._get_history_table().remove(doc_ids=[entry.doc_id])

        except Exception as e:
            logger.warning(f"Could not clean up old node process history entries: {e}")

    @staticmethod
    def _build_actor(actor: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a sanitized actor dictionary for process state attribution.

        Args:
            actor: Dictionary containing actor information. If None, it will generate a default actor with local source and username.
        Returns:
            A sanitized dictionary with allowed actor fields and defaults.
        """
        base = {
            "source": "local",
            "local_username": getpass.getuser(),
        }
        if not actor:
            return base

        allowed = {
            "source",
            "user_id",
            "email",
            "role",
            "name",
            "surname",
            "local_username",
        }
        base.update({key: value for key, value in actor.items() if key in allowed})
        return base

    def _set_process_state(
        self,
        *,
        pid: int,
        state: NodeState,
        action: str,
        actor: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        exit_code: Optional[int] = None,
        node_args: Optional[Dict[str, Any]] = None,
        background: Optional[bool] = None,
    ) -> None:
        """Persist the current process state if the node DB is available.

        Args:
            state: The new state of the node.
            action: The action that triggered the state change.
            actor: Optional user/source metadata for process state attribution.
            reason: Optional reason for the state change.
            exit_code: Optional exit code for the process.
            node_args: Effective arguments used to start the node process.
            background: Whether the node process runs in the background.
        """
        if (
            not self._get_state_table()
            or not self._get_history_table()
            or not self._node_id
            or not self._node_name
        ):
            return

        try:
            now = _utc_now()
            existing = self._get_state_table().get_by_id(self._node_id) or {}
            effective_node_args = existing.get("node_args")
            if node_args is not None:
                effective_node_args = (
                    dict(effective_node_args)
                    if isinstance(effective_node_args, dict)
                    else {}
                )
                effective_node_args.update(node_args)

            entry = NodeProcessStateEntry(
                pid=pid,
                state=state.value,
                node_id=self._node_id,
                node_name=self._node_name,
                action=action,
                reason=reason,
                actor=self._build_actor(actor),
                updated_at=now,
                started_at=existing.get("started_at"),
                stopped_at=existing.get("stopped_at"),
                exit_code=exit_code,
                node_args=effective_node_args,
                background=(
                    background if background is not None else existing.get("background")
                ),
            )

            match state:
                case NodeState.RUNNING:
                    existing_state = existing.get("state")
                    is_existing_process_active = existing_state in {
                        NodeState.RUNNING.value,
                        NodeState.STOPPING.value,
                    }
                    entry.started_at = (
                        existing.get("started_at")
                        if is_existing_process_active and existing.get("started_at")
                        else now
                    )
                    entry.stopped_at = None
                    entry.exit_code = None
                case NodeState.STOPPED:
                    entry.stopped_at = now

            self._get_state_table().update_or_insert_by_id(
                self._node_id, entry.to_dict()
            )
            self._get_history_table().insert(entry.to_dict())
        except Exception as e:
            logger.warning(f"Could not persist node process state: {e}")

    def _get_pid(self) -> Optional[int]:
        """Get the PID of the currently running node process, if any."""
        state = self._get_state_table().get_by_id(self._node_id)
        if state and state.get("state") == NodeState.RUNNING.value:
            return state.get("pid")
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(
        self,
        node_args: dict,
        background: bool = False,
        actor: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = "start_requested",
    ) -> None:
        """Spawn a node subprocess.

        Args:
            node_args: Dict of arguments forwarded to the node subprocess.
            background: If True, the node will be started in the background.
            actor: Optional user/source metadata for process state attribution.
            reason: Optional reason for starting the process, default is "start_requested".
        """

        # Start should raise an error if the user is trying to start/restart an existing node process.
        status = self.get_status()
        if status == NodeState.RUNNING:
            logger.warning("Node process is already running. Ignoring start request.")
            return

        logger.info(f"Starting node subprocess with python={sys.executable}")
        logger.info(f"Node subprocess config root={self._config.root}")
        logger.info(
            "Application logs will be logged to file: "
            f"{os.path.join(self._config.root, 'log', DEFAULT_APPLICATION_LOG_FILE)}"
        )
        # We are starting a new node process. We generate a new pid after starting the process.
        _process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "fedbiomed.node.node_pm",
                "--config",
                self._config.root,
                "--node-args",
                json.dumps(
                    node_args
                ),  # Convert to string to pass using subprocess (will be parsed back to dict in the subprocess)
            ],
        )

        self._set_process_state(
            pid=_process.pid,
            state=NodeState.RUNNING,
            action="start",
            actor=actor,
            reason=reason,
            node_args=node_args,
            background=background,
        )
        logger.info(f"Node '{self._node_id}' started (pid={_process.pid}).")

        if not background:
            self._wait(_process, actor=actor)

        return

    def _wait(
        self,
        process: subprocess.Popen,
        actor: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Wait until the managed node process exits."""

        exit_code = process.wait()

        # In case the process exits abruptly, meaning:
        # 1- It was not killed from the GUI
        # 2- It was not killed from the CLI (using Ctrl+C)
        if self.get_status() != NodeState.STOPPED:
            self._set_process_state(
                pid=process.pid,
                state=NodeState.STOPPED,
                action="wait",
                actor=actor,
                reason="process_exited_abruptly",
                exit_code=exit_code,
            )

        logger.info(f"Node process pid={process.pid} exited with code {exit_code}.")
        return exit_code

    def stop(
        self,
        actor: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = "stop_requested",
    ) -> None:
        """Terminate the running node subprocess.

        Args:
            actor: Optional user/source metadata for process state attribution.
            reason: Optional reason for stopping the process, default is "stop_requested".
        """
        # In case we try to start/restart an existing node process.
        status = self.get_status()
        if status == NodeState.STOPPING or status == NodeState.STOPPED:
            logger.warning("Node process is already stopped. Ignoring stop request.")
            return

        pid = self._get_pid()
        if not pid:
            logger.warning("No running node process to stop.")
            return

        self._set_process_state(
            pid=pid,
            state=NodeState.STOPPING,
            action="stop",
            actor=actor,
            reason=reason,
        )

        try:
            _process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            logger.warning(
                f"Node process pid={pid} does not exist. Updating the database accurately."
            )
            self._set_process_state(
                pid=pid,
                state=NodeState.STOPPED,
                action="stop",
                actor=actor,
                reason="process_not_found",
                exit_code=None,
            )
            return

        _process.terminate()
        logger.info(f"Sent termination signal to node process (pid={_process.pid}).")

        exit_code = None
        try:
            exit_code = _process.wait(timeout=5)
        except psutil.TimeoutExpired:
            logger.warning(
                f"Federated Node Process did not terminate; sending SIGKILL to (pid={_process.pid})."
            )
            _process.kill()
            try:
                exit_code = _process.wait(timeout=5)
            except psutil.TimeoutExpired:
                logger.error(
                    f"Node process (pid={_process.pid}) did not terminate after SIGKILL. "
                    "Potential process leak - manual intervention required."
                )

        self._set_process_state(
            pid=pid,
            state=NodeState.STOPPED,
            action="stop",
            actor=actor,
            reason=reason,
            exit_code=exit_code,
        )
        logger.info("Node process stopped.")

    def restart(
        self,
        node_args: Optional[Dict[str, Any]] = None,
        background: Optional[bool] = None,
        actor: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = "restart_requested",
    ) -> None:
        """Stop then start the node subprocess.

        Args:
            node_args: Partial arguments to override the last saved execution
                settings. Omitted values inherit saved settings.
            background: Whether to start the node in the background. If omitted,
                the last saved execution mode is used.
            actor: Optional user/source metadata for process state attribution.
            reason: Optional reason for restarting the process, default is "restart_requested".
        """
        effective_node_args = dict(DEFAULT_NODE_ARGS)
        effective_background = False

        if self._get_state_table() and self._node_id:
            stored_state = self._get_state_table().get_by_id(self._node_id) or {}
            stored_node_args = stored_state.get("node_args")
            if isinstance(stored_node_args, dict):
                effective_node_args.update(stored_node_args)
            if isinstance(stored_state.get("background"), bool):
                effective_background = stored_state["background"]

        if node_args:
            effective_node_args.update(
                {key: value for key, value in node_args.items() if value is not None}
            )
        if effective_node_args["gpu_only"]:
            effective_node_args["gpu"] = True
        if background is not None:
            effective_background = background

        self.stop(actor=actor, reason=reason)
        self.start(
            node_args=effective_node_args,
            background=effective_background,
            actor=actor,
            reason=reason,
        )

    def _is_process_active(self, pid: Optional[int]) -> bool:
        """Check if a process with the given PID is active and running."""
        if not pid:
            return False
        try:
            process = psutil.Process(pid)
            if process.status() == psutil.STATUS_ZOMBIE:
                return False
            return process.is_running()
        except psutil.NoSuchProcess:
            return False

    def get_status(self) -> NodeState:
        """Get the current status of the node subprocess."""
        state = self._get_state_table().get_by_id(self._node_id)
        if state is None:
            return NodeState.UNKNOWN

        pid = state.get("pid")
        db_status = NodeState(state.get("state", NodeState.UNKNOWN))

        if self._is_process_active(pid) and db_status != NodeState.RUNNING:
            logger.warning(
                f"Node process status mismatch for pid={pid}: database status is '{db_status}', but process is actually RUNNING. Updating database."
            )
            self._set_process_state(
                pid=pid,
                state=NodeState.RUNNING,
                action="status_check",
                actor=None,
                reason="status_mismatch_detected",
            )
        if not self._is_process_active(pid) and db_status == NodeState.RUNNING:
            logger.warning(
                f"Node process status mismatch for pid={pid}: database status is '{db_status}', but process does not exist. Updating database to STOPPED."
            )
            self._set_process_state(
                pid=pid,
                state=NodeState.STOPPED,
                action="status_check",
                actor=None,
                reason="status_mismatch_detected",
                exit_code=None,
            )

        final_state = self._get_state_table().get_by_id(self._node_id)
        final_db_status = NodeState(final_state.get("state", NodeState.UNKNOWN))
        return final_db_status

    def get_process_state(self) -> NodeProcessStateEntry:
        """Get the current persisted node process state.

        Returns:
            Dictionary containing the latest persisted process-state metadata,
            with `state` resolved through `get_status`.
        """
        state_entry = NodeProcessStateEntry(
            pid=self._get_pid(),
            state=self.get_status().value,
            node_id=self._node_id,
            node_name=self._node_name,
            action=None,
            reason=None,
            actor=None,
            updated_at=None,
            started_at=None,
            stopped_at=None,
            exit_code=None,
            node_args=None,
            background=None,
        )

        if not self._get_state_table() or not self._node_id:
            raise FedbiomedError(
                f"{ErrorNumbers.FB327.value}: Node process state table is not initialized or node_id is missing."
            )

        stored_state = self._get_state_table().get_by_id(self._node_id)
        if not stored_state:
            return state_entry

        state_entry = state_entry.from_dict(dict(stored_state))
        return state_entry


if __name__ == "__main__":
    """Entry point for the node subprocess when started by the NodeProcessManager.

    This function is called with the node configuration and arguments, and it starts the node runtime.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Fed-BioMed Node Process Manager")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to node configuration directory"
    )
    parser.add_argument(
        "--node-args", type=str, required=False, help="JSON string of node arguments"
    )

    args = parser.parse_args()
    _start_node_process(args.config, args.node_args)

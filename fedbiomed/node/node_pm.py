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
from datetime import datetime, timezone
from types import FrameType
from typing import Any, Dict, Optional, Union

import psutil

from fedbiomed.common.constants import CONFIG_FOLDER_NAME, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.node.config import NodeConfig
from fedbiomed.node.dataset_manager._db_tables import (
    NodeProcessStateHistoryTable,
    NodeProcessStateTable,
)
from fedbiomed.node.node import Node


class NodeState(enum.Enum):
    STARTING = "starting"
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

    print(_node)
    print("\t- Node name: ", _node.config.get("default", "name"), "\n")

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
    """Manages a single node subprocess. This node subprocess is the one where it's config is passed during initialization."""

    def __init__(self, config) -> None:
        """Initialize the NodeProcessManager with the given configuration.

        Args:
            config: Node configuration object that the manager will use to spawn the node subprocess and manage its state.
        """
        self._config = config
        self._node_id: str | None = config.get("default", "id")
        self._node_name: str | None = config.get("default", "name")
        self._state_table: NodeProcessStateTable | None = None
        self._history_table: NodeProcessStateHistoryTable | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _init_state_tables(self) -> None:
        """Initialize the node process state tables from config.

        This method should be called before any state persistence operations to ensure the tables are ready.
        """
        db_path = os.path.abspath(
            os.path.join(
                self._config.root,
                CONFIG_FOLDER_NAME,
                self._config.get("default", "db"),
            )
        )
        self._state_table = NodeProcessStateTable(db_path)
        self._history_table = NodeProcessStateHistoryTable(db_path)

    def _set_process_state(
        self,
        *,
        pid: int,
        state: NodeState,
        action: str,
        actor: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> None:
        """Persist the current process state if the node DB is available.

        Args:
            state: The new state of the node.
            action: The action that triggered the state change.
            actor: Optional user/source metadata for process state attribution.
            reason: Optional reason for the state change.
            exit_code: Optional exit code for the process.
        """
        if (
            not self._state_table
            or not self._history_table
            or not self._node_id
            or not self._node_name
        ):
            return

        try:
            now = _utc_now()
            existing = self._state_table.get_by_id(pid) or {}
            entry = {
                "pid": pid,
                "state": state.value,
                "node_id": self._node_id,
                "node_name": self._node_name,
                "action": action,
                "reason": reason,
                "actor": self._build_actor(actor),
                "updated_at": now,
                "started_at": existing.get("started_at"),
                "stopped_at": existing.get("stopped_at"),
                "exit_code": exit_code,
            }

            if state == NodeState.RUNNING:
                entry["started_at"] = existing.get("started_at") or now
                entry["stopped_at"] = None
                entry["exit_code"] = None
            elif state == NodeState.STOPPED:
                entry["stopped_at"] = now
            elif state == NodeState.STARTING:
                entry["started_at"] = None
                entry["stopped_at"] = None
                entry["exit_code"] = None

            self._state_table.update_or_insert_by_id(pid, entry)
            self._history_table.insert(entry.copy())
        except Exception as e:
            logger.warning(f"Could not persist node process state: {e}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(
        self,
        node_args: dict,
        pid: Optional[int] = None,
        actor: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Spawn a node subprocess.

        Args:
            node_args: Dict of arguments forwarded to the node subprocess.
            actor: Optional user/source metadata for process state attribution.
        """
        # In case we try to start/restart an existing node process.
        if pid and (
            self.get_status(pid) == NodeState.RUNNING.value
            or self.get_status(pid) == NodeState.STARTING.value
        ):
            logger.warning(
                f"Node process 'pid={pid}' is already running. Ignoring start request."
            )
            return pid

        # We are starting a new node process. We generate a new pid after starting the process.
        self._init_state_tables()
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
            ]
        )

        new_pid = _process.pid

        if not new_pid:
            raise FedbiomedError(
                f"{ErrorNumbers.FB327.value}: Error in starting node process, could not get the PID."
            )

        self._set_process_state(
            pid=new_pid,
            state=NodeState.RUNNING,
            action="start",
            actor=actor,
            reason="process_started",
        )

        logger.info(f"Node '{self._node_id}' started (pid={new_pid}).")
        return new_pid

    def wait(
        self,
        pid: int,
        actor: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Wait until the managed node process exits."""

        try:
            process = psutil.Process(pid)
        except psutil.NoSuchProcess:
            logger.warning(f"Node process pid={pid} does not exist.")
            return None

        try:
            exit_code = process.wait()
        except psutil.NoSuchProcess:
            exit_code = None

        # In case process exits abruptly, meaning:
        # 1- It was not killed from the GUI
        # 2- It was not killed from the CLI (using Ctrl+C)
        if self.get_status(pid) != NodeState.STOPPED.value:
            self._set_process_state(
                pid=pid,
                state=NodeState.STOPPED,
                action="wait",
                actor=actor,
                reason="process_exited_abruptly",
                exit_code=exit_code,
            )

        logger.info(f"Node process pid={pid} exited with code {exit_code}.")

        return exit_code

    def stop(
        self,
        pid: int,
        actor: Optional[Dict[str, Any]] = None,
        reason: str = "stop_requested",
    ) -> None:
        """Terminate the running node subprocess.

        Args:
            actor: Optional user/source metadata for process state attribution.
            reason: Optional reason for stopping the process, default is "stop_requested".
        """
        # In case we try to start/restart an existing node process.
        if (
            self.get_status(pid) == NodeState.STOPPING.value
            or self.get_status(pid) == NodeState.STOPPED.value
        ):
            logger.warning(
                f"Node process 'pid={pid}' is already stopped. Ignoring stop request."
            )
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

        try:
            exit_code = _process.wait(timeout=5)
        except psutil.TimeoutExpired:
            logger.warning(
                f"Federated Node Process did not terminate; sending SIGKILL to (pid={_process.pid})."
            )
            _process.kill()
            exit_code = _process.wait(timeout=5)

        if _process.is_running():
            logger.error(
                f"Node process (pid={_process.pid}) seems to be alive after the SIGKILL signal. "
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
        pid: int,
        node_args: dict,
        actor: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Stop then start the node subprocess.

        Args:
            node_args: Dict of arguments forwarded to the node subprocess on start.
            actor: Optional user/source metadata for process state attribution.
        """
        self.stop(pid=pid, actor=actor, reason="restart_requested")
        return self.start(pid=pid, node_args=node_args, actor=actor)

    def get_status(self, pid: int) -> str:
        """Get the current status of the node subprocess."""
        self._init_state_tables()
        state = self._state_table.get_by_id(pid)
        if state is not None:
            return state.get("state", NodeState.UNKNOWN.value)
        return NodeState.UNKNOWN.value

    def get_process_state(self, pid: int) -> Dict[str, Any]:
        """Get the current node process state.

        Returns:
            Dictionary containing the latest persisted process-state metadata,
            with ``state`` resolved through :meth:`get_status`.
        """
        self._init_state_tables()

        state_entry = {
            "pid": pid,
            "state": self.get_status(pid),
            "node_id": self._node_id,
            "node_name": self._node_name,
            "action": None,
            "reason": None,
            "actor": None,
            "updated_at": None,
            "started_at": None,
            "stopped_at": None,
            "exit_code": None,
            "managed_by_current_process": False,
        }

        if not self._state_table or not self._node_id:
            raise FedbiomedError(
                f"{ErrorNumbers.FB327.value}: Node process state table is not initialized or node_id is missing."
            )

        stored_state = self._state_table.get_by_id(pid)
        if not stored_state:
            raise FedbiomedError(
                f"{ErrorNumbers.FB327.value}: No process state found for pid {pid}."
            )

        state_entry.update(dict(stored_state))
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

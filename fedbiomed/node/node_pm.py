# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Node process lifecycle manager."""

import enum
import getpass
import multiprocessing
import os
import signal
import sys
import time
from datetime import datetime, timezone
from types import FrameType
from typing import Any, Dict, Optional, Union

from fedbiomed.common.constants import CONFIG_FOLDER_NAME, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger
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


def _start_node_process(config, node_args):
    """Start the node runtime inside the managed subprocess.

    Args:
        config: Node configuration.
        node_args: Arguments for the node.
    """
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
        self._process: multiprocessing.Process | None = None
        self._node_id: str | None = config.get("default", "id")
        self._node_name: str | None = config.get("default", "name")
        self._state_table: NodeProcessStateTable | None = None
        self._history_table: NodeProcessStateHistoryTable | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_process(self) -> None:
        """Clear process references."""
        self._process = None

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
        if self._state_table is not None and self._history_table is not None:
            return

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
            existing = self._state_table.get_by_id(self._node_id) or {}
            entry = {
                "node_id": self._node_id,
                "node_name": self._node_name,
                "state": state.value,
                "pid": self._process.pid if self._process else None,
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

            self._state_table.update_or_insert_by_id(self._node_id, entry)
            self._history_table.insert(entry.copy())
        except Exception as e:
            logger.warning(f"Could not persist node process state: {e}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(
        self,
        node_args: dict,
        actor: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Spawn a node subprocess.

        Args:
            node_args: Dict of arguments forwarded to the node subprocess.
            actor: Optional user/source metadata for process state attribution.
        """
        if self._process is not None:
            if self._process.is_alive():
                self._init_state_tables()
                self._set_process_state(
                    state=NodeState.RUNNING,
                    action="start",
                    actor=actor,
                    reason="already_running",
                )
                logger.warning(
                    f"Node '{self._node_id}' is already running (pid={self._process.pid}). "
                    "Ignoring start request."
                )
                return
            self._cleanup_process()

        self._init_state_tables()
        self._set_process_state(
            state=NodeState.STARTING,
            action="start",
            actor=actor,
            reason="start_requested",
        )

        self._process = multiprocessing.Process(
            target=_start_node_process,
            name=f"node-{self._node_id}",
            args=(self._config, node_args),
        )
        self._process.daemon = True
        self._process.start()
        self._set_process_state(
            state=NodeState.RUNNING,
            action="start",
            actor=actor,
            reason="process_started",
        )
        logger.info(f"Node '{self._node_id}' started (pid={self._process.pid}).")

    def stop(
        self,
        actor: Optional[Dict[str, Any]] = None,
        reason: str = "stop_requested",
    ) -> None:
        """Terminate the running node subprocess.

        Args:
            actor: Optional user/source metadata for process state attribution.
            reason: Optional reason for stopping the process, default is "stop_requested".
        """
        if self._process is None:
            self._init_state_tables()
            self._set_process_state(
                state=NodeState.STOPPED,
                action="stop",
                actor=actor,
                reason="no_process_running",
            )
            logger.warning("No node process is running.")
            return

        if not self._process.is_alive():
            self._init_state_tables()
            self._set_process_state(
                state=NodeState.STOPPED,
                action="stop",
                actor=actor,
                reason="process_exited",
                exit_code=self._process.exitcode,
            )
            self._cleanup_process()
            logger.warning("No node process is running.")
            return

        self._init_state_tables()
        self._set_process_state(
            state=NodeState.STOPPING,
            action="stop",
            actor=actor,
            reason=reason,
        )
        self._process.terminate()
        logger.info(
            f"Sent termination signal to node process (pid={self._process.pid})."
        )
        time.sleep(0.5)
        if self._process.is_alive():
            logger.warning(
                f"Federated Node Process did not terminate; sending SIGKILL to (pid={self._process.pid})."
            )
            self._process.kill()
            time.sleep(0.5)

        exit_code = self._process.exitcode
        if self._process.is_alive():
            logger.error(
                f"Failed to kill node process (pid={self._process.pid}). "
                "Process leak - manual intervention required."
            )
            return

        self._set_process_state(
            state=NodeState.STOPPED,
            action="stop",
            actor=actor,
            reason=reason,
            exit_code=exit_code,
        )
        self._cleanup_process()
        logger.info("Node process stopped.")

    def restart(
        self,
        node_args: dict,
        actor: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Stop then start the node subprocess.

        Args:
            node_args: Dict of arguments forwarded to the node subprocess on start.
            actor: Optional user/source metadata for process state attribution.
        """
        self.stop(actor=actor, reason="restart_requested")
        self.start(node_args, actor=actor)

    def get_status(self) -> NodeState:
        """Get the current status of the node subprocess.

        Returns:
            Current live process state if this manager owns a running process,
            otherwise the latest state persisted in the node process state table.
            Defaults to ``STOPPED`` if no persisted state exists or it cannot be
            read.
        """
        if self._process is not None and self._process.is_alive():
            return NodeState.RUNNING

        try:
            self._init_state_tables()
            if self._state_table and self._node_id:
                state = self._state_table.get_by_id(self._node_id)
                if state and state.get("state"):
                    return NodeState(state["state"])
        except Exception as e:
            logger.warning(f"Could not read node process status: {e}")

        return NodeState.STOPPED

    def get_process_state(self) -> Dict[str, Any]:
        """Get the current node process state.

        Returns:
            Dictionary containing the latest persisted process-state metadata,
            with ``state`` resolved through :meth:`get_status`.
        """
        self._init_state_tables()

        state_entry = {
            "node_id": self._node_id,
            "node_name": self._node_name,
            "state": self.get_status().value,
            "pid": None,
            "action": None,
            "reason": None,
            "actor": None,
            "updated_at": None,
            "started_at": None,
            "stopped_at": None,
            "exit_code": None,
            "managed_by_current_process": False,
        }

        if self._state_table and self._node_id:
            stored_state = dict(self._state_table.get_by_id(self._node_id) or {})
            state_entry.update(stored_state)
            state_entry["state"] = self.get_status().value

        if self._process is not None and self._process.is_alive():
            state_entry["managed_by_current_process"] = (
                state_entry.get("pid") == self._process.pid
            )

        return state_entry

    @property
    def process(self) -> multiprocessing.Process | None:
        """The underlying process object, or None if not running."""
        return self._process

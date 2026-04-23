# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Node process lifecycle manager."""

import enum
import getpass
import multiprocessing
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fedbiomed.common.constants import CONFIG_FOLDER_NAME
from fedbiomed.common.logger import logger
from fedbiomed.node.dataset_manager._db_tables import (
    NodeProcessStateHistoryTable,
    NodeProcessStateTable,
)


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


class NodeProcessManager:
    """Manages a single node subprocess.

    One process at a time. Intended to be used as a module-level singleton
    so that both the CLI and the API server share the same instance.
    """

    def __init__(self) -> None:
        self._process: multiprocessing.Process | None = None
        self._node_id: str | None = None
        self._node_name: str | None = None
        self._state_table: NodeProcessStateTable | None = None
        self._history_table: NodeProcessStateHistoryTable | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_process(self) -> None:
        """Clear process and metadata references."""
        self._process = None
        self._node_id = None
        self._node_name = None
        self._state_table = None
        self._history_table = None

    @staticmethod
    def _build_actor(actor: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a sanitized actor dictionary for process state attribution."""
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

    def _init_state_tables(self, config) -> None:
        """Initialize the node process state tables from config."""
        db_path = os.path.abspath(
            os.path.join(
                config.root,
                CONFIG_FOLDER_NAME,
                config.get("default", "db"),
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
        """Persist the current process state if the node DB is available."""
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
        config,
        node_args: dict,
        actor: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Spawn a node subprocess.

        Args:
            config: NodeConfig object (must already be initialised on disk).
            node_args: Dict of arguments forwarded to ``start_node``.
            actor: Optional user/source metadata for process state attribution.
        """
        # Lazy import to avoid a circular dependency at module load time:
        # cli imports this module to access the singleton.
        from fedbiomed.node.cli import start_node  # noqa: PLC0415

        node_id = config.get("default", "id")
        node_name = config.get("default", "name")

        if self._process is not None:
            if self._process.is_alive():
                if node_id == self._node_id:
                    self._set_process_state(
                        state=NodeState.RUNNING,
                        action="start",
                        actor=actor,
                        reason="already_running",
                    )
                    logger.warning(
                        f"Node '{node_id}' is already running (pid={self._process.pid}). "
                        "Ignoring start request."
                    )
                return
            self._cleanup_process()

        self._node_id = node_id
        self._node_name = node_name
        self._init_state_tables(config)
        self._set_process_state(
            state=NodeState.STARTING,
            action="start",
            actor=actor,
            reason="start_requested",
        )

        self._process = multiprocessing.Process(
            target=start_node,
            name=f"node-{node_id}",
            args=(config, node_args),
        )
        self._process.daemon = True
        self._process.start()
        self._set_process_state(
            state=NodeState.RUNNING,
            action="start",
            actor=actor,
            reason="process_started",
        )
        logger.info(f"Node '{node_id}' started (pid={self._process.pid}).")

    def stop(
        self,
        actor: Optional[Dict[str, Any]] = None,
        reason: str = "stop_requested",
    ) -> None:
        """Terminate the running node subprocess."""
        if self._process is None or not self._process.is_alive():
            logger.warning("No node process is running.")
            return

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
        config,
        node_args: dict,
        actor: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Stop then start the node subprocess."""
        self.stop(actor=actor, reason="restart_requested")
        self.start(config, node_args, actor=actor)

    def get_status(self) -> NodeState:
        if self._process is not None and self._process.is_alive():
            return NodeState.RUNNING
        return NodeState.STOPPED

    @property
    def process(self) -> multiprocessing.Process | None:
        """The underlying process object, or None if not running."""
        return self._process


# Module-level singleton — safe with a single Gunicorn worker.
node_process_manager = NodeProcessManager()

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Node process lifecycle manager."""

import enum
import multiprocessing
import time
from typing import Any, Dict, Optional

from fedbiomed.common.logger import logger
from fedbiomed.node.process_state_manager import NodeProcessStateManager


class NodeState(enum.Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    UNKNOWN = "unknown"  # used only in route layer as a fallback


class NodeProcessManager:
    """Manages a single node subprocess.

    One process at a time. Intended to be used as a module-level singleton
    so that both the CLI and the API server share the same instance.
    """

    def __init__(self) -> None:
        self._process: multiprocessing.Process | None = None
        self._node_id: str | None = None
        self._node_name: str | None = None
        self._state_manager: "NodeProcessStateManager | None" = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_process(self, clear_metadata: bool = True) -> None:
        """Clear refs if the process has already exited."""
        if self._process is None:
            if clear_metadata:
                self._node_id = None
                self._node_name = None
                self._state_manager = None
            return
        if self._process.is_alive():
            return
        self._process.join(timeout=0.1)
        self._process = None
        if clear_metadata:
            self._node_id = None
            self._node_name = None
            self._state_manager = None

    def _is_process_running(self) -> bool:
        if self._process is None:
            return False
        if self._process.is_alive():
            return True
        self._cleanup_process()
        return False

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
        if not self._state_manager or not self._node_id or not self._node_name:
            return

        try:
            self._state_manager.set_state(
                node_id=self._node_id,
                node_name=self._node_name,
                pid=self._process.pid if self._process else None,
                state=state.value,
                action=action,
                actor=actor,
                reason=reason,
                exit_code=exit_code,
            )
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

        if self._is_process_running() and node_id == self._node_id:
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
        self._state_manager = NodeProcessStateManager.from_config(config)
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
        if not self._is_process_running():
            logger.warning("No node process is running.")
            self._cleanup_process()
            return

        self._set_process_state(
            state=NodeState.STOPPING,
            action="stop",
            actor=actor,
            reason=reason,
        )
        self._cleanup_process(clear_metadata=False)
        if self._process is None:
            self._set_process_state(
                state=NodeState.STOPPED,
                action="stop",
                actor=actor,
                reason=reason,
            )
            logger.info("Node process stopped.")
            return

        self._process.terminate()
        logger.info(
            f"Sent termination signal to node process (pid={self._process.pid})."
        )
        time.sleep(0.5)
        self._cleanup_process(clear_metadata=False)
        if self._process is not None:
            logger.warning(
                f"Federated Node Process did not terminate; sending SIGKILL to (pid={self._process.pid})."
            )
            self._process.kill()
            time.sleep(0.5)

        exit_code = self._process.exitcode if self._process else None
        self._cleanup_process(clear_metadata=False)
        if self._process is not None:
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
        if self._is_process_running():
            return NodeState.RUNNING
        return NodeState.STOPPED

    @property
    def process(self) -> multiprocessing.Process | None:
        """The underlying process object, or None if not running."""
        return self._process


# Module-level singleton — safe with a single Gunicorn worker.
node_process_manager = NodeProcessManager()

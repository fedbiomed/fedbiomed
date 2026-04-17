# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Node process lifecycle manager."""

import enum
import multiprocessing
import time

from fedbiomed.common.logger import logger


class NodeState(enum.Enum):
    RUNNING = "running"
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_process(self) -> None:
        """Clear refs if the process has already exited."""
        if self._process is None:
            self._node_id = None
            return
        if self._process.is_alive():
            return
        self._process.join(timeout=0.1)
        self._process = None
        self._node_id = None

    def _is_process_running(self) -> bool:
        if self._process is None:
            return False
        if self._process.is_alive():
            return True
        self._cleanup_process()
        return False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self, config, node_args: dict) -> None:
        """Spawn a node subprocess.

        Args:
            config: NodeConfig object (must already be initialised on disk).
            node_args: Dict of arguments forwarded to ``start_node``.
        """
        # Lazy import to avoid circular dependency at module load time.
        from fedbiomed.node.cli import start_node  # noqa: PLC0415

        node_id = config.get("default", "id")

        if self._is_process_running() and node_id == self._node_id:
            logger.warning(
                f"Node '{node_id}' is already running (pid={self._process.pid}). "
                "Ignoring start request."
            )
            return

        self._cleanup_process()

        self._process = multiprocessing.Process(
            target=start_node,
            name=f"node-{node_id}",
            args=(config, node_args),
        )
        self._process.daemon = True
        self._process.start()
        self._node_id = node_id
        logger.info(f"Node '{node_id}' started (pid={self._process.pid}).")

    def stop(self) -> None:
        """Terminate the running node subprocess."""
        if not self._is_process_running():
            logger.warning("No node process is running.")
            self._cleanup_process()
            return

        self._process.terminate()
        logger.info(
            f"Sent termination signal to node process (pid={self._process.pid})."
        )
        time.sleep(0.5)
        if self._is_process_running():
            logger.warning(
                f"Federated Node Process did not terminate; sending SIGKILL to (pid={self._process.pid})."
            )
            self._process.kill()
            time.sleep(0.5)

        self._cleanup_process()
        if self._is_process_running():
            logger.error(
                f"Failed to kill node process (pid={self._process.pid}). "
                "Process leak — manual intervention required."
            )
            return

        logger.info("Node process stopped.")

    def restart(self, config, node_args: dict) -> None:
        """Stop then start the node subprocess."""
        self.stop()
        self.start(config, node_args)

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

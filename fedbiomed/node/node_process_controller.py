import enum
from multiprocessing import Process

from fedbiomed.common.logger import logger


class NodeStatus(enum.Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class NodeProcessController:
    """Draft controller for node process lifecycle management."""

    def __init__(self):
        self._process = None
        self._node_id = None

    def _cleanup_process(self) -> None:
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

    def start(self, config, node_args) -> None:
        from fedbiomed.node.cli import start_node

        node_id = config.get("default", "id")

        if self._is_process_running() and node_id == self._node_id:
            logger.warning(
                "Node process is already running with pid=%s for node_id=%s",
                self._process.pid,
                self._node_id,
            )
            return

        self._cleanup_process()

        logger.info(
            "Starting node process with config: %s and node_args: %s",
            config,
            node_args,
        )

        self._process = Process(
            target=start_node,
            name=f"node-{node_id}",
            args=(config, node_args),
        )
        self._process.daemon = True
        self._process.start()
        self._node_id = node_id

        logger.info(
            "Node process started with pid=%s for node_id=%s",
            self._process.pid,
            node_id,
        )

    def stop(self) -> None:
        if not self._is_process_running():
            logger.warning("No running node process found to stop.")
            self._cleanup_process()
            return

        logger.info(
            "Stopping node process with pid=%s for node_id=%s",
            self._process.pid,
            self._node_id,
        )

        self._process.terminate()
        self._process.join(timeout=5)

        if self._process.is_alive():
            logger.warning(
                "Node process did not stop after terminate(), killing pid=%s for node_id=%s",
                self._process.pid,
                self._node_id,
            )
            self._process.kill()
            self._process.join(timeout=2)

        if self._process.is_alive():
            logger.error(
                "Failed to stop node process with pid=%s for node_id=%s",
                self._process.pid,
                self._node_id,
            )
            return

        stopped_node_id = self._node_id
        self._cleanup_process()
        logger.info("Node process stopped successfully for node_id=%s", stopped_node_id)

    def restart(self, config, node_args) -> None:
        logger.info("Restarting node process...")
        self.stop()
        self.start(config, node_args)

    def get_status(self) -> NodeStatus:
        if self._is_process_running():
            return NodeStatus.RUNNING

        return NodeStatus.STOPPED


nodeProcessController = NodeProcessController()

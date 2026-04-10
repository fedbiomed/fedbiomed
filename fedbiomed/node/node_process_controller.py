import enum

from fedbiomed.common.logger import logger


class NodeStatus(enum.Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class NodeProcessController:
    """Draft controller for node process lifecycle management."""

    def __init__(self):
        # Check if a default database (or file) is already created for keeping the process id
        # If not, create one
        pass

    def start(self, config, node_args) -> None:
        """Draft entry point for starting the node."""

        # Get the latest entry from the database (or file) to check if a node process is already running
        # If a process is already running, log a warning and skip starting a new one
        # If no process is running (or there is no entry in the database yet), start the node backend
        logger.info(
            "Starting node process with config: %s and node_args: %s", config, node_args
        )
        pass

    def stop(self) -> None:
        """Draft entry point for stopping the node."""

        # Get the latest entry from the database (or file) to check if a node process is running
        # If a process is running, attempt to stop it gracefully (e.g., by sending a termination signal)
        # If the process stops successfully, update the database entry to reflect that it is no longer running
        # If there is no process running (or no entry in the database), log a warning and skip stopping
        logger.info("Stopping node process...")
        pass

    def restart(self, config, node_args) -> None:
        """Draft entry point for restarting the node."""

        # Get the latest entry from the database (or file) to check if a node process is running
        # If a process is running, call the stop method to stop it gracefully
        # Then, call the start method with the provided configuration and arguments to start a new process
        logger.info("Restarting node process...")
        pass

    def get_status(self) -> NodeStatus:
        """Draft status accessor for the node."""

        # Get the latest entry from the database (or file) to check if a node process is running
        # Use the process id to check if the process is still active (e.g., by checking the process table or using a system command)
        # Return the status of the node (e.g., running, stopped, or unknown)
        logger.info("Retrieving node process status...")
        pass


# To ensure that the node process controller is a singleton
nodeProcessController = NodeProcessController()

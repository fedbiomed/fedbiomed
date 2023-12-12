import asyncio
import threading
from typing import Callable, List, Dict, Optional

from fedbiomed.transport.client import GrpcClient, ResearcherCredentials

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedCommunicationError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import Message


class GrpcAsyncTaskController:
    """RPC asynchronous task controller

    Launches async tasks for listening the requests/tasks coming from researcher as well as
    listener to send the replies that are created by the node. All the methods of this class
    are awaitable, except the constructor.
    """
    def __init__(
            self,
            node_id: str,
            researchers: List[ResearcherCredentials],
            on_message: Callable,
            debug: bool = False
    ) -> None:
        """Constructs GrpcAsyncTaskController

        Args:
            node_id: The ID of the node component that runs RPC client
            researchers: List of researchers that the RPC client will connect to.
            on_message: Callback function to be executed once a task received from the researcher
            debug: Activates debug mode for `asyncio`

        Raises:
            FedbiomedCommunicationError: bad argument type
        """

        # inform all threads whether communication client is started
        self._is_started = threading.Event()

        self._node_id = node_id
        self._researchers = researchers

        self._loop = None

        # Maps researcher ip to corresponding ids
        self._ip_id_map_lock = None
        self._ip_id_map = {}

        # Clients lock not needed for now (client list not modified after initialization)
        # but guarantees to be future safe for dynamic researcher clients' list
        self._clients_lock = None
        self._clients: Dict[str, GrpcClient] = {}

        self._debug = debug
        self._on_message = on_message


    async def start(self) -> None:
        """"Starts the tasks for each GrpcClient"""

        tasks = []
        for researcher in self._researchers:
            client = GrpcClient(self._node_id, researcher, self._update_id_ip_map)
            tasks.append(client.start(on_task=self._on_message))
            self._clients[f"{researcher.host}:{researcher.port}"] = client

        self._loop = asyncio.get_running_loop()

        # Create asyncio locks
        self._ip_id_map_lock = asyncio.Lock()
        self._clients_lock = asyncio.Lock()

        self._is_started.set()

        logger.info("Starting task listeners")

        # Run GrpcClient asyncio tasks
        await asyncio.gather(*tasks)


    async def send(self, message: Message, broadcast: bool = False) -> None:
        """Sends message to researcher.

        Args:
            message: Message to send
            broadcast: Broadcast the message to all available researcher. This option should be used for general
                node state messages (e.g. general Error)
        """
        if broadcast:
            return await self._broadcast(message)

        async with self._clients_lock:
            async with self._ip_id_map_lock:
                researcher = message.researcher_id
                await self._clients[self._ip_id_map[researcher]].send(message)


    async def _broadcast(self, message: Message) -> None:
        """Broadcast given message

        Args:
            message: Message to broadcast
        """
        for _, client in self._clients.items():
            await client.send(message)


    async def _update_id_ip_map(self, ip, id_) -> None:
        """Updates researcher IP and researcher ID map

        Args:
            ip: IP of the researcher whose ID will be created or updated
            id_: ID of the researcher to be updated
        """
        async with self._ip_id_map_lock:
            self._ip_id_map.update({id_: ip})


    async def is_connected(self) -> bool:
        """Checks if there is running tasks"""

        async with self._clients_lock:
            tasks = [not task.done() for client in self._clients.values() for task in client.tasks]
            return all(tasks)


class GrpcController(GrpcAsyncTaskController):
    """"gRPC Controller class

    This class is responsible of managing GrpcConnections with researcher components.
    It is wrapper class of GrpcClients. It has been designed to be called main or
    different threads than the one grpc client runs.

    Attributes:
        _thread: background thread of gRPC controller
    """

    _thread: Optional[threading.Thread] = None

    def _run(self, on_finish: Callable) -> None:
        """Runs async task controller.

        Args:
            on_finish: Called when the tasks for handling all known researchers have finished.
                Callable has no argument.
        """
        try:
            asyncio.run(super().start(), debug=self._debug)
        except Exception as e:
            logger.critical(
                "An exception raised by running tasks within GrpcClients. This will stop "
                f"gRPC client. The exception: {type(e).__name__}. Error message: {e}")
            logger.info("Node is stopped!")

            if isinstance(on_finish, Callable):
                on_finish()

    def start(self, on_finish: Optional[Callable] = None) -> None:
        """Start GRPCClients in a thread.

        Args:
            on_finish: Called when the tasks for handling all known researchers have finished. 
                Callable has no argument. If None, then no action is taken.
        """
        # Adds grpc handler to send node logs to researchers
        logger.add_grpc_handler(on_log=self.send, node_id=self._node_id)

        self._thread = threading.Thread(target=self._run, args=(on_finish,), daemon=True)
        self._thread.start()

    def send(self, message: Message, broadcast: bool = False) -> None:
        """Sends given message to researcher

        Researcher id must exist in the message.

        Args:
            message: Message to send to researcher
            broadcast: If True, broadcasts the given message to all available.
                This does not prevent adding `researcher_id` to the message.
                The attribute `researcher_id` in the message should be `<unknown>`

        Raises:
            FedbiomedCommunicationError: bad argument type
            FedbiomedCommunicationError: node is not started
        """
        if not isinstance(message, Message):
            raise FedbiomedCommunicationError(
                f"{ErrorNumbers.FB628}: bad argument type for message, expected `Message`, got `{type(message)}`")

        if not self._is_started.is_set():
            raise FedbiomedCommunicationError(f"{ErrorNumbers.FB628}: Communication client is not initialized.")

        asyncio.run_coroutine_threadsafe(
            super().send(message, broadcast), self._loop
        )


    def is_connected(self) -> bool:
        """"Checks GrpcController is connected to any RPC client.

        This method should only be called from different thread than the one that asyncio loop running in.

        Returns:
            Connection status

        Raises:
            FedbiomedCommunicationError: node is not started
        """
        if self._thread is None or not self._is_started.is_set():
            raise FedbiomedCommunicationError(f"{ErrorNumbers.FB628}: Communication client is not initialized.")

        if not self._thread.is_alive():
            return False

        future = asyncio.run_coroutine_threadsafe(
            super().is_connected(), self._loop
        )
        return future.result()

import asyncio
import threading

from dataclasses import dataclass
from typing import Callable, Optional, List, Dict

from fedbiomed.common.logger import logger
from fedbiomed.common.message import Message

from fedbiomed.transport.client import GrpcClient


@dataclass
class ResearcherCredentials:

    port: str
    host: str
    certificate: Optional[str] = None


class RPCAsyncTaskController:
    """RPC asynchronous task controller 

    Launches async tasks for listening the requests/tasks coming form researcher as well as 
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
        """Constructs RPCAsyncTaskController

        Args: 
            node_id: The ID of the node component that runs RPC client
            researchers: List of researchers that the RPC client will connect to.
            on_message: Callback function to be executed once a task received from the researcher
            debug: Activates debug mode for `asyncio`

        """
        self._node_id = node_id
        self._researchers = researchers

        self._thread = None
        self._loop = None

        # Maps researcher ip to corresponding ids
        self._ip_id_map_lock = None
        self._ip_id_map = {}

        self._clients_lock = None
        self._clients: Dict[str, GrpcClient] = {}

        self._debug = debug
        self._on_message = on_message


    async def _start(self):
        """"Starts the tasks for each GrpcClient"""

        tasks = []
        for researcher in self._researchers:
            client = GrpcClient(self._node_id, researcher, self._update_id_ip_map)
            tasks.extend(client.start(on_task=self._on_message))
            self._clients[f"{researcher.host}:{researcher.port}"] = client

        self._loop = asyncio.get_running_loop()

        # Create asyncio locks
        self._ip_id_map_lock = asyncio.Lock()
        self._clients_lock = asyncio.Lock()


        logger.info("Starting task listeners")

        # Run GrpcClient asyncio tasks
        await asyncio.gather(*tasks)

    async def send(self, message, broadcast: bool = False) -> None:
        """Sends message to researcher.

        Args: 
            message: Message to send
            broadcast: Broadcast the message to all available researcher. This option should be used for general 
                node state messages (e.g. general Error)
        """
        if broadcast:
            return await self._broadcast(message)

        async with self._ip_id_map_lock:
            researcher = message.researcher_id
            return await self._clients[self._ip_id_map[researcher]].send(message)

    async def _broadcast(self, message: Message):
        """Broadcast given message

        Args: 
            message: Message to broadcast
        """

        async with self._clients_lock:
            for _, client in self._clients.items():
                await client.send(message)


    async def _update_id_ip_map(self, ip, id_):
        """Updates researcher IP and researcher ID map

        Args:
            ip: IP of the researcher whose ID will be created or updated
            id_: ID of the researcher to be updated
        """
        async with self._ip_id_map_lock:
            self._ip_id_map = {id_: ip}

    async def _is_connected(self):
        """Checks if there is running tasks"""

        async with self._clients_lock:
            tasks = [not task.done() for client in self._clients.values() for task in client.tasks]
            return all(tasks) and self._thread is not None and self._thread.is_alive()


class RPCController(RPCAsyncTaskController):
    """"RPC Controller class 

    This class is responsible of managing GrpcConnections with researcher components. 
    It is wrapper class of GrpcClients. It has been designed to be called main or 
    different threads than the one grpc client runs. 

    """
    def __init__(self, *args, **kwargs):
        """Constructs RPC controller"""

        super().__init__(*args, **kwargs)

        # Adds grpc handler to send node logs to researchers
        logger.add_grpc_handler(on_log=self.send, node_id=self._node_id)

    def _run(self):
        """Runs async task controller"""
        try: 
            asyncio.run(self._start(), debug=self._debug)
        except Exception as e:
            logger.critical(
                "An exception raised by running tasks within GrpcClients. This will close stopping " 
                f"gRPC client. The exception: {type(e).__name__}. Error message: {e}")
            logger.info("Node is stopped!")

    def start(self):
        """Start GRPCClients in a thread"""

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(self, message: Message, broadcast=False):
        """Sends given message to researcher

        Researcher id should be specified in the message

        Args: 
            message: Message to send to researcher
            broadcast: If True, broadcasts the given message to all available. 
                This does not prevent adding `researcher_id` to the message. 
                The attribute `researcher_id` in the message should be `<unknown>`
        """
        asyncio.run_coroutine_threadsafe(
            super().send(message, broadcast), self._loop
        )


    def is_connected(self) -> bool:
        """"Checks RPCController is connected to any RPC client.

        This method should only be called from different thread than the one that asyncio loop running in.

        Returns:
            Connection status
        """

        future = asyncio.run_coroutine_threadsafe(
            self._is_connected(), self._loop
        )

        return future.result()

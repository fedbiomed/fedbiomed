import asyncio
import concurrent.futures
import grpc
import time
import queue
import threading
import sys 
import signal 
import ctypes

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Union
from google.protobuf.message import Message as ProtobufMessage

from .client import GrpcClient

import fedbiomed.proto.researcher_pb2_grpc as researcher_pb2_grpc
from fedbiomed.proto.researcher_pb2 import TaskRequest, FeedbackMessage
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.constants import MAX_MESSAGE_BYTES_LENGTH
from fedbiomed.common.message import Message, TaskRequest, FeedbackMessage, TaskResult, ProtoSerializableMessage

class CancelTypes(Enum):
    RAISE = 0
    SILENT = 1


class _RPCStop(Exception):
    """RPC Stop action"""

def create_channel(
    address: str,
    certificate: str = None
) -> grpc.Channel :
    """ Create gRPC channel 
    
    Args: 
        address: Address to connect 
        certificate: TLS certificate
    
    Returns: 
        gRPC connection channel
    """
    channel_options = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ("grpc.keepalive_time_ms", 1000 * 2),
        ("grpc.initial_reconnect_backoff_ms", 1000),
        ("grpc.min_reconnect_backoff_ms", 500),
        ("grpc.max_reconnect_backoff_ms", 2000),
    ]

    if certificate is None: 
        channel = grpc.aio.insecure_channel(address, options=channel_options)
    else:
        # TODO: Create secure channel
        raise NotImplemented("Certificate option is not implemented")
    
    return channel


@dataclass
class ResearcherCredentials:

    port: str
    host: str
    certificate: Optional[str] = None


class RPCAsyncTaskController:

    def __init__(self, 
                node_id: str,
                researchers: List[ResearcherCredentials],
                on_message: Callable,
                debug: bool = False
    ) -> None:

        self._node_id = node_id
        self._researchers = researchers
        self.on_message = on_message

        self._thread = None

        # Maps researcher ip to corresponding ids
        self._ip_id_map_lock = asyncio.Lock()
        self._ip_id_map = {}

        self._clients_lock = asyncio.Lock()
        self._clients: Dict[str, GrpcClient] = {}


    async def start(self):
        """"Starts the tasks for each GrpcClient"""

        tasks = []
        for researcher in self._researchers:
            client = GrpcClient(self._node_id, researcher, self._update_id_ip_map)
            tasks.extend(client.start(on_task=self.on_message))
            self._clients[f"{researcher.host}:{researcher.port}"] = client
        
        self.loop = asyncio.get_running_loop()

        # Create asyncio locks
        self._ip_id_map_lock = asyncio.Lock()
        self._clients_lock = asyncio.Lock()


        logger.info("Starting task listeners")
        # Run GrpcClient asyncio tasks
        await asyncio.gather(*tasks)

    async def send(self, message):
        """Sends message to researcher.

        Args: 
            message: Message to send
        """
        async with self._ip_id_map_lock:
            researcher = message.researcher_id
            return await self._clients[self._ip_id_map[researcher]].send(message)

    async def log(self, message, broadcast, researcher_id):
        """Method to be called by logger
        Args: 
            message: Log message
            broadcast: If True, sends log message to all available researchers
            researcher_id: Sends the log only specified researcher. Ignored if broadcast is `True` 
        """

        with self._clients_lock:
            if broadcast:
                for client in self._client.values():
                    await client.send(message)

            elif researcher_id:
                await self.send(message)

    async def _update_id_ip_map(self, ip, id_):
        """Updates researcher IP and researcher ID map
        
        Args:
            ip: IP of the researcher whose ID will be created or updated
            id_: ID of the researcher to be updated
        """
        async with self._ip_id_map_lock:
            self._ip_id_map = {id_: ip}

    async def is_connected(self):
        """Check if """
        async with self._clients_lock:
            tasks = [not task.done() for client in self._clients.values() for task in client.tasks]
            return all(tasks) and self._thread.is_alive()
    

class RPCController:
    """"RPC Controller class 
    
    This class is responsible of managing GrpcConnections with researcher components. 
    It is wrapper class of GrpcClients 

    """
    def __init__(
            self,
            node_id: str,
            researchers: List[ResearcherCredentials],
            on_message: Callable = None,
            debug: bool = False
        ):
        
        self._rpc_async_task_controller = RPCAsyncTaskController(
            node_id=node_id, 
            researchers=researchers, 
            on_message=on_message, 
            debug=debug
        )
        self._node_id = node_id
        # Adds grpc handler to send node logs to researchers
        logger.add_grpc_handler(on_log=self.log,
                                node_id=self._node_id)



    def start(self):
        """Start GRPCClients in a thread"""

        def _run():
            # try: 
            asyncio.run(self._rpc_async_task_controller.start(), debug=False)
            # except Exception as e:
            #     logger.critical(
            #         "An exception raised by running tasks within GrpcClients. This will close stopping " 
            #         f"gRPC client. Please see error: {e}")
            #     logger.info("Node is stopped!")

        self._thread = threading.Thread(target=_run)
        self._thread.daemon = True
        self._thread.start()
        
    def send(self, message: Message):
        """Sends given message to researcher
        
        Researcher id should be specified in the message
        """
        # Non blocking call 
        #  - Create new thread different than main or spawn one
        #  - runs coroutine `_send` as threadsafe from that thread 
        # This guarantees that `_send` will be called from different thread
        # even `send` is called within async-thread  
        # self.loop.run_in_executor(None, 
        #                           asyncio.run_coroutine_threadsafe, 
        #                           self._send(message), 
        #                           self.loop)

        asyncio.run_coroutine_threadsafe(
            self._rpc_async_task_controller.send(message), 
            self._rpc_async_task_controller.loop
        ) 
                                  


    def log(self, message, broadcast, researcher_id):
        """Method to be called by logger
        Args: 
            message: Log message
            broadcast: If True, sends log message to all available researchers
            researcher_id: Sends the log only specified researcher. Ignored if broadcast is `True` 
        """
        asyncio.run_coroutine_threadsafe(
            self._rpc_async_task_controller.log(message, broadcast, researcher_id), 
            self._rpc_async_task_controller.loop
        ) 


    def is_connected(self , async_ = False):
        """"Checks RPCController is connected to any RPC client
        
        This method should only be called from different thread than the one that asyncio loop running in.
        """
        
        if async_ == False:
            future = asyncio.run_coroutine_threadsafe(
                self._rpc_async_task_controller.is_connected(),
                self._rpc_async_task_controller.loop
                )
            return future.result()

        return self._rpc_async_task_controller.is_connected()
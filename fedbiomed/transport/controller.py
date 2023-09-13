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
        self.on_message = on_message or self._default_callback

        self._node_id = node_id
        self._certificate: str = None
        self._client_registered = False
        self._researchers = researchers

        self._clients: Dict[str, GrpcClient] = {}
        

        self._thread = None
        self._task_channel = None
        self._feedback_channel = None
        # note: _node_configured is not enough, many race conditions remaining ...
        self._node_configured = False

        self._debug = debug

        # Maps researcher ip to corresponding ids
        self._ip_id_map = {}

        # Adds grpc handler to send node logs to researchers
        logger.add_grpc_handler(on_log=self.log,
                                node_id=self._node_id)


    async def _start(self):
        
        tasks = []
        for researcher in self._researchers:
            client = GrpcClient(self._node_id, researcher, self._update_id_ip_map)
            tasks.extend(client.start(on_task=self.on_message))
            self._clients[f"{researcher.host}:{researcher.port}"] = client
        
        self.loop = asyncio.get_running_loop()

        logger.info("Starting task listeners")
        # Run GrpcClient asyncio tasks
        try:
            await asyncio.gather(*tasks)
        except asyncio.exceptions.CancelledError as e:
            raise _RPCStop

    def start(self):
        """Start GRPCClients in a thread"""

        def _run():
            try: 
                asyncio.run(self._start(), debug=False)
            except _RPCStop: 
                logger.info("Node is stopped.")
            except Exception as e:
                logger.critical(
                    "An exception raised by running tasks within GrpcClients. This will close stopping " 
                    f"gRPC client. Please see error: {e}")
                logger.info("Node is stopped!")

        self._thread = threading.Thread(target=_run)
        self._thread.daemon = True
        self._thread.start()

        
    def send(self, message: Message):
        """Sends given message to researcher
        
        Researcher id should be specified in the message
        """
        
        researcher = message.researcher_id
        self._clients[self._ip_id_map[researcher]].send(message)


    def log(self, message, broadcast, researcher_id):
        """Method to be called by logger
        Args: 
            message: Log message
            broadcast: If True, sends log message to all available researchers
            researcher_id: Sends the log only specified researcher. Ignored if broadcast is `True` 
        """
        print("It is here")
        if broadcast:
            for client in self._client.values():
                client.send(message)

        elif researcher_id:
            self.send(message)

    async def _update_id_ip_map(self, ip, id_):
        """Updates researcher IP and researcher ID map
        
        Args:
            ip: IP of the researcher whose ID will be created or updated
            id_: ID of the researcher to be updated
        """
        async with asyncio.Lock():
            self._ip_id_map = {id_: ip}



    def stop(self):
        """Stops running asyncio loops"""
        # Silently cancel tasks
        logger.info("Gracefully stopping the node!")
        if hasattr(self, "loop"):
            tasks = asyncio.all_tasks(self.loop)

            for client in self._clients.values():
                client.cancel_tasks()

            # Remaining tasks
            tasks = asyncio.all_tasks(self.loop)
            for task in tasks:
                self.loop.call_soon_threadsafe(task.cancel, CancelTypes.SILENT)


        # Stop the thread
        self._thread.join()


    def is_connected(self):
        """"Checks RPCController is connected to any RPC client"""
        tasks = [not task.done() for client in self._clients.values() for task in client.tasks]

        return any(tasks) and self._thread.is_alive()
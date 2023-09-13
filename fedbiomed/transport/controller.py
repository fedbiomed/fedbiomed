import asyncio
import concurrent.futures
import grpc
import queue
import threading
import sys 
import signal 
import ctypes

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


def create_channel(
    address: str,
    certificate: str = None
) -> grpc.Channel :
    """ Create gRPC channel 
    
    Args: 
        ip: 
        port:
        certificate:
    
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
        # ("grpc.enable_retries", 1), # Not working
        # ("grpc.service_config", service_config) # Not working
    ]

    if certificate is None: 
        channel = grpc.aio.insecure_channel(address, options=channel_options)
    else:
        # TODO: Create secure channel
        pass
    
    # TODO: add callback fro connection state

    return channel


@dataclass
class ResearcherCredentials:

    ip: str
    host: str
    certificate: Optional[str]


class RPCController:

    def __init__(
            self,
            node_id: str,
            researchers: List[ResearcherCredentials],
            on_message: Callable = None,
            debug: bool = False
        ):

        self._node_id = node_id
        self._certificate: str = None
        self._client_registered = False
        self._researchers = researchers

        self._clients: Dict[str, GrpcClient] = {}
        
        self.on_message = on_message or self._default_callback

        self._thread = None
        self._task_channel = None
        self._feedback_channel = None
        # note: _node_configured is not enough, many race conditions remaining ...
        self._node_configured = False

        self._debug = debug

        logger.add_grpc_handler(on_log=self.send,
                              node_id=self._node_id)


    async def _start(self):
        
        tasks = []
        for researcher in self.researchers:
            client = GrpcClient(ip=researcher.ip, host=researcher.id, node_id=self._node_id)
            tasks.extend(client.start())
            self._clients[researcher.id] = client
        
        self.loop = asyncio.get_running_loop()

        logger.info("Starting task listeners")
        # Run GrpcClient asyncio tasks
        await asyncio.gather(*tasks)


    def start(self):
        """Start GRPCClients"""

        def _run():
            try: 
                asyncio.run(self._start(), debug=True)
            except Exception as e:
                logger.error(
                    "An exception raised by running tasks within GrpcClients. This will close stopping " 
                    f"gRPC client. Please see error: {e}")
            
        self._thread = threading.Thread(target=_run)
        self._thread.daemon = True
        self._thread.start()

        
    def send(self, message: Message):
        """Sends given message to researcher
        
        Researcher id should be specified in the message
        """
        researcher = message.researcher_id
        self._clients[researcher].send(message)


    def log(self, message, broadcast, researcher_id):
        """Method to be called by logger
        Args: 
            message: Log message
            broadcast: If True, sends log message to all available researchers
            researcher_id: Sends the log only specified researcher. Ignored if broadcast is `True` 
        """

        if broadcast:
            for client in self._client.values():
                client.send(message)

        elif researcher_id:
            self.send(message)



    def stop(self):
        """Stops running asyncio loops"""
        # Silently cancel tasks
        for _, client in self._clients.items():
            client.cancel_task()

        # Close asyncio running loop
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.close)

        # Stop the thread
        self._thread.join()
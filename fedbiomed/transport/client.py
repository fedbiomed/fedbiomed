from typing import List, Callable, Optional, Awaitable, Iterable
from enum import Enum
import asyncio
import abc
import ssl
import socket

from dataclasses import dataclass

import grpc

from fedbiomed.transport.protocols.researcher_pb2_grpc import ResearcherServiceStub

from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.message import Message, TaskRequest, TaskResult, FeedbackMessage
from fedbiomed.common.constants import MAX_MESSAGE_BYTES_LENGTH, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedCommunicationError


@dataclass
class ResearcherCredentials:

    port: str
    host: str
    certificate: Optional[str] = None


class ClientStatus(Enum):
    DISCONNECTED = 0
    CONNECTED = 1
    FAILED = 2


# timeout in seconds for retrying connection to the server when it does not reply or returns an error
GRPC_CLIENT_CONN_RETRY_TIMEOUT = 2

# timeout in seconds of a request to the server for a task (payload) to run on the node
GRPC_CLIENT_TASK_REQUEST_TIMEOUT = 60


def create_channel(
    port: str,
    host: str,
    certificate: Optional[str] = None
) -> grpc.Channel :
    """ Create gRPC channel

    Args:
        ip: IP address of the channel
        port: TCP port of the channel
        certificate: certificate for secure channel, or None for unsecure channel

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
        # ('grpc.ssl_target_name_override', 'localhost') # ...
    ]

    if certificate is None:
        channel = grpc.aio.insecure_channel(f"{host}:{port}", options=channel_options)
    else:
        channel = grpc.aio.secure_channel(f"{host}:{port}", certificate, options=channel_options)

    return channel

def is_server_alive(host:str, port: str):
    """Checks if the server is alive"""
    port = int(port)
    address_info = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    for family, socktype, protocol, _ , address in address_info:
        s = socket.socket(family, socktype, protocol)
        try:
            s.connect(address)
        except socket.error:
            return False
        else:
            s.close()
            return True


class ServerState:
    """Keeps server events and state.

    Holds server connection events to release gRPC request or connection
    refreshes.
    """

    def __init__(self):
        """Creates asyncio Events"""
        self._continue_requests = asyncio.Event()
        self._assure_connection = asyncio.Event()

        self._assure_connection.set()

    def connection_is_ready(self) -> Awaitable:
        """Waits for successful connection to the grpc server."""
        return self._continue_requests.wait()

    def continue_requests(self) -> None:
        """Sparks events that wait for succesful connection."""
        self._continue_requests.set()
        self._continue_requests.clear()
        self._assure_connection.clear()

    def reassure_connection(self) -> None:
        """Triggers event to reassure server connection."""
        self._assure_connection.set()

    async def do_assure_connection(self) -> bool:
        """Checks constanly if connection should be started or restarted.

        Returns:
            True if connection is dropped or required to be verified. False if
                connection is already stable
        """
        if self._assure_connection.is_set():
            return True

        self._continue_requests.set()
        self._continue_requests.clear()
        await asyncio.sleep(0.01)  # This call is required to release async tasks

        return False

class Channels:
    """Keeps gRPC server channels"""


    def __init__(self, researcher: ResearcherCredentials):
        """Create channels and stubs"""
        self.researcher = researcher

        self.task_channel: grpc.aio.Channel = None
        self.feedback_channel: grpc.aio.Channel = None

        self.task_stub: ResearhcerServiceStub = None
        self.feedback_stub: ResearcherServiceStub = None

    async def connect(self):

        # Closes fi channels are open
        if self.feedback_channel:
            await self.feedback_channel.close()

        if self.task_channel:
            await self.task_channel.close()

        self.feedback_channel = self.create()
        self.feedback_stub = ResearcherServiceStub(channel=self.feedback_channel)

        self.task_channel = self.create()
        self.task_stub = ResearcherServiceStub(channel=self.task_channel)

    def create(self):
        """Creates new channel"""
        return create_channel(
            port=self.researcher.port,
            host=self.researcher.host,
            certificate= grpc.ssl_channel_credentials(self.researcher.certificate) # grpc.ssl_channel_credentials()
        )



class GrpcClient:
    """An agent of remote researcher gRPC server."""

    def __init__(self, node_id: str, researcher: ResearcherCredentials, update_id_map: Callable) -> None:
        """Class constructor

        Args:
            node_id: unique ID of this node (connection client)
            researcher: the researcher to which the node connects (connection server)
            update_id_map: function to call when updating the researcher ID, needs proper prototype
        """
        self._id = None

        self._researcher = researcher
        self._server_state = ServerState()

        self._channels = Channels(researcher)

        self._task_stub = None
        self._feedback_stub = None

        self._feedback_channel = None
        self._task_channel = None

        self._task_listener = TaskListener(
            server_state=self._server_state,
            channels=self._channels,
            node_id=node_id,
            on_status_change = self._on_status_change,
            update_id=self._update_id)

        self._sender = Sender(
            channels=self._channels,
            server_state=self._server_state,
            on_status_change = self._on_status_change)

        # TODO: use `self._status` for finer gRPC agent handling.
        # Currently, the (tentative) status is maintained but not used
        self._status  = ClientStatus.DISCONNECTED

        self._update_id_map = update_id_map
        self._tasks = []

    async def connect(self):
        """Updates connection state and dispatch event to run listeners

        This method also implements auto-trust for server certificate
        """

        while True:
            if await self._server_state.do_assure_connection():
                if is_server_alive(self._researcher.host, self._researcher.port):
                    # Gets server certificate before creating the channel
                    # This implementation assumes that the provided IP and PORT trusted
                    self._researcher.certificate = \
                        bytes(ssl.get_server_certificate(
                            (self._researcher.host, self._researcher.port)),
                              'utf-8')
                    # Connect to channels and create stubs
                    await self._channels.connect()
                    # Dispatch event to continue requests
                    self._server_state.continue_requests()
                else:
                    logger.info(
                        "Researcher server is not available, will retry connect in "
                        f"{GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds")
                    await asyncio.sleep(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
            
    def start(self, on_task) -> List[Awaitable[Optional[Callable]]]:
        """Start researcher gRPC agent.

        Starts long-lived tasks, one waiting for server requests, one waiting on the async queue
        for the replies from the node that are going to be sent back to researcher.

        Args:
            on_task: Callback function to execute once a payload received from researcher.

        Returns:
            A list of task objects of the agent
        """

        # taks listener and sender will depend on connectivity task
        connect = asyncio.create_task(self.connect())

        self._tasks = [connect, self._task_listener.listen(on_task), self._sender.listen()]

        return self._tasks


    async def send(self, message: Message) -> None:
        """Sends messages from node to researcher server.

        Args:
            message: message to send from node to server
        """

        await self._sender.send(message)


    def _on_status_change(self, status: ClientStatus) -> None:
        """Callback function to change the researcher status

        Args:
            status: New status of the researcher client
        """
        self._status = status


    async def _update_id(self, id_: str) -> None:
        """Updates researcher ID

        Args:
            id_: Researcher Id

        Raises:
            FedbiomedCommunicationError: suspected malicious researcher
        """
        if self._id is not None and self._id != id_:
            msg = f"{ErrorNumbers.FB628}: Suspected malicious researcher activity ! " \
                f"Researcher ID changed for {self._researcher.host}:{self._researcher.port} from " \
                f"`{self._id}` to `{id_}`"
            logger.error(msg)
            raise FedbiomedCommunicationError(msg)

        self._id = id_
        await self._update_id_map(f"{self._researcher.host}:{self._researcher.port}", id_)


class Listener:
    """Abstract generic listener method for a node's communications."""

    def __init__(self, server_state, channels: Channels) -> None:
        """Constructs task listener channels
        """
        self._channels = channels
        self._server_state = server_state


    def listen(self, callback: Optional[Callable] = None) -> Awaitable[Optional[Callable]]:
        """Listens for tasks from given channels

        Args:
            callback: Callback function

        Returns:
            Asyncio task to run task listener
        """
        return asyncio.create_task(self._listen(callback))


    @abc.abstractmethod
    def _listen(self, callback: Optional[Callable] = None):
        pass


class TaskListener(Listener):
    """Listener for the task assigned by the researcher component """

    def __init__(
            self,
            server_state: ServerState,
            channels: Channels,
            node_id: str,
            on_status_change: Callable,
            update_id: Callable
    ) -> None:
        """Class constructor.

        Args:
            stub: RPC stub to be used for polling tasks from researcher
            node_id: unique ID for this node
            on_status_change: Callback function to run for changing node agent status
            update_id: Callback function to run updating peer researcher ID
        """
        super().__init__(server_state, channels)

        self._node_id = node_id
        self._on_status_change = on_status_change
        self._update_id = update_id

    async def _listen(self, callback: Optional[Callable] = None) -> None:
        """"Starts the loop for listening task

        Args:
            callback: Callback to execute once a task is received

        Raises:
            FedbiomedCommunicationError: communication error with researcher
        """

        while True:

            await self._server_state.connection_is_ready()
            try:
                await self._request(callback)
            except grpc.aio.AioRpcError as exp:
                match exp.code():
                    case grpc.StatusCode.DEADLINE_EXCEEDED:
                        logger.debug(
                            "Researcher did not request executing a task before timeout. Send new task request")
                    case grpc.StatusCode.UNAVAILABLE:
                        self._on_status_change(ClientStatus.DISCONNECTED)
                        self._server_state.reassure_connection()
                    case grpc.StatusCode.UNKNOWN:
                        self._on_status_change(ClientStatus.FAILED)
                        logger.error("Unexpected error raised by researcher gRPC server. This is probably due to "
                                     f"bug on the researcher side: {exp}. Will retry connect in "
                                     f"{GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds")
                        await asyncio.sleep(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
                    case _:
                        self._on_status_change(ClientStatus.FAILED)
                        logger.error("Unhandled gRPC call status {exp.code()}. Exception: {exp}. Will retry connect in "
                                     f"{GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds")
                        await asyncio.sleep(GRPC_CLIENT_CONN_RETRY_TIMEOUT)

            except Exception as exp:
                self._on_status_change(ClientStatus.FAILED)
                raise FedbiomedCommunicationError(
                    f"{ErrorNumbers.FB628}: Task listener has stopped due to unknown reason: {exp}") from exp


    async def _request(self, callback: Optional[Callable] = None) -> None:
        """Requests tasks from Researcher

        Args:
            callback: Callback to execute once a task is arrived
        """
        while True:
            logger.debug("Sending new task request to researcher")
            self._on_status_change(ClientStatus.CONNECTED)
            iterator = self._channels.task_stub.GetTaskUnary(
                TaskRequest(node=f"{self._node_id}").to_proto(), timeout=GRPC_CLIENT_TASK_REQUEST_TIMEOUT
            )
            # Prepare reply
            reply = bytes()
            async for answer in iterator:
                reply += answer.bytes_
                if answer.size != answer.iteration:
                    continue
                else:
                    # Execute callback
                    logger.debug("New task received from researcher")
                    task = Serializer.loads(reply)

                    # Guess ID of connected researcher, for un-authenticated connection
                    await self._update_id(task["researcher_id"])

                    if isinstance(callback, Callable):
                        # we could check the callback prototype
                        callback(task)

                    # Reset reply
                    reply = bytes()


class Sender(Listener):

    def __init__(
        self,
        server_state: ServerState,
        channels: Channels,
        on_status_change: Callable,
    ) -> None:
        """Class constructor.

        Args:
            feedback_stub: RPC stub to use for node feedback messages (logs, scalar update on training)
            task_stub: RPC stub to use for node replies to researcher task requests
            on_status_change: Callback function to run for changing node agent status
        """
        super().__init__(server_state, channels)

        self._queue = asyncio.Queue()
        self._on_status_change = on_status_change
        self._retry_count = 0

    async def _listen(self, callback: Optional[Callable] = None) -> None:
        """Listens for the messages that are going to be sent to researcher.

        Args:
            callback: Callback to execute once a task is received

        Raises:
            FedbiomedCommunicationError: communication error with researcher
        """

        # While loop retires to send if first one fails to send the result
        while True:

            await self._server_state.connection_is_ready()

            # Waits until there is something to send back to researcher
            try:
                await self._get(callback)
            except grpc.aio.AioRpcError as exp:
                match exp.code():
                    case grpc.StatusCode.DEADLINE_EXCEEDED:
                        self._on_status_change(ClientStatus.DISCONNECTED)
                        logger.warning(
                            "Researcher not answering after timeout, looks like server failure or disconnect. "
                            "Discard message.")
                        self._queue.task_done()
                    case grpc.StatusCode.UNAVAILABLE:
                        self._on_status_change(ClientStatus.DISCONNECTED)
                        self._server_state.reassure_connection()
                        self._retry_count += 1
                    case grpc.StatusCode.UNKNOWN:
                        self._on_status_change(ClientStatus.FAILED)
                        logger.error("Unexpected error raised by researcher gRPC server. This is probably due to "
                                     f"bug on the researcher side: {exp}")
                    case _:
                        self._on_status_change(ClientStatus.FAILED)

            except Exception as exp:
                self._on_status_change(ClientStatus.FAILED)
                raise FedbiomedCommunicationError(
                    f"{ErrorNumbers.FB628}: Sender has stopped due to unknown reason: {exp}") from exp


    async def _get(self, callback: Optional[Callable] = None) -> None:
        """Gets task result from the queue.

        Args:
            callback: Callback to execute once a task is received
        """

        while True:
            if self._retry_count > 5:
                logger.warning("Message can not be sent to researcher after 5 retries")
                self._queue.task_done()

            msg = await self._queue.get()

            # If it is a Unary-Unary RPC call
            if isinstance(msg["stub"], grpc.aio.UnaryUnaryMultiCallable):
                await msg["stub"](msg["message"])

            elif isinstance(msg["stub"], grpc.aio.StreamUnaryMultiCallable):
                stream_call = msg["stub"]()

                if isinstance(callback, Callable):
                    # we could check the callback prototype
                    callback(msg["message"])

                for reply in self._stream_reply(msg["message"]):
                    await stream_call.write(reply)

                await stream_call.done_writing()

            else:
                raise FedbiomedCommunicationError(
                    "Unknown type of stub has been in gRPC Sender listener {msg['stub']}"
                )

            self._queue.task_done()
            self._retry_count = 0


    def _stream_reply(self, message: Message) -> Iterable:
        """Streams task result back researcher component.

        Args:
            message: Message to stream

        Returns:
            A stream of researcher reply chunks
        """

        reply = Serializer.dumps(message.get_dict())
        chunk_range = range(0, len(reply), MAX_MESSAGE_BYTES_LENGTH)
        for start, iter_ in zip(chunk_range, range(1, len(chunk_range) + 1)):
            stop = start + MAX_MESSAGE_BYTES_LENGTH
            yield TaskResult(
                size=len(chunk_range),
                iteration=iter_,
                bytes_=reply[start:stop]
            ).to_proto()

    async def send(self, message: Message) -> None:
        """Send a message to peer researcher.

        Args:
            message: Message to send
        """
        # Switch-case for message type and gRPC calls
        match message.__class__.__name__:
            case FeedbackMessage.__name__:
                # Note: FeedbackMessage is designed as proto serializable message.
                await self._queue.put({"stub": self._channels.feedback_stub.Feedback,
                                       "message": message.to_proto()})

            case _:
                await self._queue.put({"stub": self._channels.task_stub.ReplyTask,
                                       "message": message})

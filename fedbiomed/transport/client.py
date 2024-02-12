from typing import List, Callable, Optional, Awaitable, Iterable
from enum import Enum
import asyncio
import abc
import ssl
import socket
import time

from dataclasses import dataclass

import grpc

from fedbiomed.transport.protocols.researcher_pb2_grpc import ResearcherServiceStub

from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.message import Message, TaskRequest, TaskResult, FeedbackMessage
from fedbiomed.common.constants import MAX_MESSAGE_BYTES_LENGTH, MAX_SEND_RETRIES, ErrorNumbers
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


class _StubType(Enum):
    NO_STUB = 0        # never matcher stub type
    ANY_STUB = 1       # always matches stub type
    LISTENER_TASK_STUB = 2
    SENDER_TASK_STUB = 3
    SENDER_FEEDBACK_STUB = 4


# timeout in seconds for retrying connection to the server when it does not reply or returns an error
GRPC_CLIENT_CONN_RETRY_TIMEOUT = 2

# timeout in seconds of a request to the server for a task (payload) to run on the node
GRPC_CLIENT_TASK_REQUEST_TIMEOUT = 3600


def is_server_alive(host: str, port: str):
    """Checks if the server is alive

    Args:
        host: The host/ip of researcher/server component
        port: Port number of researcher/server component
    """

    port = int(port)
    address_info = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    for family, socktype, protocol, _ , address in address_info:
        s = socket.socket(family, socktype, protocol)
        # Need this timeout for the case where the server does not answer
        # If not present, socket timeout increases and this function takes more
        # than GRPC_CLIENT_CONN_RETRY_TIMEOUT to execute
        s.settimeout(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
        try:
            s.connect(address)
        except socket.error:
            return False
        else:
            s.close()
            return True


class Channels:
    """Keeps gRPC server channels"""

    def __init__(self, researcher: ResearcherCredentials):
        """Create channels and stubs

        Args:
            researcher: An instance of ResearcherCredentials
        """
        self._researcher = researcher

        self._channels = {}
        self._stubs = {}
        self._stub_types = [
            _StubType.LISTENER_TASK_STUB,
            _StubType.SENDER_TASK_STUB,
            _StubType.SENDER_FEEDBACK_STUB
        ]
        for st in self._stub_types:
            self._channels[st]: grpc.aio.Channel = None
            self._stubs[st]: ResearcherServiceStub = None

        # lock for accessing channels and stubs
        self._channels_stubs_lock = asyncio.Lock()

    async def stub(self, stub_type: _StubType) -> ResearcherServiceStub:
        """Get stub for a given stub type.

        Args:
            stub_type: the stub type to get

        Returns:
            the stub if it exists or None
        """
        if stub_type in self._stub_types:
            async with self._channels_stubs_lock:
                return self._stubs[stub_type]
        else:
            return None

    async def connect(self, stub_type: _StubType = _StubType.ANY_STUB):
        """Connects gRPC server and instatiates stubs.

        Args:
            stub_type: only (re)connect for matching stub type(s)
        """

        async with self._channels_stubs_lock:
            # Closes if channels are open
            for st, channel in self._channels.items():
                if channel and (stub_type == _StubType.ANY_STUB or stub_type == st):
                    await channel.close()

            # Creates channels
            for st in self._channels.keys():
                if stub_type == _StubType.ANY_STUB or stub_type == st:
                    self._channels[st] = self._create()
                    self._stubs[st] = ResearcherServiceStub(channel=self._channels[st])

    def _create(self):
        """Creates new channel"""
        return self._create_channel(
            port=self._researcher.port,
            host=self._researcher.host,
            certificate= grpc.ssl_channel_credentials(self._researcher.certificate))

    @staticmethod
    def _create_channel(
        port: str,
        host: str,
        certificate: Optional[str] = None
    ) -> grpc.Channel :
        """Create gRPC channel

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
            #
            # Some references for configuring gRPC keepalive:
            # https://github.com/grpc/proposal/blob/master/A8-client-side-keepalive.md
            # https://github.com/grpc/proposal/blob/master/A9-server-side-conn-mgt.md
            # https://github.com/grpc/grpc/blob/master/doc/keepalive.md
            # https://github.com/grpc/grpc/blob/master/examples/python/keep_alive/greeter_client.py
            # https://github.com/grpc/grpc/blob/master/examples/python/keep_alive/greeter_server.py
            # https://www.evanjones.ca/grpc-is-tricky.html
            # https://www.evanjones.ca/tcp-connection-timeouts.html
            # Be sure to keep client-server configuration coherent
            ("grpc.keepalive_time_ms", 30 * GRPC_CLIENT_CONN_RETRY_TIMEOUT * 1000),
            ("grpc.keepalive_timeout_ms", 2 * 1000),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.keepalive_permit_without_calls", 1),
            #
            ("grpc.initial_reconnect_backoff_ms", 1000),
            ("grpc.min_reconnect_backoff_ms", 500),
            ("grpc.max_reconnect_backoff_ms", 2000),
            # ('grpc.ssl_target_name_override', 'localhost') # ...
            ("grpc.enable_retries", 1),
            # ("grpc.service_config", service_config)

        ]

        if certificate is None:
            channel = grpc.aio.insecure_channel(f"{host}:{port}", options=channel_options)
        else:
            channel = grpc.aio.secure_channel(f"{host}:{port}", certificate, options=channel_options)

        return channel


class GrpcClient:
    """An agent of remote researcher gRPC server."""

    def __init__(
        self,
        node_id: str,
        researcher: ResearcherCredentials,
        update_id_map: Awaitable
    ) -> None:
        """Class constructor

        Args:
            node_id: unique ID of this node (connection client)
            researcher: the researcher to which the node connects (connection server)
            update_id_map: awaitable to call when updating the researcher ID, needs proper prototype
        """
        self._id = None
        self._researcher = researcher
        self._channels = Channels(researcher)

        self._task_listener = TaskListener(
            channels=self._channels,
            node_id=node_id,
            on_status_change = self._on_status_change,
            update_id=self._update_id)

        self._sender = Sender(
            channels=self._channels,
            on_status_change = self._on_status_change)

        # TODO: use `self._status` for finer gRPC agent handling.
        # Currently, the (tentative) status is maintained but not used
        self._status  = ClientStatus.DISCONNECTED
        # lock for accessing self._status
        self._status_lock = asyncio.Lock()

        self._update_id_map = update_id_map
        self._tasks = []

    def start(self, on_task) -> List[Awaitable[Optional[Callable]]]:
        """Start researcher gRPC agent.

        Starts long-lived tasks, one waiting for server requests, one waiting on the async queue
        for the replies from the node that are going to be sent back to researcher.

        Args:
            on_task: Callback function to execute once a payload received from researcher.

        Returns:
            A list of task objects of the agent
        """

        async def run():
            """Connects and dispatches the tasks"""

            # First connects to channel
            await self._connect()

            # Launch listeners
            await asyncio.gather(
                self._task_listener.listen(on_task),
                self._sender.listen()
            )

        # Returns client task
        return asyncio.create_task(run())


    async def send(self, message: Message) -> None:
        """Sends messages from node to researcher server.

        Args:
            message: message to send from node to server
        """

        await self._sender.send(message)


    async def _connect(self):
        """Updates connection state and dispatch event to run listeners

        This method also implements auto-trust for server certificate
        """

        while True:
            time_before = time.perf_counter()
            if is_server_alive(self._researcher.host, self._researcher.port):
                # Gets server certificate before creating the channel
                # This implementation assumes that the provided IP and PORT trusted
                # == OK for honest but curious researcher and nodes (parties in the
                # network instance) but subject to attack by malicious MITM at each
                # connection to server
                #
                # TODO: implement configurable policy instead of hardcoded current version
                # in the future
                self._researcher.certificate = \
                    bytes(ssl.get_server_certificate(
                        (self._researcher.host, self._researcher.port)),
                        'utf-8')
                logger.info("Retrieved server certificate, ready to communicate with server.")

                # Connect to channels and create stubs
                await self._channels.connect()

                break
            else:
                logger.info(
                    "Researcher server is not available, will retry connect in "
                    f"{GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds")
                await asyncio.sleep(max(0, GRPC_CLIENT_CONN_RETRY_TIMEOUT - time.perf_counter() + time_before))


    async def _on_status_change(self, status: ClientStatus) -> None:
        """Callback awaitable to change the researcher status

        Args:
            status: New status of the researcher client
        """
        async with self._status_lock:
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

    def __init__(self, channels: Channels) -> None:
        """Constructs task listener channels

        Args:
            channels: Keeps channels and stubs.
        """
        self._channels = channels

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
            channels: Channels,
            node_id: str,
            on_status_change: Awaitable,
            update_id: Awaitable
    ) -> None:
        """Class constructor.

        Args:
            channels: RPC channels and stubs to be used for polling tasks from researcher
            node_id: unique ID for this node
            on_status_change: Callback awaitable to run for changing node agent status
            update_id: Callback function to run updating peer researcher ID
        """
        super().__init__(channels)

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
            try:
                await self._request(callback)
            except grpc.aio.AioRpcError as exp:
                match exp.code():
                    case grpc.StatusCode.DEADLINE_EXCEEDED:
                        logger.debug(
                            "Researcher did not request executing a task before timeout. Send new task request")
                    case grpc.StatusCode.UNAVAILABLE:
                        await self._on_status_change(ClientStatus.DISCONNECTED)
                        logger.info(
                            "Researcher server is not available, will retry connect in "
                            f"{GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds")
                        await asyncio.sleep(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
                        await self._channels.connect(_StubType.LISTENER_TASK_STUB)

                    case grpc.StatusCode.UNKNOWN:
                        await self._on_status_change(ClientStatus.FAILED)
                        logger.error("Unexpected error raised by researcher gRPC server. This is probably due to "
                                     f"bug on the researcher side: {exp}. Will retry connect in "
                                     f"{GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds")
                        await asyncio.sleep(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
                        await self._channels.connect(_StubType.LISTENER_TASK_STUB)
                    case _:
                        await self._on_status_change(ClientStatus.FAILED)
                        logger.error("Unhandled gRPC call status {exp.code()}. Exception: {exp}. Will retry connect in "
                                     f"{GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds")
                        await asyncio.sleep(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
                        await self._channels.connect(_StubType.LISTENER_TASK_STUB)

            except Exception as exp:
                await self._on_status_change(ClientStatus.FAILED)
                raise FedbiomedCommunicationError(
                    f"{ErrorNumbers.FB628}: Task listener has stopped due to unknown reason: {exp}") from exp


    async def _request(self, callback: Optional[Callable] = None) -> None:
        """Requests tasks from Researcher

        Args:
            callback: Callback to execute once a task is arrived
        """
        while True:
            logger.debug("Sending new task request to researcher")
            await self._on_status_change(ClientStatus.CONNECTED)
            request_stub = await self._channels.stub(_StubType.LISTENER_TASK_STUB)
            iterator = request_stub.GetTaskUnary(
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
        channels: Channels,
        on_status_change: Awaitable,
    ) -> None:
        """Class constructor.

        Args:
            channels: RPC channels and stubs to be used for polling tasks from researcher
            on_status_change: Callback awaitable to run for changing node agent status
        """
        super().__init__(channels)

        self._queue = asyncio.Queue()
        self._on_status_change = on_status_change
        self._retry_count = 0
        self._stub_type = _StubType.NO_STUB

    async def _listen(self, callback: Optional[Callable] = None) -> None:
        """Listens for the messages that are going to be sent to researcher.

        Args:
            callback: Callback to execute once a task is received

        Raises:
            FedbiomedCommunicationError: communication error with researcher
        """

        # While loop retires to send if first one fails to send the result
        while True:
            try:
                await self._get(callback)
            except grpc.aio.AioRpcError as exp:
                match exp.code():
                    case grpc.StatusCode.DEADLINE_EXCEEDED:
                        await self._on_status_change(ClientStatus.DISCONNECTED)
                        logger.warning(
                            "Researcher not answering after timeout, looks like server failure or disconnect. "
                            "Discard message.")
                        self._queue.task_done()
                        self._retry_count = 0
                    case grpc.StatusCode.UNAVAILABLE:
                        await self._on_status_change(ClientStatus.DISCONNECTED)
                        logger.info(
                            "Researcher server is not available, will retry connect in "
                            f"{GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds")
                        await asyncio.sleep(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
                        await self._channels.connect(self._stub_type)
                        self._retry_count += 1
                    case grpc.StatusCode.UNKNOWN:
                        await self._on_status_change(ClientStatus.FAILED)
                        logger.error("Unexpected error raised by researcher gRPC server. This is probably due to "
                                     f"bug on the researcher side: {exp}")
                    case _:
                        await self._on_status_change(ClientStatus.FAILED)

            except Exception as exp:
                await self._on_status_change(ClientStatus.FAILED)
                raise FedbiomedCommunicationError(
                    f"{ErrorNumbers.FB628}: Sender has stopped due to unknown reason: {exp}") from exp

            except GeneratorExit as exp:
                await self._on_status_change(ClientStatus.FAILED)
                raise FedbiomedCommunicationError(
                    f"{ErrorNumbers.FB628}: Sender has stopped due to unexpected gRPC abort: {exp}") from exp


    async def _get(self, callback: Optional[Callable] = None) -> None:
        """Gets task result from the queue.

        Args:
            callback: Callback to execute once a task is received
        """

        while True:
            # initialize in case of early failure
            self._stub_type = _StubType.NO_STUB

            if self._retry_count > MAX_SEND_RETRIES:
                logger.warning(
                    f"Message can not be sent to researcher after {MAX_SEND_RETRIES} retries. Discard message.")
                self._queue.task_done()
                self._retry_count = 0

            msg = await self._queue.get()
            self._stub_type = msg["stub"]
            if self._stub_type == _StubType.SENDER_FEEDBACK_STUB:
                feedback_stub = await self._channels.stub(_StubType.SENDER_FEEDBACK_STUB)
                stub_function = feedback_stub.Feedback
            elif self._stub_type == _StubType.SENDER_TASK_STUB:
                task_stub = await self._channels.stub(_StubType.SENDER_TASK_STUB)
                stub_function = task_stub.ReplyTask
            else:
                raise FedbiomedCommunicationError(
                    "Unknown type of stub in gRPC Sender listener {msg['stub']}"
                )

            # If it is a Unary-Unary RPC call
            if isinstance(stub_function, grpc.aio.UnaryUnaryMultiCallable):
                await stub_function(msg["message"])

            elif isinstance(stub_function, grpc.aio.StreamUnaryMultiCallable):
                stream_call = stub_function()

                for reply in self._stream_reply(msg["message"]):
                    await stream_call.write(reply)

                await stream_call.done_writing()

                if isinstance(callback, Callable):
                    # we could check the callback prototype
                    callback(msg["message"])

            else:
                raise FedbiomedCommunicationError(
                    "Unknown type of stub built from gRPC Sender listener {msg['stub']}"
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
                await self._queue.put({"stub": _StubType.SENDER_FEEDBACK_STUB,
                                       "message": message.to_proto()})

            case _:
                await self._queue.put({"stub": _StubType.SENDER_TASK_STUB,
                                       "message": message})

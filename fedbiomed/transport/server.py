
import time
import os
from typing import Callable, Iterable, Any, Coroutine, Optional, List
import threading

import asyncio
import grpc
from google.protobuf.message import Message as ProtoBufMessage

from fedbiomed.transport.protocols.researcher_pb2 import Empty
import fedbiomed.transport.protocols.researcher_pb2_grpc as researcher_pb2_grpc
from fedbiomed.transport.client import GRPC_CLIENT_CONN_RETRY_TIMEOUT, GRPC_CLIENT_TASK_REQUEST_TIMEOUT
from fedbiomed.transport.node_agent import AgentStore, NodeAgent

from fedbiomed.common.constants import ErrorNumbers, MAX_SEND_RETRIES
from fedbiomed.common.exceptions import FedbiomedCommunicationError
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.message import (
    Message,
    TaskResponse,
    TaskRequest,
    FeedbackMessage,
    OverlayMessage,
)

from fedbiomed.common.constants import MessageType, MAX_MESSAGE_BYTES_LENGTH


# Maximum time in seconds for sending a message, before considering it should be discarded.
MAX_SEND_DURATION = 300

# timeout in seconds for server to establish connections with nodes and initialize

server_setup_timeout = int(os.getenv('GRPC_SERVER_SETUP_TIMEOUT', 1))

GRPC_SERVER_SETUP_TIMEOUT = GRPC_CLIENT_CONN_RETRY_TIMEOUT + server_setup_timeout
MAX_GRPC_SERVER_SETUP_TIMEOUT = 20 * server_setup_timeout


class SSLCredentials:
    """Contains credentials for SSL certifcate of the gRPC server"""
    def __init__(self, key: str, cert: str):
        """Reads private key and cert file

        Args:
            key: path to private key
            cert: path to certificate
        """
        with open(key, 'rb') as f:
            self.private_key = f.read()
        with open(cert, 'rb') as f:
            self.certificate = f.read()


class ResearcherServicer(researcher_pb2_grpc.ResearcherServiceServicer):
    """RPC Servicer """


    def __init__(
            self,
            agent_store: AgentStore,
            on_message: Callable
    ) -> None:
        """Constructor of gRPC researcher servicer

        Args:
            agent_store: The class that stores node agents
            on_message: Callback function to execute once a message received from the nodes
        """
        super().__init__()
        self._agent_store = agent_store
        self._on_message = on_message


    async def GetTaskUnary(
            self,
            request: ProtoBufMessage,
            context: grpc.aio.ServicerContext
    ) -> None:
        """Gets unary RPC request and return stream of response

        Args:
            request: RPC request
            context: RPC peer context
        """
        task_request = TaskRequest.from_proto(request).get_dict()
        logger.debug(f"Node: {task_request.get('node')} polling for the tasks")

        node_agent = await self._agent_store.retrieve(node_id=task_request["node"])

        # Update node active status as active
        await node_agent.set_active()

        task = None
        try:
            while True:
                task, retry_count, first_send_time = await node_agent.get_task()

                # Choice: mark task as de-queued as soon only if really sent
                node_agent.task_done()

                # discard if message too old
                if first_send_time + MAX_SEND_DURATION > time.time():
                    break
                else:
                    task = None
                    logger.warning(f"Message to send is older than {MAX_SEND_DURATION} seconds. Discard message.")

            task_bytes = Serializer.dumps(task.to_dict())

            chunk_range = range(0, len(task_bytes), MAX_MESSAGE_BYTES_LENGTH)
            for start, iter_ in zip(chunk_range, range(1, len(chunk_range) + 1)):
                stop = start + MAX_MESSAGE_BYTES_LENGTH

                try:
                    yield TaskResponse(
                        size=len(chunk_range),
                        iteration=iter_,
                        bytes_=task_bytes[start:stop]
                    ).to_proto()
                except GeneratorExit:
                    # schedule resend if task sending could not be completed
                    # => retry send as long as (1) send not successful
                    # (2) max retries not reached
                    # => else discard message
                    #
                    # Note: if node is disconnected then back online, message is retried after reconnection.
                    # This is not fully coherent with upper layers (Requests) that may trigger an application
                    # level failure in the while, but it is mitigated by the MAX_SEND_DURATION
                    if retry_count < MAX_SEND_RETRIES:
                        await node_agent.send_async(
                            message=task, on_reply=None, retry_count=retry_count + 1, first_send_time=first_send_time
                        )
                    else:
                        logger.warning(f"Message cannot be sent after {MAX_SEND_RETRIES} retries. Discard message.")
                    await node_agent.change_node_status_after_task()
                    # need return here to avoid RuntimeError
                    return

        except asyncio.CancelledError:
            if task is not None and retry_count is not None and first_send_time is not None:
                # schedule resend if task was pulled from queue
                if retry_count < MAX_SEND_RETRIES:
                    await node_agent.send_async(
                        message=task, on_reply=None, retry_count=retry_count + 1, first_send_time=first_send_time
                    )
                else:
                    logger.warning(f"Message cannot be sent after {MAX_SEND_RETRIES} retries. Discard message.")
        finally:
            await node_agent.change_node_status_after_task()


    async def ReplyTask(
            self,
            request_iterator: Iterable[ProtoBufMessage],
            unused_context: grpc.aio.ServicerContext
    ) -> None:
        """Gets stream replies from the nodes

        Args:
            request_iterator: Iterator for streaming
            unused_context: Request service context
        """

        reply = bytes()
        async for answer in request_iterator:
            reply += answer.bytes_
            if answer.size != answer.iteration:
                continue

            # Deserialize message
            message = Serializer.loads(reply)

            # Replies are handled by node agent callbacks
            node = await self._agent_store.get(message["node_id"])
            await node.on_reply(message)

            reply = bytes()

        return Empty()


    async def Feedback(
            self,
            request: ProtoBufMessage,
            unused_context: grpc.aio.ServicerContext
    ) -> None:
        """Executed for Feedback request received from the nodes

        Args:
            request: Feedback message
            unused_context: Request service context
        """

        # Get the type of Feedback | log or scalar
        one_of = request.WhichOneof("feedback_type")
        feedback = FeedbackMessage.from_proto(request)

        # Execute on_message assigned by the researcher.requests modules
        self._on_message(feedback.get_param(one_of), MessageType.convert(one_of))

        return Empty()


class _GrpcAsyncServer:
    """GRPC Server class.

    All the methods of this class are awaitable, except the constructor.
    """
    def __init__(
            self,
            host: str,
            port: str,
            on_message: Callable,
            ssl: SSLCredentials,
            debug: bool = False,
    ) -> None:
        """Class constructor

        Args:
            host: server DNS name or IP address
            port: server TCP port
            on_message: Callback function to execute once a message received from the nodes
            ssl: Ssl credentials.
            debug: Activate debug mode for gRPC asyncio
        """

        # inform all threads whether server is started
        self._is_started = threading.Event()
        self._ssl = ssl
        self._host = host
        self._port = port

        self._server = None
        self._debug = debug
        self._on_message = on_message
        self._loop = None
        self._agent_store : Optional[AgentStore] = None


    async def start(self):
        """Starts gRPC server"""

        self._server = grpc.aio.server(
            # futures.ThreadPoolExecutor(max_workers=10),
            options=[
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
                ("grpc.http2.min_ping_interval_without_data_ms", 0.9 * GRPC_CLIENT_CONN_RETRY_TIMEOUT * 1000),
                ("grpc.max_connection_idle_ms", (GRPC_CLIENT_TASK_REQUEST_TIMEOUT + 2) * 1000),
                ("grpc.max_connection_age_ms", (GRPC_CLIENT_TASK_REQUEST_TIMEOUT + 5) * 1000),
                ("grpc.max_connection_age_grace_ms", 2 * 1000),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.keepalive_permit_without_calls", 1),
                #
                ("grpc.http2.max_ping_strikes", 100),
                #
                # Prevent multiple servers on same port
                ('grpc.so_reuseport', 0),
            ])

        self._loop = asyncio.get_running_loop()
        self._agent_store = AgentStore(loop=self._loop, on_forward=self._on_forward)

        researcher_pb2_grpc.add_ResearcherServiceServicer_to_server(
            ResearcherServicer(
                agent_store=self._agent_store,
                on_message=self._on_message),
            server=self._server
        )

        # TODO: current version does not require or check client certificate
        # In other words: hardcoded policy that researcher does not check node identity yet.
        # To be extended in a future version.
        server_credentials = grpc.ssl_server_credentials(
            ( (self._ssl.private_key, self._ssl.certificate), )
        )

        self._server.add_secure_port(self._host + ':' + str(self._port), server_credentials)
        # self._server.add_insecure_port(self._host + ':' + str(self._port))

        # Starts async gRPC server
        await self._server.start()

        self._is_started.set()
        try:
            if self._debug:
                logger.debug("Waiting for gRPC server termination")
            await self._server.wait_for_termination()
        finally:
            if self._debug:
                logger.debug("gRPC server has stopped")

    async def _on_forward(self, message: OverlayMessage) -> None:
        """Handle overlay messages received by the server by forwarding them to the destination node.

        Args:
            message: Message to forward
        """
        # caveat: intentionally use `_GrpcAyncServer.send()`
        # if using `self.send()` it uses `GrpcServer.send()`, normally used from another thread
        # if using `super().send()` it's less explicit
        await _GrpcAsyncServer.send(self, message, message.dest_node_id)


    async def send(self, message: Message, node_id: str) -> None:
        """Send given message to a given client

        Args:
            message: Message to broadcast
            node_id: unique ID of node
        """

        agent = await self._agent_store.get(node_id)

        if not agent:
            logger.info(f"Node {node_id} is not registered on server. Discard message.")
            return

        await agent.send_async(message)


    async def broadcast(self, message: Message) -> None:
        """Broadcasts given message to all active clients.

        Args:
            message: Message to broadcast
        """

        agents = await self._agent_store.get_all()
        for _, agent in agents.items():
            await agent.send_async(message)

    async def get_node(self, node_id: str) -> Optional[NodeAgent]:
        """Returns given node

        Args:
            node_id: ID of node to retrieve

        Returns:
            A node agent
        """

        return await self._agent_store.get(node_id)

    async def get_all_nodes(self) -> List[NodeAgent]:
        """Returns all known nodes

        Returns:
            A list of node agents
        """

        agents = await self._agent_store.get_all()

        return [node for node in agents.values()]



class GrpcServer(_GrpcAsyncServer):
    """Grpc server implementation to be used by threads

    This class extends async implementation of gRPC server to be able to
    call async methods from different thread. Currently, it is used by
    [fedbiomed.researcher.requests.Requests][`Requests`] class that is
    instantiated in the main thread

    Attributes:
        _thread: background thread of gRPC server
    """

    _thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        """Runs asyncio application"""
        try:
            asyncio.run(super().start())
        except Exception as e:
            logger.error(f"Researcher gRPC server has stopped. Please try to restart: {e}")

    def start(self) -> None:
        """Starts async GrpcServer """

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # FIXME: This implementation assumes that nodes will be able connect and server complete setup with this delay
        logger.info("Starting researcher service...")


        logger.info(f'Waiting {GRPC_SERVER_SETUP_TIMEOUT}s for nodes to connect...')
        time.sleep(GRPC_SERVER_SETUP_TIMEOUT)

        sleep_ = 0
        while len(self.get_all_nodes()) == 0:

            if sleep_ == 0:
                logger.info(f"No nodes found, server will wait "
                            f"{MAX_GRPC_SERVER_SETUP_TIMEOUT - GRPC_SERVER_SETUP_TIMEOUT} "
                            "more seconds until a node creates connection.")

            if sleep_ > MAX_GRPC_SERVER_SETUP_TIMEOUT - GRPC_SERVER_SETUP_TIMEOUT:
                if len(self.get_all_nodes()) == 0:
                    logger.warning("Server has not received connection from any remote nodes in "
                                   f"MAX_GRPC_SERVER_SETUP_TIMEOUT: {MAX_GRPC_SERVER_SETUP_TIMEOUT} "
                                   "This may effect the request created right after the server initialization. "
                                   "However, server will keep running in the background so you can retry the "
                                   "operations for sending requests to remote nodes until one receives.")
                break

            time.sleep(1)
            sleep_ += 1


    def send(self, message: Message, node_id: str) -> None:
        """Send message to a specific node.

        Args:
            message: Message to send
            node_id: Destination node unique ID

        Raises:
            FedbiomedCommunicationError: bad argument type
            FedbiomedCommunicationError: server is not started
        """
        if not isinstance(message, Message):
            raise FedbiomedCommunicationError(
                f"{ErrorNumbers.FB628}: bad argument type for message, expected `Message`, got `{type(message)}`")

        if not self._is_started.is_set():
            raise FedbiomedCommunicationError(
                f"{ErrorNumbers.FB628.value}: Can not send message. "
                "Communication client is not initialized.")

        self._run_threadsafe(super().send(message, node_id))


    def broadcast(self, message: Message) -> None:
        """Broadcast message to all known and reachable nodes

        Args:
            message: Message to broadcast

        Raises:
            FedbiomedCommunicationError: bad argument type
            FedbiomedCommunicationError: server is not started
        """
        if not isinstance(message, Message):
            raise FedbiomedCommunicationError(
                f"{ErrorNumbers.FB628}: bad argument type for message, expected `Message`, got `{type(message)}`")

        if not self._is_started.is_set():
            raise FedbiomedCommunicationError(
                f"{ErrorNumbers.FB628}: Can not broadcast given message. "
                "Communication client is not initialized.")

        self._run_threadsafe(super().broadcast(message))

    def get_all_nodes(self) -> List[NodeAgent]:
        """Returns all known nodes

        Returns:
            A list of node agents

        Raises:
            FedbiomedCommunicationError: server is not started
        """
        if not self._is_started.is_set():
            raise FedbiomedCommunicationError(
                f"{ErrorNumbers.FB628}: Error while getting all nodes "
                "connected:  Communication client is not initialized.")

        return self._run_threadsafe(super().get_all_nodes())

    def get_node(self, node_id) -> Optional[NodeAgent]:
        """Returns given node

        Args:
            node_id: ID of node to retrieve

        Returns:
            A node agent

        Raises:
            FedbiomedCommunicationError: server is not started
        """
        if not self._is_started.is_set():
            raise FedbiomedCommunicationError(
                f"{ErrorNumbers.FB628}: Error while getting node '{node_id}':"
                "Communication client is not initialized.")

        return self._run_threadsafe(super().get_node(node_id))

    # TODO: Currently unused

    def is_alive(self) -> bool:
        """Checks if the thread running gRPC server still alive

        Returns:
            gRPC server running status

        Raises:
            FedbiomedCommunicationError: server is not started
        """
        if not self._is_started.is_set():
            raise FedbiomedCommunicationError(
                f"{ErrorNumbers.FB628}: Can not check if thread is alive."
                "Communication client is not initialized.")

        # TODO: more tests about gRPC server and task status ?
        return False if not isinstance(self._thread, threading.Thread) else self._thread.is_alive()

    def _run_threadsafe(self, coroutine: Coroutine) -> Any:
        """Runs given coroutine threadsafe

        Args:
            coroutine: Awaitable function to be executed as threadsafe

        Returns:
            Coroutine return value.
        """

        future = asyncio.run_coroutine_threadsafe(
            coroutine, self._loop
        )


        return future.result()

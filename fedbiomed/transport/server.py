# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import threading
import time
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Optional

import grpc
from google.protobuf.message import Message as ProtoBufMessage

import fedbiomed.transport.protocols.researcher_pb2_grpc as researcher_pb2_grpc
from fedbiomed.common.certificate_manager import (
    TrustedCertificateBundle,
    certificate_audit_fields,
)
from fedbiomed.common.config import Config
from fedbiomed.common.constants import (
    MAX_MESSAGE_BYTES_LENGTH,
    MAX_SEND_RETRIES,
    ErrorNumbers,
    MessageType,
)
from fedbiomed.common.exceptions import (
    FedbiomedCertificateError,
    FedbiomedCommunicationError,
)
from fedbiomed.common.logger import logger
from fedbiomed.common.message import (
    FeedbackMessage,
    Message,
    OverlayMessage,
    TaskRequest,
    TaskResponse,
)
from fedbiomed.common.serializer import Serializer
from fedbiomed.transport.client import (
    GRPC_CLIENT_CONN_RETRY_TIMEOUT,
    GRPC_CLIENT_TASK_REQUEST_TIMEOUT,
)
from fedbiomed.transport.node_agent import AgentStore, NodeAgent
from fedbiomed.transport.protocols.researcher_pb2 import Empty

# Maximum time in seconds for sending a message, before considering it should be discarded.
MAX_SEND_DURATION = 300

# timeout in seconds for server to establish connections with nodes and initialize

server_setup_timeout = int(os.getenv("GRPC_SERVER_SETUP_TIMEOUT", 1))

GRPC_SERVER_SETUP_TIMEOUT = GRPC_CLIENT_CONN_RETRY_TIMEOUT + server_setup_timeout
MAX_GRPC_SERVER_SETUP_TIMEOUT = 20 * server_setup_timeout

# gRPC service nodes connect to, reported as the destination of connection events.
_SERVICE_NAME = "researcher.ResearcherService"


class SSLCredentials:
    """Contains credentials for SSL certificate of the gRPC server"""

    def __init__(
        self,
        key: str,
        cert: str,
        trusted_node_certificates: Optional[TrustedCertificateBundle] = None,
    ):
        """Reads private key and cert file

        Args:
            key: path to private key
            cert: path to certificate
            trusted_node_certificates: view of the registered node certificates.
                Called for its PEM bundle on each mutual TLS handshake, so nodes
                registered after startup are trusted without a restart, and asked
                which party a presented certificate belongs to when serving an
                RPC. None disables node identity verification (server-auth only).
        """
        with open(key, "rb") as f:
            self.private_key = f.read()
        with open(cert, "rb") as f:
            self.certificate = f.read()
        self.trusted_node_certificates = trusted_node_certificates

    @property
    def mtls(self) -> bool:
        """Whether mutual TLS (node certificate verification) is enabled."""
        return self.trusted_node_certificates is not None


def _peer_certificate(context: grpc.aio.ServicerContext) -> Optional[bytes]:
    """PEM of the peer client certificate, or None when none was presented.

    Args:
        context: RPC peer context.

    Returns:
        The presented client certificate, or None under server-only TLS.
    """
    pem = context.auth_context().get("x509_pem_cert")
    if not pem:
        return None

    certificate = pem[0]
    if not isinstance(certificate, bytes):
        certificate = certificate.encode("utf-8")

    return certificate


def _connection_audit_fields(context: grpc.aio.ServicerContext) -> Dict[str, str]:
    """Connection facts common to every audit event of an RPC call.

    Args:
        context: RPC peer context.

    Returns:
        Origin and destination of the call, plus the identifying fields of the
        peer certificate when one was presented.
    """
    return {
        "source_address": context.peer(),
        "destination_service": _SERVICE_NAME,
        **certificate_audit_fields(_peer_certificate(context)),
    }


async def _verify_peer_identity(
    context: grpc.aio.ServicerContext,
    declared_node_id: Optional[str],
    identities: Optional[TrustedCertificateBundle],
) -> Optional[str]:
    """Binds the node id a message declares to the peer's registered identity.

    Applied to every RPC carrying a node id, so a node holding a registered
    certificate cannot act under another node's identity. The identity is the
    party id the presented certificate is registered under, which is
    authoritative even for certificates embedding no Fed-BioMed identity.

    Resolution goes through the same registry view that supplies the TLS trust
    bundle, so a certificate deleted while the researcher runs stops being
    accepted once the change is picked up. A certificate that resolves to no
    party id is refused: the handshake proved it chains to the trusted bundle,
    so an unresolved one means the registry and that bundle disagree.

    Args:
        context: RPC peer context.
        declared_node_id: Node id declared by the message being served.
        identities: Registered certificates of the peer's component type, or
            None when node identity verification is disabled.

    Returns:
        The registered party id of the peer, or None when no client certificate
        was presented — under server-only TLS there is no identity to bind to.

    Raises:
        grpc.aio.AbortError: the peer could not be resolved to a registered
            party id, or that party id is not the one declared.
    """
    certificate = _peer_certificate(context)
    if certificate is None:
        return None

    peer_node_id = identities.party_id(certificate) if identities else None

    if peer_node_id is None:
        # Name which of the three ways resolution came up empty, so a broken
        # registry does not read as an unregistered certificate.
        if identities is None:
            reason = "no_registry_configured"
            cause = "no node certificate registry is configured on this researcher"
        elif not identities.loaded:
            reason = "registry_unreadable"
            cause = "its certificate registry could not be read"
        else:
            reason = "certificate_not_registered"
            cause = "its certificate is not registered"

        msg = (
            f"{ErrorNumbers.FB628.value}: Refusing the node declaring id "
            f"`{declared_node_id}`: {cause}."
        )
        logger.error(msg)
        logger.security_event(
            operation="mtls_identity_unresolved",
            status="failure",
            reason=reason,
            declared_node_id=declared_node_id,
            detail=msg,
            **_connection_audit_fields(context),
        )
        await context.abort(grpc.StatusCode.UNAUTHENTICATED, msg)

    if peer_node_id != declared_node_id:
        msg = (
            f"{ErrorNumbers.FB628.value}: Declared node id `{declared_node_id}` does "
            f"not match the identity `{peer_node_id}` its certificate is registered "
            "under."
        )
        logger.error(msg)
        logger.security_event(
            operation="mtls_identity_mismatch",
            status="failure",
            node_id=peer_node_id,
            declared_node_id=declared_node_id,
            detail=msg,
            **_connection_audit_fields(context),
        )
        await context.abort(grpc.StatusCode.UNAUTHENTICATED, msg)

    return peer_node_id


class ResearcherServicer(researcher_pb2_grpc.ResearcherServiceServicer):
    """RPC Servicer"""

    def __init__(
        self,
        agent_store: AgentStore,
        on_message: Callable,
        identities: Optional[TrustedCertificateBundle] = None,
    ) -> None:
        """Constructor of gRPC researcher servicer

        Args:
            agent_store: The class that stores node agents
            on_message: Callback function to execute once a message received from the nodes
            identities: Registered node certificates, resolving the certificate a
                node presents to the party id it is registered under. None when
                node identity verification is disabled (server-auth only TLS).
        """
        super().__init__()
        self._agent_store = agent_store
        self._on_message = on_message
        self._identities = identities
        # Last audited (certificate serial, source address) per node: a node holds
        # one connection at a time, so a change to either is a new handshake.
        self._peer_identity: Dict[str, tuple] = {}

    async def GetTaskUnary(
        self, request: ProtoBufMessage, context: grpc.aio.ServicerContext
    ) -> None:
        """Gets unary RPC request and return stream of response

        Args:
            request: RPC request
            context: RPC peer context
        """
        task_request = TaskRequest.from_proto(request).get_dict()
        logger.debug(f"Node: {task_request.get('node')} polling for the tasks")

        peer_node_id = await _verify_peer_identity(
            context, task_request["node"], self._identities
        )

        if peer_node_id:
            connection = _connection_audit_fields(context)
            identity = (connection.get("cert_serial"), connection["source_address"])
            if self._peer_identity.get(peer_node_id) != identity:
                self._peer_identity[peer_node_id] = identity
                logger.info(f"Node `{peer_node_id}` authenticated via mutual TLS.")
                logger.security_event(
                    operation="mtls_node_authenticated",
                    status="success",
                    node_id=peer_node_id,
                    **connection,
                )

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
                    logger.warning(
                        f"Message to send is older than {MAX_SEND_DURATION} seconds. Discard message."
                    )
                    logger.debug(
                        "[WIRE][S->N][DROP] node=%s type=%s req=%s retry=%d age_s=%.1f reason=expired",
                        task_request["node"],
                        task.__class__.__name__ if task else None,
                        getattr(task, "request_id", None) if task else None,
                        retry_count,
                        time.time() - first_send_time,
                    )

            task_bytes = Serializer.dumps(task.to_dict())

            logger.debug(
                "[WIRE][S->N][TX] node=%s type=%s req=%s retry=%d age_s=%.1f bytes=%d",
                task_request["node"],
                task.__class__.__name__,
                getattr(task, "request_id", None),
                retry_count,
                time.time() - first_send_time,
                len(task_bytes),
            )

            chunk_range = range(0, len(task_bytes), MAX_MESSAGE_BYTES_LENGTH)
            for start, iter_ in zip(
                chunk_range, range(1, len(chunk_range) + 1), strict=True
            ):
                stop = start + MAX_MESSAGE_BYTES_LENGTH

                try:
                    yield TaskResponse(
                        size=len(chunk_range),
                        iteration=iter_,
                        bytes_=task_bytes[start:stop],
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
                        logger.debug(
                            "[WIRE][S->N][REQUEUE] node=%s type=%s req=%s retry=%d reason=stream_interrupted error=%s",
                            task_request["node"],
                            task.__class__.__name__,
                            getattr(task, "request_id", None),
                            retry_count,
                            str(GeneratorExit),
                        )
                        await node_agent.send_async(
                            message=task,
                            on_reply=None,
                            retry_count=retry_count + 1,
                            first_send_time=first_send_time,
                        )
                    else:
                        logger.warning(
                            f"Message cannot be sent after {MAX_SEND_RETRIES} retries. Discard message."
                        )
                        logger.debug(
                            "[WIRE][S->N][DROP] node=%s type=%s req=%s retry=%d age_s=%.1f reason=expired error=%s",
                            task_request["node"],
                            task.__class__.__name__ if task else None,
                            getattr(task, "request_id", None) if task else None,
                            retry_count,
                            time.time() - first_send_time,
                            str(GeneratorExit),
                        )
                    await node_agent.change_node_status_after_task()
                    # need return here to avoid RuntimeError
                    return

        except asyncio.CancelledError:
            if (
                task is not None
                and retry_count is not None
                and first_send_time is not None
            ):
                # schedule resend if task was pulled from queue
                if retry_count < MAX_SEND_RETRIES:
                    logger.debug(
                        "[WIRE][S->N][REQUEUE] node=%s type=%s req=%s retry=%d reason=stream_interrupted error=%s",
                        task_request["node"],
                        task.__class__.__name__,
                        getattr(task, "request_id", None),
                        retry_count,
                        str(asyncio.CancelledError),
                    )
                    await node_agent.send_async(
                        message=task,
                        on_reply=None,
                        retry_count=retry_count + 1,
                        first_send_time=first_send_time,
                    )
                else:
                    logger.warning(
                        f"Message cannot be sent after {MAX_SEND_RETRIES} retries. Discard message."
                    )
                    logger.debug(
                        "[WIRE][S->N][DROP] node=%s type=%s req=%s retry=%d age_s=%.1f reason=expired error=%s",
                        task_request["node"],
                        task.__class__.__name__ if task else None,
                        getattr(task, "request_id", None) if task else None,
                        retry_count,
                        time.time() - first_send_time,
                        str(asyncio.CancelledError),
                    )
        finally:
            await node_agent.change_node_status_after_task()

    async def ReplyTask(
        self,
        request_iterator: Iterable[ProtoBufMessage],
        context: grpc.aio.ServicerContext,
    ) -> None:
        """Gets stream replies from the nodes

        Args:
            request_iterator: Iterator for streaming
            context: Request service context
        """

        reply = bytes()
        async for answer in request_iterator:
            reply += answer.bytes_
            if answer.size != answer.iteration:
                continue

            # Deserialize message
            message = Serializer.loads(reply)

            await _verify_peer_identity(
                context, message.get("node_id"), self._identities
            )

            logger.debug(
                "[WIRE][N->S][RX] node=%s req=%s type=%s bytes=%d",
                message.get("node_id"),
                message.get("request_id"),
                Message.from_dict(message).__class__.__name__,
                len(reply),
            )

            # Replies are handled by node agent callbacks
            node = await self._agent_store.get(message["node_id"])
            await node.on_reply(message)

            reply = bytes()

        return Empty()

    async def Feedback(
        self, request: ProtoBufMessage, context: grpc.aio.ServicerContext
    ) -> None:
        """Executed for Feedback request received from the nodes

        Args:
            request: Feedback message
            context: Request service context
        """

        # Get the type of Feedback | log or scalar
        one_of = request.WhichOneof("feedback_type")
        feedback = FeedbackMessage.from_proto(request)
        # The node id is carried by the payload; `FeedbackMessage` has none.
        payload = feedback.get_param(one_of)

        await _verify_peer_identity(
            context, getattr(payload, "node_id", None), self._identities
        )

        logger.debug(
            "[WIRE][N->S][RX] node=%s type=Feedback oneof=%s",
            getattr(payload, "node_id", None),
            one_of,
        )

        # Execute on_message assigned by the researcher.requests modules
        self._on_message(payload, MessageType.convert(one_of))

        return Empty()


class _GrpcAsyncServer:
    """GRPC Server class.

    All the methods of this class are awaitable, except the constructor.
    """

    def __init__(
        self,
        host: str,
        port: str,
        config: Config,
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
        self._config = config
        self._server = None
        self._debug = debug
        self._on_message = on_message
        self._loop = None
        self._agent_store: Optional[AgentStore] = None

    def _server_credentials(self) -> grpc.ServerCredentials:
        """Builds the gRPC server credentials.

        Under mutual TLS, node client certificates are required and pinned to the
        registered bundle. The bundle is re-read per handshake, so nodes registered
        after startup are trusted without a restart. Otherwise server-auth only.

        Returns:
            Credentials to serve the researcher endpoint with.

        Raises:
            FedbiomedCertificateError: mutual TLS is enabled but no node
                certificate is registered.
        """
        key_cert_pairs = ((self._ssl.private_key, self._ssl.certificate),)

        if not self._ssl.mtls:
            return grpc.ssl_server_credentials(key_cert_pairs)

        # gRPC refuses to bind the port when the trust bundle is empty, so report
        # the cause instead of an opaque binding failure.
        if not self._ssl.trusted_node_certificates():
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: mutual TLS is enabled but no node "
                "certificate is registered, so the researcher server cannot start. "
                "Register at least one node certificate with `fedbiomed researcher "
                "certificate register`."
            )

        def certificate_configuration():
            return grpc.ssl_server_certificate_configuration(
                key_cert_pairs,
                root_certificates=self._ssl.trusted_node_certificates(),
            )

        return grpc.dynamic_ssl_server_credentials(
            certificate_configuration(),
            certificate_configuration,
            require_client_authentication=True,
        )

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
                (
                    "grpc.http2.min_ping_interval_without_data_ms",
                    0.9 * GRPC_CLIENT_CONN_RETRY_TIMEOUT * 1000,
                ),
                (
                    "grpc.max_connection_idle_ms",
                    (GRPC_CLIENT_TASK_REQUEST_TIMEOUT + 2) * 1000,
                ),
                (
                    "grpc.max_connection_age_ms",
                    (GRPC_CLIENT_TASK_REQUEST_TIMEOUT + 5) * 1000,
                ),
                ("grpc.max_connection_age_grace_ms", 2 * 1000),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.keepalive_permit_without_calls", 1),
                #
                ("grpc.http2.max_ping_strikes", 100),
                #
                # Prevent multiple servers on same port
                ("grpc.so_reuseport", 0),
            ]
        )

        self._loop = asyncio.get_running_loop()
        self._agent_store = AgentStore(
            loop=self._loop,
            on_forward=self._on_forward,
            node_disconnection_timeout=self._config.getint(
                "server", "node_disconnection_timeout"
            ),
        )

        researcher_pb2_grpc.add_ResearcherServiceServicer_to_server(
            ResearcherServicer(
                agent_store=self._agent_store,
                on_message=self._on_message,
                identities=self._ssl.trusted_node_certificates,
            ),
            server=self._server,
        )

        self._server.add_secure_port(
            self._host + ":" + str(self._port), self._server_credentials()
        )
        # self._server.add_insecure_port(self._host + ':' + str(self._port))

        if self._ssl.mtls:
            # Rejections happen inside the TLS handshake, out of reach of this process
            logger.info(
                "Mutual TLS is enabled: nodes whose certificate is not registered "
                "are rejected during the TLS handshake. Run with GRPC_VERBOSITY=INFO "
                "to have gRPC report each rejection."
            )

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
        logger.debug(
            f"Researcher relay forwarding overlay: src_node_id={message.node_id} "
            f"dest_node_id={message.dest_node_id} setup={message.setup} payload_bytes={len(message.overlay)}"
        )
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
            if isinstance(message, OverlayMessage):
                logger.debug(
                    f"Researcher relay drop: dest_node_id={node_id} src_node_id={message.node_id} "
                    f"setup={message.setup} reason=node_not_registered"
                )
            logger.info(f"Node {node_id} is not registered on server. Discard message.")
            return

        if isinstance(message, OverlayMessage):
            logger.debug(
                f"Researcher relay dispatching overlay to node agent: dest_node_id={node_id} "
                f"src_node_id={message.node_id} setup={message.setup} payload_bytes={len(message.overlay)}"
            )

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
            logger.error(
                f"Researcher gRPC server has stopped. Please try to restart: {e}"
            )

    def start(self) -> None:
        """Starts async GrpcServer"""

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # FIXME: This implementation assumes that nodes will be able connect and server complete setup with this delay
        logger.info("Starting researcher service...")

        logger.info(f"Waiting {GRPC_SERVER_SETUP_TIMEOUT}s for nodes to connect...")
        time.sleep(GRPC_SERVER_SETUP_TIMEOUT)

        sleep_ = 0
        while len(self.get_all_nodes()) == 0:
            if sleep_ == 0:
                logger.info(
                    f"No nodes found, server will wait "
                    f"{MAX_GRPC_SERVER_SETUP_TIMEOUT - GRPC_SERVER_SETUP_TIMEOUT} "
                    "more seconds until a node creates connection."
                )

            if sleep_ > MAX_GRPC_SERVER_SETUP_TIMEOUT - GRPC_SERVER_SETUP_TIMEOUT:
                if len(self.get_all_nodes()) == 0:
                    logger.warning(
                        "Server has not received connection from any remote nodes in "
                        f"MAX_GRPC_SERVER_SETUP_TIMEOUT: {MAX_GRPC_SERVER_SETUP_TIMEOUT} "
                        "This may effect the request created right after the server initialization. "
                        "However, server will keep running in the background so you can retry the "
                        "operations for sending requests to remote nodes until one receives."
                    )
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
                f"{ErrorNumbers.FB628.value}: bad argument type for message, expected `Message`, got `{type(message)}`"
            )

        if not self._is_started.is_set():
            raise FedbiomedCommunicationError(
                f"{ErrorNumbers.FB628.value}: Can not send message. "
                "Communication client is not initialized."
            )

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
                f"{ErrorNumbers.FB628.value}: bad argument type for message, expected `Message`, got `{type(message)}`"
            )

        if not self._is_started.is_set():
            raise FedbiomedCommunicationError(
                f"{ErrorNumbers.FB628.value}: Can not broadcast given message. "
                "Communication client is not initialized."
            )

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
                f"{ErrorNumbers.FB628.value}: Error while getting all nodes "
                "connected:  Communication client is not initialized."
            )

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
                f"{ErrorNumbers.FB628.value}: Error while getting node '{node_id}':"
                "Communication client is not initialized."
            )

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
                f"{ErrorNumbers.FB628.value}: Can not check if thread is alive."
                "Communication client is not initialized."
            )

        # TODO: more tests about gRPC server and task status ?
        return (
            False
            if not isinstance(self._thread, threading.Thread)
            else self._thread.is_alive()
        )

    def _run_threadsafe(self, coroutine: Coroutine) -> Any:
        """Runs given coroutine threadsafe

        Args:
            coroutine: Awaitable function to be executed as threadsafe

        Returns:
            Coroutine return value.
        """

        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)

        return future.result()

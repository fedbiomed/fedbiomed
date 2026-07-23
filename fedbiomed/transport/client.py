# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import abc
import asyncio
import socket
import ssl
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Iterable, List, Optional

import grpc
from cryptography import x509

from fedbiomed.common.certificate_manager import certificate_subject_field
from fedbiomed.common.constants import (
    MAX_MESSAGE_BYTES_LENGTH,
    MAX_RETRIEVE_ERROR_RETRIES,
    MAX_SEND_RETRIES,
    ErrorNumbers,
)
from fedbiomed.common.exceptions import FedbiomedCommunicationError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import FeedbackMessage, Message, TaskRequest, TaskResult
from fedbiomed.common.serializer import Serializer
from fedbiomed.transport.protocols.researcher_pb2_grpc import ResearcherServiceStub

# UNAVAILABLE error-detail markers of a TLS/pinning failure (gRPC reports it
# with the same status as an unreachable server).
_TLS_HANDSHAKE_ERROR_MARKERS = ("handshake", "certificate", "ssl", "tls")

# UNAVAILABLE error-detail markers of a connection closed by the peer, the only
# trace of a server rejecting the client certificate mid-handshake.
_CONNECTION_CLOSED_ERROR_MARKERS = ("socket closed", "connection reset", "broken pipe")


@dataclass
class NodeClientIdentity:
    """The node's own mutual-TLS client identity, presented to the researcher.

    Owned by the node, not the researcher. Only populated when mutual TLS is
    enabled.
    """

    # `private_key` is secret and kept out of repr to avoid leaking into logs.
    private_key: Optional[bytes] = field(default=None, repr=False)
    certificate_chain: Optional[bytes] = None


@dataclass
class ResearcherCredentials:
    """Connection details and pinned server certificate of a researcher.

    Identifies the researcher endpoint (`host`/`port`) and pins its public
    server `certificate`. Under mutual TLS the node additionally presents its
    own client identity, carried separately in `node_identity`.
    """

    port: str
    host: str
    # Researcher server certificate to pin (public).
    certificate: Optional[bytes] = None
    mtls: bool = False
    # Node's own client identity, presented to the researcher under mutual TLS.
    node_identity: Optional[NodeClientIdentity] = None
    # Whether the researcher was observed to demand certificates. None until probed.
    client_auth_enforced: Optional[bool] = None


class ClientStatus(Enum):
    DISCONNECTED = 0
    CONNECTED = 1
    FAILED = 2


class _StubType(Enum):
    NO_STUB = 0  # never matcher stub type
    ANY_STUB = 1  # always matches stub type
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
    for family, socktype, protocol, _, address in address_info:
        # Use a context manager so the socket is always closed, even when
        # connect() raises (previously the socket leaked on failure).
        with socket.socket(family, socktype, protocol) as s:
            # Need this timeout for the case where the server does not answer
            # If not present, socket timeout increases and this function takes more
            # than GRPC_CLIENT_CONN_RETRY_TIMEOUT to execute
            s.settimeout(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
            try:
                s.connect(address)
            except socket.error:
                return False
            else:
                return True


def _is_tls_handshake_error(exp: grpc.aio.AioRpcError) -> bool:
    """Whether an UNAVAILABLE RPC error is really a TLS/pinning failure."""
    detail = f"{exp.details()} {exp.debug_error_string()}".lower()
    return any(m in detail for m in _TLS_HANDSHAKE_ERROR_MARKERS)


def _is_connection_closed_error(exp: grpc.aio.AioRpcError) -> bool:
    """Whether an UNAVAILABLE RPC error is a connection closed by the peer."""
    detail = f"{exp.details()} {exp.debug_error_string()}".lower()
    return any(m in detail for m in _CONNECTION_CLOSED_ERROR_MARKERS)


def _researcher_requires_client_auth(host: str, port: str) -> bool:
    """Whether the researcher's TLS server demands a client certificate.

    gRPC hides from the client whether its certificate was requested, so this
    probes with a raw TLS handshake presenting none, and reads the server's
    first reply: a researcher accepting an anonymous client answers with its
    HTTP/2 SETTINGS frame, one enforcing mutual TLS closes or aborts instead.

    Completing the handshake is not evidence of acceptance. Under TLS 1.3 the
    client's handshake completes before the server validates the client
    certificate, so the rejection only shows up on the first read. Reading the
    reply is what makes this work on both TLS 1.2 and TLS 1.3.

    Args:
        host: The host/ip of the researcher server.
        port: Port number of the researcher server.

    Returns:
        True if a client certificate is required, False if an anonymous client
        is accepted.
    """
    context = ssl.create_default_context()
    # Testing the client-auth requirement only, not the server certificate.
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    # gRPC serves HTTP/2 only over ALPN; without it the server drops any
    # connection, which is indistinguishable from a client-auth rejection.
    context.set_alpn_protocols(["h2"])
    try:
        with socket.create_connection(
            (host, int(port)), timeout=GRPC_CLIENT_CONN_RETRY_TIMEOUT
        ) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                ssock.settimeout(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
                # Empty read == closed without replying == identity demanded.
                return not ssock.recv(1)
    except (ssl.SSLError, OSError):
        # A server enforcing client auth rejects the anonymous connection with a
        # TLS alert (SSLError) or by resetting it (OSError). Any transient socket
        # failure is also treated conservatively as "required", so this
        # diagnostic probe never warns spuriously nor crashes the connect loop.
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
            _StubType.SENDER_FEEDBACK_STUB,
        ]
        for st in self._stub_types:
            self._channels[st]: grpc.aio.Channel = None
            self._stubs[st]: ResearcherServiceStub = None

        # lock for accessing channels and stubs
        self._channels_stubs_lock = asyncio.Lock()

    @property
    def mtls(self) -> bool:
        """Whether the node connects to the researcher with mutual TLS."""
        return self._researcher.mtls

    @property
    def endpoint(self) -> str:
        """Researcher server endpoint as `host:port`."""
        return f"{self._researcher.host}:{self._researcher.port}"

    @property
    def host(self) -> str:
        """Researcher server host."""
        return self._researcher.host

    @property
    def port(self) -> str:
        """Researcher server port."""
        return self._researcher.port

    @property
    def client_auth_enforced(self) -> Optional[bool]:
        """Whether the researcher was observed to demand client certificates."""
        return self._researcher.client_auth_enforced

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
        """Connects gRPC server and instantiates stubs.

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
        if self._researcher.mtls:
            node_identity = self._researcher.node_identity
            credentials = grpc.ssl_channel_credentials(
                root_certificates=self._researcher.certificate,
                private_key=node_identity.private_key,
                certificate_chain=node_identity.certificate_chain,
            )
            target_name_override = certificate_subject_field(
                self._researcher.certificate, x509.oid.NameOID.COMMON_NAME
            )
        else:
            credentials = grpc.ssl_channel_credentials(self._researcher.certificate)
            target_name_override = None

        return self._create_channel(
            port=self._researcher.port,
            host=self._researcher.host,
            certificate=credentials,
            target_name_override=target_name_override,
        )

    @staticmethod
    def _create_channel(
        port: str,
        host: str,
        certificate: Optional[grpc.ChannelCredentials] = None,
        target_name_override: Optional[str] = None,
    ) -> grpc.Channel:
        """Create gRPC channel

        Args:
            ip: IP address of the channel
            port: TCP port of the channel
            certificate: channel credentials for secure channel, or None for insecure channel
            target_name_override: expected server name to verify against the pinned
                certificate, used when the connect host differs from the cert CN

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

        if target_name_override is not None:
            channel_options.append(
                ("grpc.ssl_target_name_override", target_name_override)
            )

        if certificate is None:
            channel = grpc.aio.insecure_channel(
                f"{host}:{port}", options=channel_options
            )
        else:
            channel = grpc.aio.secure_channel(
                f"{host}:{port}", certificate, options=channel_options
            )

        return channel


class GrpcClient:
    """An agent of remote researcher gRPC server."""

    def __init__(
        self, node_id: str, researcher: ResearcherCredentials, update_id_map: Awaitable
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
            on_status_change=self._on_status_change,
            update_id=self._update_id,
        )

        self._sender = Sender(
            channels=self._channels, on_status_change=self._on_status_change
        )

        # TODO: use `self._status` for finer gRPC agent handling.
        # Currently, the (tentative) status is maintained but not used
        self._status = ClientStatus.DISCONNECTED
        # lock for accessing self._status
        self._status_lock = asyncio.Lock()

        self._update_id_map = update_id_map
        self._tasks = []
        # Report the repeating connect-loop mTLS mismatch once.
        self._auth_mismatch_logged = False

    @property
    def tasks(self) -> List[asyncio.Task]:
        """Returns running asyncio task(s) owned by this client."""

        return self._tasks

    def start(self, on_task) -> asyncio.Task:
        """Start researcher gRPC agent.

        Starts long-lived tasks, one waiting for server requests, one waiting on the async queue
        for the replies from the node that are going to be sent back to researcher.

        Args:
            on_task: Callback function to execute once a payload received from researcher.

        Returns:
            The main task object of the agent
        """

        async def run():
            """Connects and dispatches the tasks"""

            # First connects to channel
            await self._connect()

            # Launch listeners
            await asyncio.gather(
                self._task_listener.listen(on_task), self._sender.listen()
            )

        # Keep a stable reference so controller health checks can inspect client tasks.
        task = asyncio.create_task(run())
        self._tasks = [task]
        return task

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
                if not self._researcher.mtls:
                    if _researcher_requires_client_auth(
                        self._researcher.host, self._researcher.port
                    ):
                        # A researcher enforcing mutual TLS would reject this node
                        self._log_auth_mismatch_once(
                            f"{ErrorNumbers.FB628.value}: The researcher requires "
                            "mutual-TLS client authentication but mutual-TLS is "
                            "disabled on this node. Enable it in the node `[mtls]` "
                            "configuration, register the researcher certificate and "
                            "ask the researcher to register this node's certificate."
                        )
                        await asyncio.sleep(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
                        continue
                    # Gets server certificate before creating the channel
                    # This implementation assumes that the provided IP and PORT trusted
                    # == OK for honest but curious researcher and nodes (parties in the
                    # network instance) but subject to attack by malicious MITM at each
                    # connection to server.
                    # Skipped under mutual TLS, where the cert is pinned, not fetched.
                    self._researcher.certificate = bytes(
                        ssl.get_server_certificate(
                            (self._researcher.host, self._researcher.port)
                        ),
                        "utf-8",
                    )
                    logger.info("Retrieved server certificate, connecting to server.")
                else:
                    self._researcher.client_auth_enforced = (
                        _researcher_requires_client_auth(
                            self._researcher.host, self._researcher.port
                        )
                    )
                    if not self._researcher.client_auth_enforced:
                        msg = (
                            "This node is configured for mutual-TLS but the "
                            "researcher does not require client certificates: "
                            "node identity will NOT be verified. Connecting "
                            "anyway with the researcher certificate registered."
                        )
                        logger.warning(msg)
                        logger.security_event(
                            operation="mtls_not_enforced_by_researcher",
                            status="failure",
                            detail=msg,
                        )

                if self._id is None:
                    # auto-detect researcher_id from the peer certificate O= field
                    self._id = certificate_subject_field(
                        self._researcher.certificate,
                        x509.oid.NameOID.ORGANIZATION_NAME,
                    )

                # Connect to channels and create stubs
                await self._channels.connect()
                logger.info(
                    "Channel created to researcher server at "
                    f"{self._researcher.host}:{self._researcher.port}",
                    extra={"is_security": True},
                )

                break
            else:
                logger.debug(
                    "Researcher server is not available, will retry connecting in "
                    f"{GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds"
                )
                await asyncio.sleep(
                    max(
                        0,
                        GRPC_CLIENT_CONN_RETRY_TIMEOUT
                        - time.perf_counter()
                        + time_before,
                    )
                )

    def _log_auth_mismatch_once(self, message: str) -> None:
        """Logs an mTLS configuration mismatch at error level with a security
        audit event, once; the connect loop repeats it at debug only."""
        if self._auth_mismatch_logged:
            logger.debug(message)
            return
        self._auth_mismatch_logged = True
        logger.error(message)
        logger.security_event(
            operation="mtls_configuration_mismatch",
            status="failure",
            detail=message,
        )

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
            msg = (
                f"{ErrorNumbers.FB628.value}: Suspected malicious researcher activity ! "
                f"Researcher ID changed for {self._researcher.host}:{self._researcher.port} from "
                f"`{self._id}` to `{id_}`"
            )
            logger.error(msg)
            raise FedbiomedCommunicationError(msg)

        self._id = id_
        await self._update_id_map(
            f"{self._researcher.host}:{self._researcher.port}", id_
        )


class Listener:
    """Abstract generic listener method for a node's communications."""

    def __init__(self, channels: Channels) -> None:
        """Constructs task listener channels

        Args:
            channels: Keeps channels and stubs.
        """
        self._channels = channels
        self._retry_on_error = False
        # Report the repeating TLS failure once until the connection recovers.
        self._tls_failure_logged = False

    @abc.abstractmethod
    async def _handle_after_process(
        self,
        status: ClientStatus,
        retry: bool = False,
        reconnect: bool = False,
        post_noretry_function: Optional[Callable] = None,
        *args,
    ):
        """Actions after each call to the researcher, successful or not

        Args:
            status: new gRPC client status to set
            retry: want to retry same action, if applicable
            reconnect: want to redo connection to server, if applicable
            post_noretry_function: optional final function to execute, if applicable
            args: arguments for `post_noretry_function`
        """

    @abc.abstractmethod
    def _message_deadline_exceeded(self):
        """Logger message to issue when deadline is exceeded in call to researcher"""

    @abc.abstractmethod
    async def _call_researcher(self, callback: Optional[Callable] = None) -> None:
        """Requests tasks from Researcher

        Args:
            callback: Callback to execute once a task is submitted
        """

    def _server_reachable(self) -> bool:
        """Whether the researcher endpoint accepts TCP connections; resolution
        failures count as unreachable."""
        try:
            return is_server_alive(self._channels.host, self._channels.port)
        except OSError:
            return False

    def _log_tls_failure_once(self, message: str) -> None:
        """Logs a TLS failure at error level with a security audit event, once
        per disconnection; the retry loop repeats it at debug only."""
        if self._tls_failure_logged:
            logger.debug(message)
            return
        self._tls_failure_logged = True
        logger.error(message)
        logger.security_event(
            operation="mtls_handshake_failure",
            status="failure",
            detail=message,
        )

    async def _post_handle_raise(self, exp: BaseException):
        """Raise a transformed exception from a base exception.

        To be called as final function after handling process in a listener task

        Args:
            exp: Base exception to use
        """
        raise FedbiomedCommunicationError(
            f"{ErrorNumbers.FB628.value}: {self.__class__.__name__} has stopped due to unknown reason: "
            f"{type(exp).__name__} : {exp}"
        ) from exp

    def listen(
        self, callback: Optional[Callable] = None
    ) -> Awaitable[Optional[Callable]]:
        """Listens for tasks from given channels

        Args:
            callback: Callback function to execute once a task is processed

        Returns:
            Asyncio task to run task listener
        """
        return asyncio.create_task(self._listen(callback))

    async def _listen(self, callback: Optional[Callable] = None) -> None:
        """ "Starts the loop for the listening task

        Args:
            callback: Callback function to execute once a task is processed

        Raises:
            FedbiomedCommunicationError: communication error with researcher
        """

        while True:
            try:
                await self._call_researcher(callback)
            except grpc.aio.AioRpcError as exp:
                match exp.code():
                    case grpc.StatusCode.DEADLINE_EXCEEDED:
                        self._message_deadline_exceeded()
                        await self._handle_after_process(ClientStatus.DISCONNECTED)
                    case grpc.StatusCode.UNAVAILABLE:
                        await self._on_status_change(ClientStatus.DISCONNECTED)
                        if self._channels.mtls and _is_tls_handshake_error(exp):
                            self._log_tls_failure_once(
                                f"{ErrorNumbers.FB628.value}: Mutual-TLS handshake with "
                                f"researcher failed in {self.__class__.__name__}: "
                                f"{exp.details()}. Check pinned/registered certificates "
                                "and possible MITM. Retrying silently."
                            )
                        elif (
                            self._channels.mtls
                            and _is_connection_closed_error(exp)
                            and self._server_reachable()
                        ):
                            # Reachable but closing during the handshake: the
                            # researcher does not trust this node's certificate.
                            self._log_tls_failure_once(
                                f"{ErrorNumbers.FB628.value}: The researcher at "
                                f"{self._channels.endpoint} is reachable but closes "
                                "the connection during the TLS handshake: it likely "
                                "does not recognize this node's certificate. Ask "
                                "the researcher to register it."
                            )
                        elif (
                            not self._channels.mtls
                            and (
                                _is_connection_closed_error(exp)
                                or _is_tls_handshake_error(exp)
                            )
                            and self._server_reachable()
                            and _researcher_requires_client_auth(
                                self._channels.host, self._channels.port
                            )
                        ):
                            self._log_tls_failure_once(
                                f"{ErrorNumbers.FB628.value}: The researcher requires "
                                "mutual-TLS client authentication but mutual-TLS is "
                                "disabled on this node. Enable it in the node `[mtls]` "
                                "configuration, register the researcher certificate and "
                                "ask the researcher to register this node's certificate."
                            )
                        else:
                            logger.debug(
                                f"Researcher server is not available to {self.__class__.__name__}, will retry connect in "
                                f"{GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds"
                            )
                        await self._handle_after_process(
                            ClientStatus.DISCONNECTED,
                            retry=self._retry_on_error,
                            reconnect=True,
                        )

                    case grpc.StatusCode.UNAUTHENTICATED:
                        # Identity rejected by researcher; static config, retry
                        # cannot help: stop.
                        await self._on_status_change(ClientStatus.FAILED)
                        msg = (
                            f"{ErrorNumbers.FB628.value}: Researcher rejected this "
                            f"node's identity in {self.__class__.__name__}: "
                            f"{exp.details()}. Declared node id does not match its "
                            "certificate, or the certificate is not registered."
                        )
                        logger.error(msg)
                        logger.security_event(
                            operation="mtls_identity_rejected",
                            status="failure",
                            detail=msg,
                        )
                        raise FedbiomedCommunicationError(msg) from exp

                    case grpc.StatusCode.UNKNOWN | _:
                        logger.error(
                            "Unexpected error raised by researcher gRPC server in "
                            f"{self.__class__.__name__}: {exp}. "
                            f"Will retry connect in {GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds "
                            f"to the channel {self._channels._channels} "
                            f"with stubs {self._channels._stubs}",
                            extra={"is_security": True},
                        )
                        await self._handle_after_process(
                            ClientStatus.FAILED,
                            retry=self._retry_on_error,
                            reconnect=True,
                        )

            except (Exception, GeneratorExit) as exp:
                logger.error(
                    f"Unexpected error raised by node gRPC client in {self.__class__.__name__}: "
                    f"{type(exp).__name__} : {exp} "
                    f"to the channel {self._channels._channels} "
                    f"with stubs {self._channels._stubs}",
                    extra={"is_security": True},
                    exc_info=True,
                )
                await self._handle_after_process(
                    ClientStatus.FAILED, True, False, self._post_handle_raise, exp
                )
            else:
                self._tls_failure_logged = False
                await self._handle_after_process(ClientStatus.CONNECTED)


class TaskListener(Listener):
    """Listener for the task assigned by the researcher component"""

    def __init__(
        self,
        channels: Channels,
        node_id: str,
        on_status_change: Awaitable,
        update_id: Awaitable,
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
        self._retry_count = 0
        self._communication_established = False

    async def _handle_after_process(
        self,
        status: ClientStatus,
        retry: bool = False,
        reconnect: bool = False,
        post_noretry_function: Optional[Callable] = None,
        *args,
    ):
        """Actions after each tentative to retrieve a task, successful or not

        Args:
            status: new gRPC client status to set
            retry: if True (and MAX_RETRIEVE_ERROR_RETRIES is not exceeded) then retry to get a task
            reconnect: if True and `retry` is False, then redo connection to server
            post_noretry_function: optional final function to execute if not retrying to get a task.
                If None, no final function is executed
            args: arguments for `post_noretry_function`
        """
        await self._on_status_change(status)

        if retry or reconnect:
            self._communication_established = False

        if retry and self._retry_count < MAX_RETRIEVE_ERROR_RETRIES:
            await asyncio.sleep(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
            await self._channels.connect(_StubType.LISTENER_TASK_STUB)
            self._retry_count += 1
        else:
            if reconnect:
                await asyncio.sleep(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
                await self._channels.connect(_StubType.LISTENER_TASK_STUB)
            self._retry_count = 0

            if post_noretry_function:
                # works only if args are provided
                await post_noretry_function(*args)

    def _announce_communication_established(self):
        """Logs, once per connection, that the node reached the researcher.

        Called from the first poll cycle that completes without a connection
        error (a received task, or a deadline with no task queued), which proves
        the TLS handshake succeeded and, under mutual TLS, that the researcher
        accepted this node's identity.
        """
        if self._communication_established:
            return
        self._communication_established = True

        if self._channels.mtls and self._channels.client_auth_enforced is not False:
            logger.info(
                "Mutual-TLS communication established with researcher at "
                f"{self._channels.endpoint}; node identity verified by the researcher."
            )
        elif self._channels.mtls:
            # The connect probe saw the researcher accept anonymous clients.
            logger.info(
                "Communication established with researcher at "
                f"{self._channels.endpoint} over TLS with pinned researcher "
                "certificate; node identity NOT verified by the researcher "
                "(mutual TLS not enforced)."
            )
        else:
            logger.info(
                "Communication established with researcher at "
                f"{self._channels.endpoint} over server-authenticated TLS "
                "(node identity not verified)."
            )

    def _message_deadline_exceeded(self):
        """Task listener issues debug message when researcher does not submit task before deadline"""
        self._announce_communication_established()
        logger.debug(
            "Task polling timed out: node=%s timeout_s=%s; sending a new task request",
            self._node_id,
            GRPC_CLIENT_TASK_REQUEST_TIMEOUT,
        )

    async def _call_researcher(self, callback: Optional[Callable] = None) -> None:
        """Requests tasks from Researcher

        Args:
            callback: Callback to execute once a task is arrived
        """
        logger.debug(
            "Polling researcher for task: node=%s retry=%d timeout_s=%s",
            self._node_id,
            self._retry_count,
            GRPC_CLIENT_TASK_REQUEST_TIMEOUT,
        )
        # TODO: improve status management. At this point it is not sure we are CONNECTED to server
        # but setting later will leave the client DISCONNECTED when waiting for initial task
        await self._on_status_change(ClientStatus.CONNECTED)

        request_stub = await self._channels.stub(_StubType.LISTENER_TASK_STUB)
        iterator = request_stub.GetTaskUnary(
            TaskRequest(node=f"{self._node_id}").to_proto(),
            timeout=GRPC_CLIENT_TASK_REQUEST_TIMEOUT,
        )
        # Prepare reply
        reply = bytes()
        async for answer in iterator:
            reply += answer.bytes_
            if answer.size != answer.iteration:
                continue
            else:
                # Execute callback
                task = Serializer.loads(reply)

                logger.debug(
                    "[WIRE][S->N][RX] req=%s node=%s type=%s  bytes=%d retry=%d",
                    task.get("request_id", None),
                    self._node_id,
                    Message.from_dict(task).__class__.__name__,
                    len(reply),
                    self._retry_count,
                )

                self._announce_communication_established()

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
        self._retry_item = None
        self._stub_type = _StubType.NO_STUB
        self._retry_on_error = True

    async def _handle_after_process(
        self,
        status: ClientStatus,
        retry: bool = False,
        reconnect: bool = False,
        post_noretry_function: Optional[Callable] = None,
        *args,
    ):
        """Actions after each tentative to send a message, successful or not

        Args:
            status: new gRPC client status to set
            retry: if True (and MAX_SEND_RETRIES is not exceeded) then re-send message
            reconnect: unused
            post_noretry_function: optional final function to execute if message is not re-sent.
                If None, no final function is executed
            args: arguments for `post_noretry_function`
        """
        await self._on_status_change(status)

        if retry and self._retry_count < MAX_SEND_RETRIES:
            if isinstance(self._retry_item, dict):
                msg = self._retry_item["message"]
                logger.debug(
                    "Retrying sender message req=%s type=%s stub=%s retry=%d/%d",
                    getattr(msg, "request_id", None),
                    msg.__class__.__name__,
                    self._stub_type.name
                    if self._stub_type != _StubType.NO_STUB
                    else None,
                    self._retry_count + 1,
                    MAX_SEND_RETRIES,
                )
            await asyncio.sleep(GRPC_CLIENT_CONN_RETRY_TIMEOUT)
            await self._channels.connect(self._stub_type)
            self._retry_count += 1
        else:
            if self._retry_count >= MAX_SEND_RETRIES:
                logger.warning(
                    f"Message can not be sent to researcher after {MAX_SEND_RETRIES} retries. Discard message."
                )
            # Only cleanup if not already done (defensive against double task_done)
            self._queue.task_done()
            self._retry_count = 0
            self._retry_item = None
            self._stub_type = _StubType.NO_STUB

            if post_noretry_function:
                # works only if args are provided
                await post_noretry_function(*args)

    def _message_deadline_exceeded(self):
        """Sender issues warning when researcher does not complete request before deadline"""
        logger.warning(
            "Researcher not answering after timeout, looks like server failure or disconnect. "
            "Discard message."
        )

    async def _call_researcher(self, callback: Optional[Callable] = None) -> None:
        """Gets task result from the queue.

        Args:
            callback: Callback to execute once a task is received
        """
        if self._retry_count == 0:
            # only pick a new message if not retrying to send
            self._retry_item = await self._queue.get()
        item = self._retry_item

        self._stub_type = item["stub"]
        if self._stub_type == _StubType.SENDER_FEEDBACK_STUB:
            feedback_stub = await self._channels.stub(_StubType.SENDER_FEEDBACK_STUB)
            stub_function = feedback_stub.Feedback
        elif self._stub_type == _StubType.SENDER_TASK_STUB:
            task_stub = await self._channels.stub(_StubType.SENDER_TASK_STUB)
            stub_function = task_stub.ReplyTask
        else:
            raise FedbiomedCommunicationError(
                f"Unknown type of stub in gRPC Sender listener {item['stub']}"
            )

        logger.debug(
            "[WIRE][N->S][TX] req=%s stub=%s node=%s type=%s retry=%d",
            getattr(item["message"], "request_id", None),
            self._stub_type.name,
            getattr(item["message"], "node_id", None),
            item["message"].__class__.__name__,
            self._retry_count,
        )

        # If it is a Unary-Unary RPC call
        if isinstance(stub_function, grpc.aio.UnaryUnaryMultiCallable):
            await stub_function(item["message"].to_proto())
            # Clear retry state immediately after successful send to prevent duplicate sends

        elif isinstance(stub_function, grpc.aio.StreamUnaryMultiCallable):
            stream_call = stub_function()

            for reply in self._stream_reply(item["message"]):
                await stream_call.write(reply)

            await stream_call.done_writing()
            # Clear retry state immediately after successful send to prevent duplicate sends

            if isinstance(callback, Callable):
                # we could check the callback prototype
                callback(item["message"])

        else:
            raise FedbiomedCommunicationError(
                f"Unknown type of stub built from gRPC Sender listener {item['stub']}"
            )

    def _stream_reply(self, message: Message) -> Iterable:
        """Streams task result back researcher component.

        Args:
            message: Message to stream

        Returns:
            A stream of researcher reply chunks
        """

        reply = Serializer.dumps(message.to_dict())
        chunk_range = range(0, len(reply), MAX_MESSAGE_BYTES_LENGTH)
        for start, iter_ in zip(
            chunk_range, range(1, len(chunk_range) + 1), strict=True
        ):
            stop = start + MAX_MESSAGE_BYTES_LENGTH
            yield TaskResult(
                size=len(chunk_range), iteration=iter_, bytes_=reply[start:stop]
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
                await self._queue.put(
                    {"stub": _StubType.SENDER_FEEDBACK_STUB, "message": message}
                )

            case _:
                await self._queue.put(
                    {"stub": _StubType.SENDER_TASK_STUB, "message": message}
                )

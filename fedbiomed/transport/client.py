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

from fedbiomed.common.certificate_manager import (
    certificate_audit_fields,
    certificate_component_id,
    certificate_san_names,
    is_loopback_name,
)
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

# UNAVAILABLE detail marker of a failed gRPC name check; matches none of the terms above
_NAME_CHECK_ERROR_MARKER = "hostname verification"

# UNAVAILABLE error-detail markers of a connection closed by the peer, the only
# trace of a server rejecting the client certificate mid-handshake.
_CONNECTION_CLOSED_ERROR_MARKERS = ("socket closed", "connection reset", "broken pipe")


class _ResearcherAuthenticationPending(Exception):
    """The researcher has not authenticated this node, in a state only it can leave.

    Signals the listen loop to close the channel and try again rather than stop.
    """


@dataclass
class NodeClientIdentity:
    """The node's own client identity, presented to the researcher.

    Owned by the node, not the researcher. Only populated when mutual
    authentication is enabled.
    """

    # `private_key` is secret and kept out of repr to avoid leaking into logs.
    private_key: Optional[bytes] = field(default=None, repr=False)
    certificate_chain: Optional[bytes] = None


@dataclass
class ResearcherCredentials:
    """Connection details and pinned server certificate of a researcher.

    Identifies the researcher endpoint (`host`/`port`) and pins its public
    server `certificate`. Under mutual authentication the node additionally
    presents its own client identity, carried separately in `node_identity`.
    """

    port: str
    host: str
    # Researcher server certificate to pin (public).
    certificate: Optional[bytes] = None
    mtls: bool = False
    # Node's own client identity, presented under mutual authentication.
    node_identity: Optional[NodeClientIdentity] = None


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

# gRPC initial-metadata key carrying the component id the researcher resolved from the
# certificate the node presented. Only a researcher that required, received and matched
# that certificate can name the node, so the header proves mutual authentication is in
# force rather than asserting it. Absent when the researcher does not enforce it.
MTLS_PEER_ID_HEADER = "fbm-peer-component-id"


def is_server_alive(host: str, port: str):
    """Checks if the server is alive

    Args:
        host: The host/ip of researcher/server component
        port: Port number of researcher/server component
    """

    port = int(port)
    address_info = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    for family, socktype, protocol, _, address in address_info:
        # Context manager so the socket is closed even when connect() raises
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


def _is_name_check_failure(exp: grpc.aio.AioRpcError) -> bool:
    """Whether an UNAVAILABLE RPC error is gRPC refusing the peer's name."""
    detail = f"{exp.details()} {exp.debug_error_string()}".lower()
    return _NAME_CHECK_ERROR_MARKER in detail


def _is_connection_recycled_error(exp: grpc.aio.AioRpcError) -> bool:
    """Whether an UNAVAILABLE RPC error is the server retiring a connection on the
    maximum age it grants, which the client replaces rather than fails."""
    detail = f"{exp.details()} {exp.debug_error_string()}".lower()
    return "max connection age" in detail


def _researcher_requires_client_auth(host: str, port: str) -> Optional[bool]:
    """Whether the researcher's TLS server demands a client certificate.

    Diagnoses a node that has mutual authentication disabled while the researcher
    requires it: the handshake never completes, so nothing the researcher sends can
    tell the node why. A node that does have it enabled learns the researcher's
    stance from `MTLS_PEER_ID_HEADER` on its task request instead, which needs
    no probing.

    Probes with a raw TLS handshake presenting no certificate, and reads the
    server's first reply: a researcher accepting an anonymous client answers with
    its HTTP/2 SETTINGS frame, one enforcing mutual authentication closes or aborts
    instead.

    Completing the handshake is not evidence of acceptance. Under TLS 1.3 the
    client's handshake completes before the server validates the client
    certificate, so the rejection only shows up on the first read. Reading the
    reply is what makes this work on both TLS 1.2 and TLS 1.3.

    Blocking: run it off the event loop.

    Args:
        host: The host/ip of the researcher server.
        port: Port number of the researcher server.

    Returns:
        True if a client certificate is demanded, False if an anonymous client is
        accepted, None if the exchange ended without saying either way. Callers act
        on a definite answer only, so an unreachable or slow server cannot be
        mistaken for a configuration mismatch.
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
    except TimeoutError:
        # Neither answered nor refused: the server may just be slow.
        return None
    except (ConnectionResetError, BrokenPipeError):
        # Aborted once connected: how a server rejects an anonymous client.
        return True
    except ssl.SSLError:
        # A TLS alert in reply to a handshake presenting no certificate.
        return True
    except OSError:
        # Never got far enough to learn anything (refused, unreachable, DNS).
        return None


def _name_verified_under(host: str, san_names: List[str]) -> Optional[str]:
    """The certificate name a connection to `host` is verified under, None for the
    host itself. TLS matches a name as written, so one loopback form does not
    verify another and a name never verifies an address."""
    return None if host in san_names else san_names[0]


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
        """Whether the node connects to the researcher with mutual authentication."""
        return self._researcher.mtls

    @property
    def certificate(self) -> Optional[bytes]:
        """Researcher server certificate the channel verifies the peer against."""
        return self._researcher.certificate

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
            san_names = certificate_san_names(self._researcher.certificate)

            # gRPC would verify on the Common Name; a fetched certificate passes no
            # registry. Refused before the close loop, so it tears down no channel.
            if not san_names:
                msg = (
                    f"{ErrorNumbers.FB628.value}: The researcher certificate at "
                    f"{self.endpoint} states no host: its Subject Alternative Name "
                    "carries no host name and no address, so nothing in it says which "
                    "server it is valid for. Request the researcher to reissue its "
                    "certificate for the hosts nodes reach it at."
                )
                logger.error(msg)
                raise FedbiomedCommunicationError(msg)

            # Closes if channels are open
            for st, channel in self._channels.items():
                if channel and (stub_type == _StubType.ANY_STUB or stub_type == st):
                    await channel.close()

            # Creates channels
            for st in self._channels.keys():
                if stub_type == _StubType.ANY_STUB or stub_type == st:
                    self._channels[st] = self._create(san_names)
                    self._stubs[st] = ResearcherServiceStub(channel=self._channels[st])

    def _create(self, san_names: List[str]):
        """Creates new channel

        Args:
            san_names: hosts the researcher certificate states, read once per connect
        """
        if self._researcher.mtls:
            node_identity = self._researcher.node_identity
            credentials = grpc.ssl_channel_credentials(
                root_certificates=self._researcher.certificate,
                private_key=node_identity.private_key,
                certificate_chain=node_identity.certificate_chain,
            )
        else:
            credentials = grpc.ssl_channel_credentials(self._researcher.certificate)

        return self._create_channel(
            port=self._researcher.port,
            host=self._researcher.host,
            certificate=credentials,
            # Verify the address dialled where the certificate names it; one issued
            # before the deployment address was known is verified against a name it
            # does carry.
            target_name_override=_name_verified_under(self._researcher.host, san_names),
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
                certificate, used when the certificate does not name the connect host.
                None verifies the connect host itself.

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
                    # Gets server certificate before creating the channel
                    # This implementation assumes that the provided IP and PORT trusted
                    # == OK for honest but curious researcher and nodes (parties in the
                    # network instance) but subject to attack by malicious MITM at each
                    # connection to server.
                    # Skipped under mutual authentication, where the cert is pinned,
                    # not fetched.
                    self._researcher.certificate = bytes(
                        ssl.get_server_certificate(
                            (self._researcher.host, self._researcher.port)
                        ),
                        "utf-8",
                    )
                    msg = "Retrieved server certificate, connecting to server."
                    logger.info(msg)
                    logger.security_event(
                        operation="server_certificate_auto_trusted",
                        status="success",
                        host=self._researcher.host,
                        port=self._researcher.port,
                        detail=msg,
                    )

                # Reported here, run once, not in `_create`: per stub and per reconnect
                san_names = certificate_san_names(self._researcher.certificate)
                verified_under = (
                    _name_verified_under(self._researcher.host, san_names)
                    if san_names
                    else None
                )
                # Silent when that name is a loopback form of the machine dialled
                if verified_under and not (
                    is_loopback_name(self._researcher.host)
                    and is_loopback_name(verified_under)
                ):
                    logger.warning(
                        "The researcher certificate does not name the host "
                        f"`{self._researcher.host}` dialled from `[researcher] ip`; "
                        "it is issued for "
                        f"{', '.join(f'`{name}`' for name in san_names)}. This does "
                        "not prevent the connection."
                    )

                if self._id is None:
                    # auto-detect researcher_id from the peer certificate; a
                    # certificate Fed-BioMed did not issue carries no component id
                    self._id = certificate_component_id(self._researcher.certificate)

                # Connect to channels and create stubs
                await self._channels.connect()
                logger.info(
                    "Connecting to researcher server at "
                    f"{self._researcher.host}:{self._researcher.port}"
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
            FedbiomedCommunicationError: the researcher id changed mid-connection
        """
        if self._id is not None and self._id != id_:
            msg = (
                f"{ErrorNumbers.FB628.value}: Researcher ID changed for "
                f"{self._researcher.host}:{self._researcher.port} from "
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
        # Report the repeating failure once until the connection recovers.
        self._connection_failure_logged = False
        # Whether the last connection was retired on its maximum age.
        self._connection_recycled = False

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

    async def _server_reachable(self) -> bool:
        """Whether the researcher endpoint accepts TCP connections; resolution
        failures count as unreachable. Connecting blocks, so it runs off the event
        loop: every client of this node shares it."""

        def probe() -> bool:
            try:
                return is_server_alive(self._channels.host, self._channels.port)
            except OSError:
                return False

        return await asyncio.to_thread(probe)

    def _log_connection_failure_once(
        self, message: str, operation: str = "mtls_handshake_failure"
    ) -> None:
        """Logs a connection failure the node retries, at warning level with a
        security audit event, once per disconnection; the retry loop repeats it at
        debug only."""
        if self._connection_failure_logged:
            logger.debug(message)
            return
        self._connection_failure_logged = True
        logger.warning(message)
        logger.security_event(
            operation=operation,
            status="failure",
            host=self._channels.host,
            port=self._channels.port,
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
                self._connection_recycled = _is_connection_recycled_error(exp)
                match exp.code():
                    case grpc.StatusCode.DEADLINE_EXCEEDED:
                        self._message_deadline_exceeded()
                        await self._handle_after_process(ClientStatus.DISCONNECTED)
                    case grpc.StatusCode.UNAVAILABLE:
                        await self._on_status_change(ClientStatus.DISCONNECTED)
                        if self._connection_recycled:
                            # Diagnosing it would report a failure where there is none.
                            logger.debug(
                                f"Researcher retired the {self.__class__.__name__} "
                                "connection on its maximum age, will reconnect in "
                                f"{GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds"
                            )
                        elif _is_name_check_failure(exp):
                            # Verified under a name read from the certificate held here
                            await self._on_status_change(ClientStatus.FAILED)
                            if self._channels.mtls:
                                msg = (
                                    f"{ErrorNumbers.FB628.value}: gRPC refused the "
                                    f"researcher at {self._channels.endpoint}: it does "
                                    "not carry the name this node verifies it under, "
                                    "read from the registered researcher certificate. "
                                    "Register the researcher's current certificate on "
                                    "the node and restart it."
                                )
                            else:
                                msg = (
                                    f"{ErrorNumbers.FB628.value}: gRPC refused the "
                                    f"researcher at {self._channels.endpoint}: it does "
                                    "not carry the name this node verifies it under, "
                                    "read from the certificate fetched at startup. "
                                    "Restart the node, which fetches the certificate "
                                    "the researcher serves now."
                                )
                            logger.error(msg)
                            logger.security_event(
                                operation="researcher_failed_name_check",
                                status="failure",
                                host=self._channels.host,
                                port=self._channels.port,
                                detail=msg,
                            )
                            raise FedbiomedCommunicationError(msg) from exp
                        elif self._channels.mtls and _is_tls_handshake_error(exp):
                            self._log_connection_failure_once(
                                "Mutual authentication (mTLS) handshake with "
                                f"researcher failed in {self.__class__.__name__}: "
                                f"{exp.details()}. The certificates the two sides "
                                "hold for each other may not match, or a third party "
                                "may be answering for the researcher. Retrying; "
                                "repeats are logged at debug level."
                            )
                        elif (
                            self._channels.mtls
                            and _is_connection_closed_error(exp)
                            and await self._server_reachable()
                        ):
                            self._log_connection_failure_once(
                                f"The researcher at {self._channels.endpoint} is "
                                "reachable but closes the connection during the TLS "
                                f"handshake: {exp.details()}. Most often this node's "
                                "certificate is not registered there, or has expired; "
                                "a researcher that is restarting closes connections "
                                "the same way. Retrying; repeats are logged at debug "
                                "level."
                            )
                        elif (
                            not self._channels.mtls
                            and (
                                _is_connection_closed_error(exp)
                                or _is_tls_handshake_error(exp)
                            )
                            and await self._server_reachable()
                            and (
                                await asyncio.to_thread(
                                    _researcher_requires_client_auth,
                                    self._channels.host,
                                    self._channels.port,
                                )
                                is True
                            )
                        ):
                            # Static config on both sides, retry cannot help: stop.
                            await self._on_status_change(ClientStatus.FAILED)
                            msg = (
                                f"{ErrorNumbers.FB628.value}: The researcher requires "
                                "mutual authentication but it is disabled on this "
                                "node. Enable it in this node's `[authentication]` "
                                "configuration and register the researcher "
                                "certificate here, then request the researcher to "
                                "register this node's certificate."
                            )
                            logger.error(msg)
                            logger.security_event(
                                operation="mtls_required_by_researcher",
                                status="failure",
                                host=self._channels.host,
                                port=self._channels.port,
                                detail=msg,
                            )
                            raise FedbiomedCommunicationError(msg) from exp
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
                        self._log_connection_failure_once(
                            "Researcher rejected this node's identity in "
                            f"{self.__class__.__name__}: this node's certificate is "
                            "not registered there, or the node id it declares is not "
                            "the one that certificate is registered under. Retrying; "
                            "repeats are logged at debug level. The researcher "
                            f"replied: {exp.details()}",
                            operation="mtls_identity_rejected",
                        )
                        await self._handle_after_process(
                            ClientStatus.DISCONNECTED,
                            retry=self._retry_on_error,
                            reconnect=True,
                        )

                    case grpc.StatusCode.UNKNOWN | _:
                        msg = (
                            "Unexpected error raised by researcher gRPC server in "
                            f"{self.__class__.__name__}: {exp}. "
                            f"Will retry connect in {GRPC_CLIENT_CONN_RETRY_TIMEOUT} seconds "
                            f"to the channel {self._channels._channels} "
                            f"with stubs {self._channels._stubs}"
                        )
                        logger.error(msg)
                        logger.security_event(
                            operation="grpc_client_error",
                            status="failure",
                            origin="server",
                            grpc_status=exp.code().name,
                            detail=msg,
                        )
                        await self._handle_after_process(
                            ClientStatus.FAILED,
                            retry=self._retry_on_error,
                            reconnect=True,
                        )

            except _ResearcherAuthenticationPending as exp:
                self._connection_recycled = False
                self._log_connection_failure_once(
                    str(exp), operation="mtls_not_enforced_by_researcher"
                )
                await self._handle_after_process(
                    ClientStatus.DISCONNECTED, reconnect=True
                )
            except FedbiomedCommunicationError:
                # Raised by this client where retrying cannot resolve the problem:
                # let it stop the node instead of being retried as an unexpected
                # error.
                raise
            except (Exception, GeneratorExit) as exp:
                self._connection_recycled = False
                msg = (
                    f"Unexpected error raised by node gRPC client in {self.__class__.__name__}: "
                    f"{type(exp).__name__} : {exp} "
                    f"to the channel {self._channels._channels} "
                    f"with stubs {self._channels._stubs}"
                )
                logger.error(msg, exc_info=True)
                logger.security_event(
                    operation="grpc_client_error",
                    status="failure",
                    origin="client",
                    error_type=type(exp).__name__,
                    detail=msg,
                )
                await self._handle_after_process(
                    ClientStatus.FAILED, True, False, self._post_handle_raise, exp
                )
            else:
                self._connection_failure_logged = False
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
        """Reports, once per connection, that the node reached the researcher.

        The only place a connection is reported established: creating a channel
        opens none, so only an answer from the researcher proves the handshake
        succeeded.

        A connection replacing one retired on its maximum age tells an operator
        nothing the retired one did not, and is reported at debug level; the first
        connection and any following an interruption are announced. Every connection
        is recorded as a security event.
        """
        if self._communication_established:
            return
        self._communication_established = True

        log_level = logger.debug if self._connection_recycled else logger.info
        self._connection_recycled = False

        if self._channels.mtls:
            log_level(
                "Mutually authenticated communication established with researcher at "
                f"{self._channels.endpoint}; node identity verified by the researcher."
            )
        else:
            log_level(
                "Communication established with researcher at "
                f"{self._channels.endpoint} over server-authenticated TLS "
                "(node identity not verified)."
            )

        logger.security_event(
            operation="researcher_channel_established",
            status="success",
            researcher_id=certificate_component_id(self._channels.certificate),
            host=self._channels.host,
            port=self._channels.port,
            mtls=self._channels.mtls,
            **certificate_audit_fields(self._channels.certificate),
        )

    async def _require_researcher_verified_this_node(
        self, call: grpc.aio.UnaryStreamCall
    ) -> bool:
        """Checks the researcher named this node from the certificate it presented.

        The counterpart of the researcher requiring client certificates: a node
        configured for mutual authentication has no other way to learn whether the
        researcher enforces it, since gRPC does not tell a client whether its
        certificate was requested. Read from the response headers, which arrive when
        the call starts, so an idle federation does not delay the answer.

        A researcher rejects a declared id that its certificate does not resolve to
        before answering at all, so a correct one names this node or names nobody.
        Being named as another node is therefore not a configuration case but a
        researcher that does not behave as one: refused rather than trusted.

        A call that never reached the researcher carries no metadata either, which
        gRPC reports as empty headers rather than as an error. That absence is left
        to the RPC error handling instead of being read as an answer.

        Args:
            call: The task request whose initial metadata is read.

        Returns:
            Whether the researcher named this node; False when the call carried no
            answer to read.

        Raises:
            _ResearcherAuthenticationPending: the researcher named no node.
            FedbiomedCommunicationError: the researcher named another node.
        """
        named = dict(await call.initial_metadata()).get(MTLS_PEER_ID_HEADER)
        if named == self._node_id:
            return True

        if named is None and call.done() and await call.code() != grpc.StatusCode.OK:
            return False

        if named is None:
            raise _ResearcherAuthenticationPending(
                f"The researcher at {self._channels.endpoint} does not verify node "
                "identities: NO node in the federation is authenticated, this one "
                "included. Request the researcher to enable mutual authentication and "
                "to register this node's certificate; otherwise disable it in this "
                "node's `[authentication]` configuration. Retrying; repeats are "
                "logged at debug level."
            )

        # A researcher answering for someone else does not change by asking again.
        await self._on_status_change(ClientStatus.FAILED)
        msg = (
            f"{ErrorNumbers.FB628.value}: the researcher verified this connection as "
            f"`{named}`, not as `{self._node_id}`: the identity it reports is not "
            "this node's."
        )
        logger.error(msg)
        logger.security_event(
            operation="mtls_verified_as_another_node",
            status="failure",
            host=self._channels.host,
            port=self._channels.port,
            detail=msg,
        )
        raise FedbiomedCommunicationError(msg)

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
        if self._channels.mtls and await self._require_researcher_verified_this_node(
            iterator
        ):
            self._announce_communication_established()

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

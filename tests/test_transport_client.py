import asyncio
import ssl
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import grpc
import pytest

from fedbiomed.common.certificate_manager import (
    CERT_ORGANIZATION,
    CERT_PURPOSE_SERVER,
    CertificateManager,
)
from fedbiomed.common.constants import MAX_RETRIEVE_ERROR_RETRIES, MAX_SEND_RETRIES
from fedbiomed.common.exceptions import FedbiomedCommunicationError
from fedbiomed.common.message import (
    FeedbackMessage,
    Log,
    Scalar,
    SearchReply,
    SearchRequest,
)
from fedbiomed.transport.client import (
    MTLS_PEER_ID_HEADER,
    Channels,
    ClientStatus,
    GrpcClient,
    NodeClientIdentity,
    ResearcherCredentials,
    Sender,
    TaskListener,
    _is_tls_handshake_error,
    _researcher_requires_client_auth,
    _StubType,
)
from fedbiomed.transport.protocols.researcher_pb2 import TaskResponse
from fedbiomed.transport.protocols.researcher_pb2_grpc import ResearcherServiceStub

_RESEARCHER_A = "RESEARCHER_9c2b1d70-1111-2222-3333-444455556666"
_NODE_A = "NODE_4f2c8a10-0e7d-4a11-9c33-8b7f0a1d2e44"
_NODE_B = "NODE_0a1b2c3d-aaaa-bbbb-cccc-ddddeeeeffff"
_RESEARCHER_B = "RESEARCHER_7e6d5c40-9999-8888-7777-666655554444"


def _rpc_error(code, details=None):
    return grpc.aio.AioRpcError(
        code=code,
        trailing_metadata=grpc.aio.Metadata(("test", "test")),
        initial_metadata=grpc.aio.Metadata(("test", "test")),
        details=details,
    )


async def _async_iterator(items):
    for item in items:
        yield item


class _Call:
    """Stands in for the `UnaryStreamCall` gRPC returns for `GetTaskUnary`.

    Streams the given responses and answers `initial_metadata` with the headers,
    which is where the researcher names the node it authenticated. Answers with
    the `grpc.aio.Metadata` the real call returns, not a plain sequence of pairs,
    so the reading code is exercised against the type it meets in production.
    """

    def __init__(
        self,
        responses,
        metadata=((MTLS_PEER_ID_HEADER, _NODE_A),),
        code=grpc.StatusCode.OK,
        done=False,
    ):
        self._responses = responses
        self._metadata = grpc.aio.Metadata(*metadata)
        self._code = code
        self._done = done

    def __aiter__(self):
        return self._responses.__aiter__()

    async def initial_metadata(self):
        return self._metadata

    def done(self):
        return self._done

    async def code(self):
        return self._code


class _FailedCall(_Call):
    """Stands in for the call gRPC hands back when the connection never held.

    Its headers are empty and it is already terminated, the connection error
    surfacing only once it is iterated: the shape that makes empty headers
    indistinguishable from a researcher that verified nobody.
    """

    def __init__(self, error):
        super().__init__(None, metadata=(), code=error.code(), done=True)
        self._error = error

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self._error


async def _responses(bytes_):
    yield TaskResponse(bytes_=bytes_, iteration=0, size=0)


def _one_task(bytes_, **kwargs):
    return _Call(_responses(bytes_), **kwargs)


# -----------------------------------------------------------------------------
# GrpcClient
# -----------------------------------------------------------------------------


@pytest.fixture
def grpc_client():
    with (
        patch("fedbiomed.transport.client.ResearcherServiceStub", autospec=True),
        patch("fedbiomed.transport.client.Sender", autospec=True) as sender,
        patch("fedbiomed.transport.client.TaskListener", autospec=True),
    ):
        update_id_map = AsyncMock()
        yield SimpleNamespace(
            client=GrpcClient(
                node_id=_NODE_A,
                researcher=ResearcherCredentials(port="50051", host="localhost"),
                update_id_map=update_id_map,
            ),
            sender=sender,
            update_id_map=update_id_map,
        )


@pytest.mark.asyncio
async def test_grpc_client_start(grpc_client):
    task = grpc_client.client.start(on_task=MagicMock())
    assert isinstance(task, asyncio.Future)

    # Cancel the background task before it runs so it never opens a real
    # connection socket (would otherwise leak as a ResourceWarning).
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_grpc_client_send(grpc_client):
    message = {"test": "test"}
    await grpc_client.client.send(message)
    grpc_client.sender.return_value.send.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_grpc_client_on_status_change(grpc_client):
    await grpc_client.client._on_status_change(ClientStatus.CONNECTED)
    assert grpc_client.client._status == ClientStatus.CONNECTED


@pytest.mark.asyncio
async def test_grpc_client_update_id(grpc_client):
    client = grpc_client.client
    await client._update_id(id_=_RESEARCHER_A)
    # The observable effect: the endpoint is mapped to the researcher it answered as
    grpc_client.update_id_map.assert_called_once_with("localhost:50051", _RESEARCHER_A)

    # A second researcher answering on the endpoint the first is mapped to
    with pytest.raises(FedbiomedCommunicationError):
        await client._update_id(id_=_RESEARCHER_B)


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.certificate_component_id", autospec=True)
@patch("fedbiomed.transport.client.ssl.get_server_certificate", autospec=True)
@patch("fedbiomed.transport.client.is_server_alive", autospec=True)
async def test_grpc_client_connect_security_log(
    is_server_alive,
    get_server_certificate,
    component_id,
    security_event,
    grpc_client,
):
    is_server_alive.return_value = True
    get_server_certificate.return_value = "DUMMY-CERT"
    component_id.return_value = _RESEARCHER_A

    # Avoid creating real grpc channels
    grpc_client.client._channels.connect = AsyncMock()  # no spec

    await grpc_client.client._connect()

    grpc_client.client._channels.connect.assert_called_once()
    # Auto-trusting the researcher certificate is registered; the channel is not
    # yet established, so nothing reports it as such
    operations = [c.kwargs.get("operation") for c in security_event.call_args_list]
    assert operations == ["server_certificate_auto_trusted"]


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.security_event")
async def test_channel_event_identifies_researcher_certificate(
    security_event, listener_env, tmp_path
):
    """The node records which researcher certificate it connected with, once the
    researcher has answered."""
    _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=str(tmp_path),
        certificate_name="researcher",
        component_id=_RESEARCHER_A,
        purpose=CERT_PURPOSE_SERVER,
        san=["localhost"],
    )
    with open(pem_file, "rb") as f:
        listener_env.channels.certificate = f.read()

    await listener_env.drain([_one_task(b"t1")])

    events = [
        c.kwargs
        for c in security_event.call_args_list
        if c.kwargs.get("operation") == "researcher_channel_established"
    ]
    assert len(events) == 1
    assert events[0]["status"] == "success"
    # Read off the certificate itself, the node having been given no id to start from
    assert events[0]["researcher_id"] == _RESEARCHER_A
    assert events[0]["cert_subject"] == f"CN={_RESEARCHER_A},O={CERT_ORGANIZATION}"
    assert {"cert_issuer", "cert_serial", "cert_not_after"} <= events[0].keys()
    # The certificate itself is never emitted
    assert not any("BEGIN CERTIFICATE" in str(v) for v in events[0].values())


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.ssl.get_server_certificate", autospec=True)
@patch("fedbiomed.transport.client.certificate_component_id", autospec=True)
@patch("fedbiomed.transport.client.is_server_alive", autospec=True)
async def test_grpc_client_connect_pins_the_certificate_under_mutual_authentication(
    is_server_alive,
    component_id,
    get_server_certificate,
    grpc_client,
):
    """Under mutual authentication the researcher certificate is pinned, so the node
    connects with the certificate it was given rather than fetching one."""
    is_server_alive.return_value = True
    component_id.return_value = "test-researcher"

    client = GrpcClient(
        node_id=_NODE_A,
        researcher=ResearcherCredentials(
            port="50051", host="localhost", certificate=b"CERT", mtls=True
        ),
        update_id_map=grpc_client.update_id_map,
    )
    client._channels.connect = AsyncMock()

    await client._connect()

    get_server_certificate.assert_not_called()
    client._channels.connect.assert_called_once()


# -----------------------------------------------------------------------------
# TaskListener
# -----------------------------------------------------------------------------


@pytest.fixture
def listener_env():
    with patch("fedbiomed.transport.client.Serializer") as serializer:
        serializer.loads.return_value = SearchRequest(
            researcher_id="test-researcher-id",
            tags=["test"],
        ).to_dict()
        channels = MagicMock()
        # Deterministic placeholders for assertions on logged message content
        channels._channels = "CHANNELS"
        channels._stubs = "STUBS"
        # Real values: the listener probes this endpoint on connection-closed
        # errors under mutual authentication (port 1 is reliably closed).
        channels.host = "localhost"
        channels.port = "1"
        channels.connect = AsyncMock()
        # Off unless a test turns it on, as in a component that never enabled it:
        # a MagicMock attribute would read as enabled and demand the identity header.
        channels.mtls = False

        env = SimpleNamespace(
            serializer=serializer,
            channels=channels,
            on_status_change=AsyncMock(),
            update_id=AsyncMock(),
            callback=MagicMock(),
        )
        env.listener = TaskListener(
            channels=channels,
            node_id=_NODE_A,
            on_status_change=env.on_status_change,
            update_id=env.update_id,
        )

        async def drain(side_effects, expect=asyncio.CancelledError):
            """Runs the listener over the given GetTaskUnary results until it
            ends, `expect` being the exception expected to end it."""
            request_stub = MagicMock()
            channels.stub = AsyncMock(return_value=request_stub)
            request_stub.GetTaskUnary.side_effect = [
                *side_effects,
                asyncio.CancelledError,
            ]
            channels.endpoint = "localhost:50051"
            task = env.listener.listen(env.callback)
            with pytest.raises(expect):
                await task
            task.cancel()
            return request_stub

        env.drain = drain
        yield env


@pytest.mark.asyncio
async def test_task_listener_listen(listener_env):
    with patch("fedbiomed.transport.client.logger.debug") as logger_debug:
        request_stub = await listener_env.drain(
            [
                _Call(
                    _async_iterator(
                        [
                            TaskResponse(bytes_=b"test-1", iteration=0, size=1),
                            TaskResponse(bytes_=b"test-2", iteration=1, size=1),
                        ]
                    )
                )
            ]
        )

    listener_env.callback.assert_called_once()
    listener_env.serializer.loads.assert_called_once()
    assert request_stub.GetTaskUnary.call_count == 2
    listener_env.update_id.assert_called_once()
    debug_messages = [call.args[0] for call in logger_debug.call_args_list]
    assert any("Polling researcher for task" in msg for msg in debug_messages)
    assert any("[WIRE][S->N][RX]" in msg for msg in debug_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,sleeps,logs_error",
    [
        (grpc.StatusCode.DEADLINE_EXCEEDED, 0, False),
        (grpc.StatusCode.UNAVAILABLE, 1, False),
        (grpc.StatusCode.UNKNOWN, 1, True),
        (grpc.StatusCode.ABORTED, 1, True),
    ],
)
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.logger._logger.error")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_listen_grpc_exceptions(
    sleep, log_error, security_event, listener_env, code, sleeps, logs_error
):
    request_stub = await listener_env.drain([_rpc_error(code)])

    assert request_stub.GetTaskUnary.call_count == 2
    assert sleep.call_count == sleeps
    if logs_error:
        # Logged with channel/stub details, and registered as an audit event
        log_error.assert_called_once()
        log_args, _ = log_error.call_args
        assert "CHANNELS" in log_args[0]
        assert "STUBS" in log_args[0]
        security_event.assert_called_once()
        event = security_event.call_args.kwargs
        assert event["operation"] == "grpc_client_error"
        assert event["status"] == "failure"
        assert event["origin"] == "server"
        assert event["grpc_status"] == code.name
    else:
        log_error.assert_not_called()
        # A deadline with no task still reports the exchange that did complete
        operations = [c.kwargs.get("operation") for c in security_event.call_args_list]
        assert "grpc_client_error" not in operations


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [RuntimeError, Exception, GeneratorExit])
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.logger._logger.error")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_listen_non_grpc_exceptions(
    sleep, log_error, security_event, listener_env, exception
):
    """Retries are capped: beyond MAX_RETRIEVE_ERROR_RETRIES the listener stops
    with FedbiomedCommunicationError; a successful poll resets the counter."""
    request_stub = MagicMock()
    listener_env.channels.stub = AsyncMock(return_value=request_stub)

    # Wrap to count calls
    listener_env.listener._post_handle_raise = MagicMock(
        wraps=listener_env.listener._post_handle_raise
    )

    # Increasing number of errors until over the maximum authorized (MAX + 4)
    for nb_errors in range(1, MAX_RETRIEVE_ERROR_RETRIES + 5):
        request_stub.GetTaskUnary.side_effect = [exception] * nb_errors + [
            asyncio.CancelledError
        ]

        task = listener_env.listener.listen(listener_env.callback)
        if nb_errors <= MAX_RETRIEVE_ERROR_RETRIES:
            signal = asyncio.CancelledError
        else:
            signal = FedbiomedCommunicationError
        with pytest.raises(signal):
            await task

        # Logging assertions: audit event + exc_info + includes channel/stub details
        assert log_error.call_count >= 1
        log_args, log_kwargs = log_error.call_args
        assert "CHANNELS" in log_args[0]
        assert "STUBS" in log_args[0]
        assert log_kwargs.get("exc_info")
        assert security_event.call_count >= 1
        event = security_event.call_args.kwargs
        assert event["operation"] == "grpc_client_error"
        assert event["status"] == "failure"
        assert event["origin"] == "client"
        assert event["error_type"] == exception.__name__

        assert sleep.call_count == min(nb_errors, MAX_RETRIEVE_ERROR_RETRIES)
        assert request_stub.GetTaskUnary.call_count == min(
            nb_errors + 1, MAX_RETRIEVE_ERROR_RETRIES + 1
        )
        assert listener_env.listener._post_handle_raise.call_count == max(
            0, nb_errors - MAX_RETRIEVE_ERROR_RETRIES
        )

        task.cancel()

        # Need a successful task retrieve to reset the retry counters
        request_stub.GetTaskUnary.side_effect = [
            _Call(
                _async_iterator(
                    [
                        TaskResponse(bytes_=b"test-1", iteration=0, size=1),
                        TaskResponse(bytes_=b"test-2", iteration=1, size=1),
                    ]
                )
            ),
            asyncio.CancelledError,
        ]
        task = listener_env.listener.listen(listener_env.callback)
        with pytest.raises(asyncio.CancelledError):
            await task

        task.cancel()
        request_stub.reset_mock()
        sleep.reset_mock()
        log_error.reset_mock()


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.logger._logger.warning")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_unauthenticated_retries(
    sleep, log_warning, security_event, listener_env
):
    """An identity rejection is retried: registering the node's certificate on the
    researcher connects it with no restart."""
    rejection = _rpc_error(
        grpc.StatusCode.UNAUTHENTICATED,
        "its certificate is not registered",
    )

    request_stub = await listener_env.drain([rejection, rejection])

    # The channel is closed and reopened for each new attempt
    assert listener_env.channels.connect.await_count == 2
    assert request_stub.GetTaskUnary.call_count == 3
    listener_env.on_status_change.assert_any_await(ClientStatus.DISCONNECTED)
    assert (
        call(ClientStatus.FAILED) not in listener_env.on_status_change.await_args_list
    )
    # A warning, not an error: the node recovers on its own
    log_warning.assert_called_once()
    log_args, log_kwargs = log_warning.call_args
    assert "not registered there" in log_args[0]
    assert not log_kwargs.get("extra", {}).get("is_security")
    # The audit event names the endpoint that rejected this node
    event = security_event.call_args_list[0].kwargs
    assert event["operation"] == "mtls_identity_rejected"
    assert (event["host"], event["port"]) == ("localhost", "1")


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.logger._logger.warning")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_unavailable_mtls_handshake_logs_warning(
    sleep, log_warning, security_event, listener_env
):
    """Under mutual authentication, a handshake/pinning failure is logged loudly
    but still retried."""
    listener_env.channels.mtls = True

    request_stub = await listener_env.drain(
        [
            _rpc_error(
                grpc.StatusCode.UNAVAILABLE,
                "Ssl handshake failed: certificate verify failed",
            )
        ]
    )

    # Surfaced on the console (not a silent debug), still reconnects
    log_warning.assert_called_once()
    log_args, log_kwargs = log_warning.call_args
    assert "handshake with researcher failed" in log_args[0]
    assert not log_kwargs.get("extra", {}).get("is_security")
    sleep.assert_called_once()
    assert request_stub.GetTaskUnary.call_count == 2
    # The audit event names the endpoint the handshake failed against
    failure = security_event.call_args_list[0].kwargs
    assert failure["operation"] == "mtls_handshake_failure"
    assert (failure["host"], failure["port"]) == ("localhost", "1")


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger._logger.error")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_unavailable_plain_stays_debug(
    sleep, log_error, listener_env
):
    """Ordinary unavailability keeps the quiet debug-and-retry behaviour."""
    listener_env.channels.mtls = False

    request_stub = await listener_env.drain(
        [_rpc_error(grpc.StatusCode.UNAVAILABLE, "failed to connect to all addresses")]
    )

    # No security error, normal retry
    log_error.assert_not_called()
    sleep.assert_called_once()
    assert request_stub.GetTaskUnary.call_count == 2


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.info")
async def test_task_listener_announces_communication_once(log_info, listener_env):
    """A received task announces the server-auth channel, exactly once."""
    listener_env.channels.mtls = False
    await listener_env.drain([_one_task(b"t1"), _one_task(b"t2")])

    msgs = [
        c for c in log_info.call_args_list if "Communication established" in c.args[0]
    ]
    assert len(msgs) == 1
    assert "server-authenticated TLS" in msgs[0].args[0]
    assert "localhost:50051" in msgs[0].args[0]


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.info")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_mtls_announce_and_reannounce_on_reconnect(
    sleep, log_info, listener_env
):
    """An idle deadline confirms the authenticated channel; a reconnect re-announces
    it."""
    listener_env.channels.mtls = True
    await listener_env.drain(
        [
            _rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED, "deadline"),
            _rpc_error(grpc.StatusCode.UNAVAILABLE, "connection reset"),
            _rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED, "deadline"),
        ]
    )

    msgs = [
        c
        for c in log_info.call_args_list
        if "Mutually authenticated communication established" in c.args[0]
    ]
    assert len(msgs) == 2


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.logger.debug")
@patch("fedbiomed.transport.client.logger.info")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_demotes_a_recycled_connection(
    sleep, log_info, log_debug, security_event, listener_env
):
    """A researcher retiring a connection on its maximum age replaces it with an
    identical one, which is not announced again; it is still audited."""
    listener_env.channels.mtls = True
    listener_env.channels.certificate = None

    await listener_env.drain(
        [
            _Call(_async_iterator([])),
            _rpc_error(grpc.StatusCode.UNAVAILABLE, "max connection age"),
            _Call(_async_iterator([])),
        ]
    )

    announced = [
        c
        for c in log_info.call_args_list
        if "Mutually authenticated communication established" in c.args[0]
    ]
    assert len(announced) == 1
    assert any(
        "Mutually authenticated communication established" in c.args[0]
        for c in log_debug.call_args_list
    )
    operations = [c.kwargs.get("operation") for c in security_event.call_args_list]
    assert operations.count("researcher_channel_established") == 2


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.logger.info")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_announces_a_connection_recovered_after_a_failure(
    sleep, log_info, security_event, listener_env
):
    """A connection lost to anything but a retirement is an interruption, so the
    one that recovers from it is announced even after a retirement demoted one."""
    listener_env.channels.mtls = True
    listener_env.channels.certificate = None

    await listener_env.drain(
        [
            _Call(_async_iterator([])),
            _rpc_error(grpc.StatusCode.UNAVAILABLE, "max connection age"),
            _Call(_async_iterator([])),
            _rpc_error(grpc.StatusCode.UNAVAILABLE, "failed to connect"),
            _Call(_async_iterator([])),
        ]
    )

    announced = [
        c
        for c in log_info.call_args_list
        if "Mutually authenticated communication established" in c.args[0]
    ]
    assert len(announced) == 2


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.is_server_alive", return_value=True)
@patch("fedbiomed.transport.client.logger._logger.warning")
@patch("fedbiomed.transport.client.logger._logger.debug")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_mtls_rejection_logged_once(
    sleep, log_debug, log_warning, alive, listener_env
):
    """A reachable researcher closing the connection under mutual authentication is
    reported once as a suspected certificate rejection, then demoted to debug."""
    listener_env.channels.mtls = True
    closed = "ipv4:127.0.0.1:50051: Socket closed"
    await listener_env.drain(
        [
            _rpc_error(grpc.StatusCode.UNAVAILABLE, closed),
            _rpc_error(grpc.StatusCode.UNAVAILABLE, closed),
        ]
    )

    # One actionable console-visible warning despite two identical failures
    log_warning.assert_called_once()
    assert "is not registered there" in log_warning.call_args[0][0]
    # The repeat went to debug with the same explanation
    assert any("TLS handshake" in c.args[0] for c in log_debug.call_args_list)


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.logger._logger.info")
async def test_task_listener_announces_an_mtls_connection_on_the_headers(
    log_info, security_event, listener_env
):
    """The headers naming this node prove the handshake, so a federation with no
    task to send is still reported as connected."""
    listener_env.channels.mtls = True
    listener_env.channels.certificate = None

    await listener_env.drain([_Call(_async_iterator([]))])

    assert any(
        "Mutually authenticated communication established" in c.args[0]
        for c in log_info.call_args_list
    )
    operations = [c.kwargs.get("operation") for c in security_event.call_args_list]
    assert operations.count("researcher_channel_established") == 1


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.is_server_alive", return_value=True)
@patch("fedbiomed.transport.client.logger._logger.warning")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_reports_a_rejected_handshake_not_an_identity_answer(
    sleep, log_warning, alive, security_event, listener_env
):
    """A call that never reached the researcher must not be read as an answer.

    gRPC answers the headers of a failed call with empty metadata, which is what
    the researcher sends when it verifies nobody: taken as an answer, every
    rejected handshake would be reported as a researcher not enforcing mutual
    authentication.
    """
    listener_env.channels.mtls = True
    error = _rpc_error(
        grpc.StatusCode.UNAVAILABLE, "ipv4:127.0.0.1:50051: Socket closed"
    )

    await listener_env.drain([_FailedCall(error), _FailedCall(error)])

    log_warning.assert_called_once()
    assert (
        "closes the connection during the TLS handshake" in log_warning.call_args[0][0]
    )
    # A handshake that never completed is reported as no connection at all
    operations = [c.kwargs.get("operation") for c in security_event.call_args_list]
    assert "researcher_channel_established" not in operations


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger._logger.warning")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_reports_a_failed_tls_handshake_not_an_identity_answer(
    sleep, log_warning, listener_env
):
    """A researcher certificate that does not verify is reported as such."""
    listener_env.channels.mtls = True
    error = _rpc_error(
        grpc.StatusCode.UNAVAILABLE,
        "Tls handshake failed (TSI_PROTOCOL_FAILURE): CERTIFICATE_VERIFY_FAILED",
    )

    await listener_env.drain([_FailedCall(error)])

    log_warning.assert_called_once()
    assert "handshake with researcher failed" in log_warning.call_args[0][0]


@pytest.mark.asyncio
@patch("fedbiomed.transport.client._researcher_requires_client_auth", return_value=True)
@patch("fedbiomed.transport.client.is_server_alive", return_value=True)
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.logger._logger.error")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_non_mtls_node_against_mtls_researcher(
    sleep, log_error, security_event, alive, requires_auth, listener_env
):
    """A node without mutual authentication, rejected by a researcher enforcing it,
    stops: both sides are statically configured, so retrying cannot connect."""
    listener_env.channels.mtls = False
    await listener_env.drain(
        [
            _rpc_error(
                grpc.StatusCode.UNAVAILABLE, "ipv4:127.0.0.1:50051: Socket closed"
            )
        ],
        expect=FedbiomedCommunicationError,
    )

    errors = [c for c in log_error.call_args_list if "FB628" in c.args[0]]
    assert len(errors) == 1
    assert "mutual authentication but it is disabled on this node" in errors[0].args[0]
    assert "register the researcher certificate" in errors[0].args[0]
    operations = [c.kwargs.get("operation") for c in security_event.call_args_list]
    assert "mtls_required_by_researcher" in operations
    listener_env.on_status_change.assert_awaited_with(ClientStatus.FAILED)


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.logger._logger.warning")
async def test_task_listener_retries_when_researcher_names_no_node(
    log_warning, security_event, listener_env
):
    """A researcher not verifying identities is refused, however it behaves.

    The task is never read, so the node never runs with an identity nobody
    checked; it keeps trying, the researcher enabling mutual authentication being
    all that is needed to connect it.
    """
    listener_env.channels.mtls = True
    listener_env.channels.connect.reset_mock()

    await listener_env.drain(
        [_one_task(b"t1", metadata=()), _one_task(b"t2", metadata=())]
    )

    listener_env.callback.assert_not_called()
    # The channel is closed and reopened for each new attempt
    assert listener_env.channels.connect.await_count == 2
    log_warning.assert_called_once()
    assert "does not verify node identities" in log_warning.call_args[0][0]
    operations = [c.kwargs.get("operation") for c in security_event.call_args_list]
    assert "mtls_not_enforced_by_researcher" in operations
    listener_env.on_status_change.assert_any_await(ClientStatus.DISCONNECTED)
    assert (
        call(ClientStatus.FAILED) not in listener_env.on_status_change.await_args_list
    )


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.security_event")
@patch("fedbiomed.transport.client.logger._logger.error")
async def test_task_listener_stops_when_researcher_names_another_node(
    log_error, security_event, listener_env
):
    """Being named as another node is refused, not taken as an authentication.

    A researcher rejects a declared id its certificate does not resolve to before
    answering, so no correct one names another node: this guards against one that
    does not behave as a researcher, which is what makes the header worth reading.
    """
    listener_env.channels.mtls = True

    await listener_env.drain(
        [_one_task(b"t1", metadata=((MTLS_PEER_ID_HEADER, _NODE_B),))],
        expect=FedbiomedCommunicationError,
    )

    errors = [c for c in log_error.call_args_list if "FB628" in c.args[0]]
    assert f"verified this connection as `{_NODE_B}`" in errors[0].args[0]
    operations = [c.kwargs.get("operation") for c in security_event.call_args_list]
    assert "mtls_verified_as_another_node" in operations
    listener_env.on_status_change.assert_awaited_with(ClientStatus.FAILED)


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.security_event")
async def test_task_listener_ignores_the_header_without_mutual_authentication(
    security_event, listener_env
):
    """A node that did not enable it does not require the researcher to name it."""
    listener_env.channels.mtls = False

    await listener_env.drain([_one_task(b"t1", metadata=())])

    listener_env.callback.assert_called_once()
    operations = [c.kwargs.get("operation") for c in security_event.call_args_list]
    assert "mtls_not_enforced_by_researcher" not in operations


# -----------------------------------------------------------------------------
# TLS/pinning failure discriminator
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "details",
    [
        "Ssl handshake failed",
        "CERTIFICATE_VERIFY_FAILED",
        "TLS peer did not return a certificate",
    ],
)
def test_detects_handshake_failures(details):
    error = _rpc_error(grpc.StatusCode.UNAVAILABLE, details)
    assert _is_tls_handshake_error(error)


def test_ignores_ordinary_unavailability():
    error = _rpc_error(
        grpc.StatusCode.UNAVAILABLE, "failed to connect to all addresses"
    )
    assert not _is_tls_handshake_error(error)


# -----------------------------------------------------------------------------
# Mutual authentication client-auth probe
#
# The probe's real behaviour is covered against live gRPC servers in
# `test_transport_mtls.py`; these only pin the conservative failure paths.
# -----------------------------------------------------------------------------


@patch("fedbiomed.transport.client.socket.create_connection")
@patch("fedbiomed.transport.client.ssl.create_default_context")
def test_probe_true_when_handshake_rejected(context, create_connection):
    context.return_value.wrap_socket.side_effect = ssl.SSLError(
        "peer did not return a certificate"
    )
    assert _researcher_requires_client_auth("localhost", "50051")


@patch("fedbiomed.transport.client.socket.create_connection")
@patch("fedbiomed.transport.client.ssl.create_default_context")
def test_probe_true_when_server_closes_without_replying(context, create_connection):
    # An enforcing server under TLS 1.3 completes the handshake, then closes.
    wrap_socket = context.return_value.wrap_socket.return_value
    wrap_socket.__enter__.return_value.recv.return_value = b""
    assert _researcher_requires_client_auth("localhost", "50051")


@patch("fedbiomed.transport.client.socket.create_connection")
@patch("fedbiomed.transport.client.ssl.create_default_context")
def test_probe_false_when_server_replies(context, create_connection):
    wrap_socket = context.return_value.wrap_socket.return_value
    wrap_socket.__enter__.return_value.recv.return_value = b"\x00"
    assert not _researcher_requires_client_auth("localhost", "50051")


@patch("fedbiomed.transport.client.socket.create_connection")
@patch("fedbiomed.transport.client.ssl.create_default_context")
def test_probe_true_when_server_aborts_the_connection(context, create_connection):
    wrap_socket = context.return_value.wrap_socket.return_value
    wrap_socket.__enter__.return_value.recv.side_effect = ConnectionResetError()
    assert _researcher_requires_client_auth("localhost", "50051") is True


# A probe that learns nothing must say so: reporting "required" would let a slow
# or unreachable server look like a researcher enforcing mutual authentication.
@patch("fedbiomed.transport.client.socket.create_connection")
@patch("fedbiomed.transport.client.ssl.create_default_context")
def test_probe_unknown_when_connection_times_out(context, create_connection):
    wrap_socket = context.return_value.wrap_socket.return_value
    wrap_socket.__enter__.return_value.recv.side_effect = TimeoutError()
    assert _researcher_requires_client_auth("localhost", "50051") is None


# -----------------------------------------------------------------------------
# Sender
# -----------------------------------------------------------------------------

message_search = SearchReply(
    researcher_id="test",
    databases=[],
    node_id="node-id",
    node_name="node-name",
    count=1,
)

message_log = FeedbackMessage(
    researcher_id="test",
    log=Log(node_id="test", level="DEBUG", msg="Error message"),
)

message_scalar = FeedbackMessage(
    researcher_id="test",
    scalar=Scalar(
        node_id="test",
        node_name="test-name",
        experiment_id="my_exp",
        train=True,
        test=False,
        test_on_local_updates=False,
        test_on_global_updates=False,
        metric={},
        total_samples=3,
        batch_samples=2,
        num_batches=1,
        iteration=1,
        epoch=2,
        num_samples_trained=3,
    ),
)


@pytest.fixture
def sender_env(request):
    serializer_patch = patch("fedbiomed.transport.client.Serializer")
    serializer_patch.start()
    request.addfinalizer(serializer_patch.stop)

    channels = MagicMock()
    channels.stub = AsyncMock()
    channels.connect = AsyncMock()
    channels.feedback_stub.Feedback = MagicMock(spec=grpc.aio.UnaryUnaryMultiCallable)
    channels.task_stub.ReplyTask = MagicMock(spec=grpc.aio.StreamUnaryMultiCallable)
    return SimpleNamespace(
        serializer_patch=serializer_patch,
        channels=channels,
        sender=Sender(channels=channels, on_status_change=AsyncMock()),
    )


async def _sender_feedback_cycle(env, message, side_effects):
    """Queues `message` twice and runs the sender over the given Feedback results."""
    env.channels.stub.return_value = env.channels.feedback_stub
    env.channels.feedback_stub.Feedback.side_effect = side_effects
    await env.sender.send(message=message)
    await env.sender.send(message=message)
    return env.sender.listen()


async def _sender_reset(env, message):
    """A successful send cycle, resetting the sender retry counters."""
    future = asyncio.Future()
    future.set_result("x")
    task = await _sender_feedback_cycle(env, message, [future, asyncio.CancelledError])
    with pytest.raises(asyncio.CancelledError):
        await task
    task.cancel()
    env.channels.feedback_stub.reset_mock()


@pytest.mark.asyncio
async def test_sender_send(sender_env):
    await sender_env.sender.send(message=message_search)
    item = await sender_env.sender._queue.get()
    assert item == {"stub": _StubType.SENDER_TASK_STUB, "message": message_search}

    await sender_env.sender.send(message=message_log)
    item = await sender_env.sender._queue.get()
    assert item == {"stub": _StubType.SENDER_FEEDBACK_STUB, "message": message_log}

    await sender_env.sender.send(message=message_scalar)
    item = await sender_env.sender._queue.get()
    assert item == {"stub": _StubType.SENDER_FEEDBACK_STUB, "message": message_scalar}


@pytest.mark.asyncio
async def test_sender_listen(sender_env):
    sender_env.serializer_patch.stop()

    future = asyncio.Future()
    future.set_result("x")

    task = await _sender_feedback_cycle(
        sender_env, message_log, [future, asyncio.CancelledError]
    )
    with patch("fedbiomed.transport.client.logger.debug") as logger_debug:
        with pytest.raises(asyncio.CancelledError):
            await task
    assert sender_env.channels.feedback_stub.Feedback.call_count == 2
    assert any(
        "[WIRE][N->S][TX]" in call.args[0] for call in logger_debug.call_args_list
    )

    task.cancel()

    stream_call = AsyncMock()
    sender_env.channels.task_stub.ReplyTask.side_effect = [
        stream_call,
        asyncio.CancelledError,
    ]
    sender_env.channels.stub.return_value = sender_env.channels.task_stub
    await sender_env.sender.send(message=message_search)
    await sender_env.sender.send(message=message_search)

    with patch("fedbiomed.transport.client.logger.debug") as logger_debug:
        task = sender_env.sender.listen()
        with pytest.raises(asyncio.CancelledError):
            await task

    task.cancel()
    assert sender_env.channels.task_stub.ReplyTask.call_count == 2
    stream_call.write.assert_called_once()
    stream_call.done_writing.assert_called_once()
    assert any(
        "[WIRE][N->S][TX]" in call.args[0] for call in logger_debug.call_args_list
    )

    # Restart for the fixture finalizer's stop
    sender_env.serializer_patch.start()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [message_log, message_scalar])
@pytest.mark.parametrize(
    "code",
    [grpc.StatusCode.UNKNOWN, grpc.StatusCode.ABORTED, grpc.StatusCode.UNAVAILABLE],
)
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_sender_listen_retryable_grpc_errors(sleep, sender_env, message, code):
    """Retryable gRPC errors re-send after a pause; a success resets counters."""
    for retry in range(1, MAX_SEND_RETRIES + 5):
        task = await _sender_feedback_cycle(
            sender_env, message, [_rpc_error(code)] * retry + [asyncio.CancelledError]
        )
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sender_env.channels.feedback_stub.Feedback.call_count == retry + 1
        assert sleep.call_count == retry - int((retry - 1) / MAX_SEND_RETRIES)

        task.cancel()
        await _sender_reset(sender_env, message)
        sleep.reset_mock()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [message_log, message_scalar])
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_sender_listen_deadline_consumes_message_without_sleep(
    sleep, sender_env, message
):
    """A deadline consumes the current message and re-sends the next
    immediately, without pausing."""
    deadlines = 3
    sender_env.channels.stub.return_value = sender_env.channels.feedback_stub
    sender_env.channels.feedback_stub.Feedback.side_effect = [
        _rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED)
    ] * deadlines + [asyncio.CancelledError]
    # One message per deadline, plus one carrying the terminating error
    for _ in range(deadlines + 1):
        await sender_env.sender.send(message=message)

    task = sender_env.sender.listen()
    with pytest.raises(asyncio.CancelledError):
        await task
    task.cancel()

    assert sender_env.channels.feedback_stub.Feedback.call_count == deadlines + 1
    sleep.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [message_log, message_scalar])
@pytest.mark.parametrize("exception", [RuntimeError, Exception, GeneratorExit])
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_sender_listen_non_grpc_exceptions(sleep, sender_env, message, exception):
    """Non-gRPC errors are capped: beyond MAX_SEND_RETRIES the sender stops
    with FedbiomedCommunicationError; a success resets the counter."""
    for retry in range(1, MAX_SEND_RETRIES + 5):
        task = await _sender_feedback_cycle(
            sender_env, message, [exception] * retry + [asyncio.CancelledError]
        )
        if retry <= MAX_SEND_RETRIES:
            signal = asyncio.CancelledError
        else:
            signal = FedbiomedCommunicationError

        with pytest.raises(signal):
            await task
        assert sender_env.channels.feedback_stub.Feedback.call_count == min(
            retry + 1, MAX_SEND_RETRIES + 1
        )
        assert sleep.call_count == min(retry, MAX_SEND_RETRIES)

        task.cancel()
        await _sender_reset(sender_env, message)
        sleep.reset_mock()


# -----------------------------------------------------------------------------
# Channels
# -----------------------------------------------------------------------------


@pytest.fixture
def channels_env():
    with (
        patch(
            "fedbiomed.transport.client.Channels._create_channel", autospec=True
        ) as create_channel,
        patch("fedbiomed.transport.client.ResearcherServiceStub", autospec=True),
    ):
        create_channel.return_value.close = AsyncMock()
        yield SimpleNamespace(
            create_channel=create_channel,
            channels=Channels(
                researcher=ResearcherCredentials(
                    host="localhost", port="50051", certificate=b"test"
                )
            ),
        )


def test_channels_endpoint(channels_env):
    assert channels_env.channels.endpoint == "localhost:50051"


@pytest.mark.asyncio
async def test_channels_connect_and_stub(channels_env):
    stubs = [
        _StubType.LISTENER_TASK_STUB,
        _StubType.SENDER_TASK_STUB,
        _StubType.SENDER_FEEDBACK_STUB,
    ]
    await channels_env.channels.connect()
    for stub in stubs:
        assert isinstance(await channels_env.channels.stub(stub), ResearcherServiceStub)

    # Recall connect
    await channels_env.channels.connect()
    for stub in stubs:
        assert isinstance(await channels_env.channels.stub(stub), ResearcherServiceStub)

    # test non existing stub
    assert await channels_env.channels.stub("dummy") is None


@patch("fedbiomed.transport.client.certificate_san_names")
@patch("fedbiomed.transport.client.grpc.ssl_channel_credentials")
def test_channels_create_without_mtls(
    ssl_channel_credentials, certificate_san_names, channels_env
):
    """Without mutual authentication only the server certificate is pinned, and the
    researcher is still verified against a name that certificate carries."""
    certificate_san_names.return_value = ["fbm-researcher"]
    channels = Channels(
        researcher=ResearcherCredentials(
            host="localhost", port="50051", certificate=b"server-cert"
        )
    )

    channels._create()

    # Server certificate pinned, no client identity presented
    ssl_channel_credentials.assert_called_once_with(b"server-cert")
    _, kwargs = channels_env.create_channel.call_args
    assert kwargs["target_name_override"] == "fbm-researcher"
    assert kwargs["certificate"] == ssl_channel_credentials.return_value


@patch("fedbiomed.transport.client.certificate_san_names")
@patch("fedbiomed.transport.client.grpc.ssl_channel_credentials")
def test_channels_create_with_mtls(
    ssl_channel_credentials, certificate_san_names, channels_env
):
    """Under mutual authentication the node presents its identity and still pins
    the name."""
    certificate_san_names.return_value = ["fbm-researcher"]
    channels = Channels(
        researcher=ResearcherCredentials(
            host="localhost",
            port="50051",
            certificate=b"server-cert",
            mtls=True,
            node_identity=NodeClientIdentity(
                private_key=b"node-key",
                certificate_chain=b"node-cert",
            ),
        )
    )

    channels._create()

    ssl_channel_credentials.assert_called_once_with(
        root_certificates=b"server-cert",
        private_key=b"node-key",
        certificate_chain=b"node-cert",
    )
    certificate_san_names.assert_called_once_with(b"server-cert")
    _, kwargs = channels_env.create_channel.call_args
    assert kwargs["target_name_override"] == "fbm-researcher"


@patch("fedbiomed.transport.client.certificate_san_names")
def test_channels_verify_the_dialled_address_when_named(
    certificate_san_names, channels_env
):
    """No override where the certificate names the address: the address dialled is
    then what TLS verifies, which is what the name check is for."""
    certificate_san_names.return_value = ["fbm-researcher", "localhost", "127.0.0.1"]

    Channels(
        researcher=ResearcherCredentials(
            host="localhost", port="50051", certificate=b"server-cert"
        )
    )._create()

    _, kwargs = channels_env.create_channel.call_args
    assert kwargs["target_name_override"] is None


@patch("fedbiomed.transport.client.certificate_san_names")
def test_channels_fall_back_to_the_first_name_for_an_unnamed_address(
    certificate_san_names, channels_env
):
    """A certificate is issued when the component is created, so it cannot always
    name the address nodes reach it at."""
    certificate_san_names.return_value = ["fbm-researcher", "fbm.example.org"]

    Channels(
        researcher=ResearcherCredentials(
            host="10.0.0.9", port="50051", certificate=b"server-cert"
        )
    )._create()

    _, kwargs = channels_env.create_channel.call_args
    assert kwargs["target_name_override"] == "fbm-researcher"


@patch("fedbiomed.transport.client.certificate_san_names")
def test_channels_verify_the_dialled_address_for_a_nameless_certificate(
    certificate_san_names, channels_env
):
    """A nameless certificate leaves nothing to override with."""
    certificate_san_names.return_value = []

    Channels(
        researcher=ResearcherCredentials(
            host="10.0.0.9", port="50051", certificate=b"server-cert"
        )
    )._create()

    _, kwargs = channels_env.create_channel.call_args
    assert kwargs["target_name_override"] is None


def test_channels_create_channel_adds_target_name_override():
    """`target_name_override` is forwarded as a gRPC channel option."""
    with patch("fedbiomed.transport.client.grpc.aio.secure_channel") as secure_channel:
        Channels._create_channel(
            port="50051",
            host="localhost",
            certificate=MagicMock(),
            target_name_override="fbm-researcher",
        )
    options = dict(secure_channel.call_args.kwargs["options"])
    assert options.get("grpc.ssl_target_name_override") == "fbm-researcher"


def test_channels_create_channel_omits_override_when_absent():
    """No override option is set when `target_name_override` is None."""
    with patch("fedbiomed.transport.client.grpc.aio.secure_channel") as secure_channel:
        Channels._create_channel(
            port="50051",
            host="localhost",
            certificate=MagicMock(),
            target_name_override=None,
        )
    options = dict(secure_channel.call_args.kwargs["options"])
    assert "grpc.ssl_target_name_override" not in options

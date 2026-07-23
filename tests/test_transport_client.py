import asyncio
import ssl
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from cryptography import x509

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


async def _one_task(bytes_):
    yield TaskResponse(bytes_=bytes_, iteration=0, size=0)


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
                node_id="test-node-id",
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
    await client._update_id(id_="test")
    assert client._id == "test"
    grpc_client.update_id_map.assert_called_once_with(
        f"{client._researcher.host}:{client._researcher.port}", "test"
    )

    with pytest.raises(FedbiomedCommunicationError):
        await client._update_id(id_="test-malicious")


@pytest.mark.asyncio
@patch(
    "fedbiomed.transport.client._researcher_requires_client_auth", return_value=False
)
@patch("fedbiomed.transport.client.logger._logger.info")
@patch("fedbiomed.transport.client.x509.load_pem_x509_certificate", autospec=True)
@patch("fedbiomed.transport.client.ssl.get_server_certificate", autospec=True)
@patch("fedbiomed.transport.client.is_server_alive", autospec=True)
async def test_grpc_client_connect_security_log(
    is_server_alive,
    get_server_certificate,
    load_pem_x509_certificate,
    log_info,
    requires_client_auth,
    grpc_client,
):
    is_server_alive.return_value = True
    get_server_certificate.return_value = "DUMMY-CERT"

    load_pem_x509_certificate.return_value = MagicMock(
        subject=MagicMock(
            get_attributes_for_oid=MagicMock(
                return_value=[MagicMock(value="test-researcher")]
            )
        )
    )

    # Avoid creating real grpc channels
    grpc_client.client._channels.connect = AsyncMock()  # no spec

    await grpc_client.client._connect()

    grpc_client.client._channels.connect.assert_called_once()
    security_calls = [
        c
        for c in log_info.call_args_list
        if c.kwargs.get("extra", {}).get("is_security") is True
    ]
    assert len(security_calls) == 1
    assert security_calls[0].args[0]


@pytest.mark.asyncio
@patch("fedbiomed.transport.client._researcher_requires_client_auth", return_value=True)
@patch("fedbiomed.transport.client.logger._logger.error")
@patch("fedbiomed.transport.client.ssl.get_server_certificate", autospec=True)
@patch("fedbiomed.transport.client.asyncio.sleep")
@patch("fedbiomed.transport.client.is_server_alive", autospec=True)
async def test_grpc_client_connect_refuses_in_band_cert_from_mtls_researcher(
    is_server_alive,
    sleep,
    get_server_certificate,
    log_error,
    requires_client_auth,
    grpc_client,
):
    """A non-mTLS node facing an mTLS-enforcing researcher never adopts an
    in-band certificate: it reports the mismatch and keeps waiting."""
    is_server_alive.return_value = True
    # Break out of the connect loop after two retry sleeps
    sleep.side_effect = [None, asyncio.CancelledError]
    grpc_client.client._channels.connect = AsyncMock()  # no spec

    with pytest.raises(asyncio.CancelledError):
        await grpc_client.client._connect()

    # No in-band certificate fetch, no channel creation
    get_server_certificate.assert_not_called()
    grpc_client.client._channels.connect.assert_not_called()
    # One console-visible error despite two detections
    errors = [c for c in log_error.call_args_list if "FB628" in c.args[0]]
    assert len(errors) == 1
    assert "mutual-TLS is disabled on this node" in errors[0].args[0]


# Warn iff the researcher does not require the node's client certificate
@pytest.mark.asyncio
@pytest.mark.parametrize("requires_auth,expected_warnings", [(False, 1), (True, 0)])
@patch("fedbiomed.transport.client.logger._logger.warning")
@patch("fedbiomed.transport.client._researcher_requires_client_auth", autospec=True)
@patch("fedbiomed.transport.client.certificate_subject_field", autospec=True)
@patch("fedbiomed.transport.client.is_server_alive", autospec=True)
async def test_grpc_client_connect_mtls_warns_only_when_not_enforced(
    is_server_alive,
    subject_field,
    requires_client_auth,
    log_warning,
    grpc_client,
    requires_auth,
    expected_warnings,
):
    is_server_alive.return_value = True
    subject_field.return_value = "test-researcher"
    requires_client_auth.return_value = requires_auth

    client = GrpcClient(
        node_id="test-node-id",
        researcher=ResearcherCredentials(
            port="50051", host="localhost", certificate=b"CERT", mtls=True
        ),
        update_id_map=grpc_client.update_id_map,
    )
    client._channels.connect = AsyncMock()

    await client._connect()

    # Console-visible warning (no security flag: flagged records are diverted
    # to the security file)
    warnings = [
        c
        for c in log_warning.call_args_list
        if "node identity will NOT be verified" in c.args[0]
    ]
    assert len(warnings) == expected_warnings


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
        # errors under mTLS (port 1 is reliably closed).
        channels.host = "localhost"
        channels.port = "1"
        channels.connect = AsyncMock()

        env = SimpleNamespace(
            serializer=serializer,
            channels=channels,
            on_status_change=AsyncMock(),
            update_id=AsyncMock(),
            callback=MagicMock(),
        )
        env.listener = TaskListener(
            channels=channels,
            node_id="test-node-id",
            on_status_change=env.on_status_change,
            update_id=env.update_id,
        )

        async def drain(side_effects):
            """Runs the listener over the given GetTaskUnary results until
            cancelled."""
            request_stub = MagicMock()
            channels.stub = AsyncMock(return_value=request_stub)
            request_stub.GetTaskUnary.side_effect = [
                *side_effects,
                asyncio.CancelledError,
            ]
            channels.endpoint = "localhost:50051"
            task = env.listener.listen(env.callback)
            with pytest.raises(asyncio.CancelledError):
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
                _async_iterator(
                    [
                        TaskResponse(bytes_=b"test-1", iteration=0, size=1),
                        TaskResponse(bytes_=b"test-2", iteration=1, size=1),
                    ]
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
@patch("fedbiomed.transport.client.logger._logger.error")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_listen_grpc_exceptions(
    sleep, log_error, listener_env, code, sleeps, logs_error
):
    request_stub = await listener_env.drain([_rpc_error(code)])

    assert request_stub.GetTaskUnary.call_count == 2
    assert sleep.call_count == sleeps
    if logs_error:
        # Logged with channel/stub details as a security record
        log_error.assert_called_once()
        log_args, log_kwargs = log_error.call_args
        assert "CHANNELS" in log_args[0]
        assert "STUBS" in log_args[0]
        assert log_kwargs["extra"].get("is_security")
    else:
        log_error.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [RuntimeError, Exception, GeneratorExit])
@patch("fedbiomed.transport.client.logger._logger.error")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_listen_non_grpc_exceptions(
    sleep, log_error, listener_env, exception
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

        # Logging assertions: security + exc_info + includes channel/stub details
        assert log_error.call_count >= 1
        log_args, log_kwargs = log_error.call_args
        assert "CHANNELS" in log_args[0]
        assert "STUBS" in log_args[0]
        assert log_kwargs.get("exc_info")
        assert log_kwargs["extra"].get("is_security")

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
            _async_iterator(
                [
                    TaskResponse(bytes_=b"test-1", iteration=0, size=1),
                    TaskResponse(bytes_=b"test-2", iteration=1, size=1),
                ]
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
@patch("fedbiomed.transport.client.logger._logger.error")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_unauthenticated_stops(sleep, log_error, listener_env):
    """A researcher identity rejection (UNAUTHENTICATED) is fatal, not retried."""
    request_stub = MagicMock()
    listener_env.channels.stub = AsyncMock(return_value=request_stub)
    listener_env.channels.connect.reset_mock()

    request_stub.GetTaskUnary.side_effect = [
        _rpc_error(
            grpc.StatusCode.UNAUTHENTICATED,
            "declared node id does not match certificate",
        ),
    ]

    task = listener_env.listener.listen(listener_env.callback)
    with pytest.raises(FedbiomedCommunicationError):
        await task

    # Identity rejection is permanent: no reconnect, no retry sleep, no re-poll
    listener_env.channels.connect.assert_not_called()
    sleep.assert_not_called()
    assert request_stub.GetTaskUnary.call_count == 1
    # Node is marked failed
    listener_env.on_status_change.assert_awaited_with(ClientStatus.FAILED)
    # Console-visible FB628 error (no security flag: flagged records are
    # diverted to the security file and would not reach the user)
    log_error.assert_called_once()
    log_args, log_kwargs = log_error.call_args
    assert "FB628" in log_args[0]
    assert not log_kwargs.get("extra", {}).get("is_security")

    task.cancel()


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger._logger.error")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_unavailable_mtls_handshake_logs_error(
    sleep, log_error, listener_env
):
    """Under mTLS, a handshake/pinning failure is logged loudly but still retried."""
    listener_env.channels.mtls = True

    request_stub = await listener_env.drain(
        [
            _rpc_error(
                grpc.StatusCode.UNAVAILABLE,
                "Ssl handshake failed: certificate verify failed",
            )
        ]
    )

    # Surfaced as a console-visible error (not a silent debug), still reconnects
    log_error.assert_called_once()
    log_args, log_kwargs = log_error.call_args
    assert "FB628" in log_args[0]
    assert not log_kwargs.get("extra", {}).get("is_security")
    sleep.assert_called_once()
    assert request_stub.GetTaskUnary.call_count == 2


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
    """An idle deadline confirms the mTLS channel; a reconnect re-announces it."""
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
        if "Mutual-TLS communication established" in c.args[0]
    ]
    assert len(msgs) == 2


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.is_server_alive", return_value=True)
@patch("fedbiomed.transport.client.logger._logger.error")
@patch("fedbiomed.transport.client.logger._logger.debug")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_mtls_rejection_logged_once(
    sleep, log_debug, log_error, alive, listener_env
):
    """A reachable researcher closing the connection under mTLS is reported
    once as a suspected certificate rejection, then demoted to debug."""
    listener_env.channels.mtls = True
    closed = "ipv4:127.0.0.1:50051: Socket closed"
    await listener_env.drain(
        [
            _rpc_error(grpc.StatusCode.UNAVAILABLE, closed),
            _rpc_error(grpc.StatusCode.UNAVAILABLE, closed),
        ]
    )

    # One actionable console-visible error despite two identical failures
    errors = [c for c in log_error.call_args_list if "FB628" in c.args[0]]
    assert len(errors) == 1
    assert "Ask the researcher to register it" in errors[0].args[0]
    # The repeat went to debug with the same explanation
    assert any("TLS handshake" in c.args[0] for c in log_debug.call_args_list)


@pytest.mark.asyncio
@patch("fedbiomed.transport.client._researcher_requires_client_auth", return_value=True)
@patch("fedbiomed.transport.client.is_server_alive", return_value=True)
@patch("fedbiomed.transport.client.logger._logger.error")
@patch("fedbiomed.transport.client.asyncio.sleep")
async def test_task_listener_non_mtls_node_against_mtls_researcher(
    sleep, log_error, alive, requires_auth, listener_env
):
    """A non-mTLS node rejected by a researcher enforcing client authentication
    is told to enable mutual TLS."""
    listener_env.channels.mtls = False
    await listener_env.drain(
        [_rpc_error(grpc.StatusCode.UNAVAILABLE, "ipv4:127.0.0.1:50051: Socket closed")]
    )

    errors = [c for c in log_error.call_args_list if "FB628" in c.args[0]]
    assert len(errors) == 1
    assert "mutual-TLS is disabled on this node" in errors[0].args[0]
    assert "register the researcher certificate" in errors[0].args[0]


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.logger.info")
async def test_task_listener_announce_honest_when_not_enforced(log_info, listener_env):
    """When the researcher accepts anonymous clients, the announce does not
    claim the node identity was verified."""
    listener_env.channels.mtls = True
    listener_env.channels.client_auth_enforced = False

    await listener_env.drain([_one_task(b"t1")])

    msgs = [
        c for c in log_info.call_args_list if "Communication established" in c.args[0]
    ]
    assert len(msgs) == 1
    assert "NOT verified" in msgs[0].args[0]
    assert "Mutual-TLS communication established" not in msgs[0].args[0]


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
# Mutual-TLS client-auth probe
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
def test_probe_true_when_connection_times_out(context, create_connection):
    wrap_socket = context.return_value.wrap_socket.return_value
    wrap_socket.__enter__.return_value.recv.side_effect = TimeoutError()
    assert _researcher_requires_client_auth("localhost", "50051")


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


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.grpc.ssl_channel_credentials")
async def test_channels_create_without_mtls(ssl_channel_credentials, channels_env):
    """Without mutual TLS only the server certificate is pinned."""
    channels = Channels(
        researcher=ResearcherCredentials(
            host="localhost", port="50051", certificate=b"server-cert"
        )
    )

    channels._create()

    # Server certificate pinned, no client identity, no target-name override
    ssl_channel_credentials.assert_called_once_with(b"server-cert")
    _, kwargs = channels_env.create_channel.call_args
    assert kwargs["target_name_override"] is None
    assert kwargs["certificate"] == ssl_channel_credentials.return_value


@pytest.mark.asyncio
@patch("fedbiomed.transport.client.certificate_subject_field")
@patch("fedbiomed.transport.client.grpc.ssl_channel_credentials")
async def test_channels_create_with_mtls(
    ssl_channel_credentials, subject_field, channels_env
):
    """With mutual TLS the node presents its identity and pins the CN."""
    subject_field.return_value = "researcher-cn"
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
    subject_field.assert_called_once_with(b"server-cert", x509.oid.NameOID.COMMON_NAME)
    _, kwargs = channels_env.create_channel.call_args
    assert kwargs["target_name_override"] == "researcher-cn"


def test_channels_create_channel_adds_target_name_override():
    """`target_name_override` is forwarded as a gRPC channel option."""
    with patch("fedbiomed.transport.client.grpc.aio.secure_channel") as secure_channel:
        Channels._create_channel(
            port="50051",
            host="localhost",
            certificate=MagicMock(),
            target_name_override="researcher-cn",
        )
    options = dict(secure_channel.call_args.kwargs["options"])
    assert options.get("grpc.ssl_target_name_override") == "researcher-cn"


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

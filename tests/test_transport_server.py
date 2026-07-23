import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from fedbiomed.common.exceptions import FedbiomedCommunicationError
from fedbiomed.common.message import OverlayMessage, SearchReply, SearchRequest
from fedbiomed.researcher.config import ResearcherConfig
from fedbiomed.transport.node_agent import AgentStore, NodeActiveStatus
from fedbiomed.transport.protocols.researcher_pb2 import (
    Empty,
    FeedbackMessage,
    TaskRequest,
    TaskResult,
)
from fedbiomed.transport.server import (
    GrpcServer,
    NodeAgent,
    ResearcherServicer,
    _GrpcAsyncServer,
)

example_task = SearchRequest(
    researcher_id="r-id",
    tags=["test"],
)

reply = SearchReply(
    researcher_id="researcher-id",
    node_id="node-id",
    node_name="node-name",
    databases=[],
    count=0,
)

overlay_message = OverlayMessage(
    researcher_id="test-id",
    node_id="node-id",
    dest_node_id="node-id-1",
    overlay=b"sss",
    setup=False,
    salt=b"dummy",
    nonce=b"any nonce",
)


def _ssl_mock(key=b"test", cert=b"test", mtls=False):
    ssl_credentials = MagicMock()
    type(ssl_credentials).private_key = PropertyMock(return_value=key)
    type(ssl_credentials).certificate = PropertyMock(return_value=cert)
    type(ssl_credentials).mtls = PropertyMock(return_value=mtls)
    return ssl_credentials


# -----------------------------------------------------------------------------
# ResearcherServicer
# -----------------------------------------------------------------------------


@pytest.fixture
def servicer_env():
    context = MagicMock()
    # No client certificate presented (mutual TLS disabled)
    context.auth_context.return_value = {}

    agent_store = MagicMock(spec=AgentStore)
    on_message = MagicMock()

    return SimpleNamespace(
        request=TaskRequest(node="node-1", protocol_version="x"),
        context=context,
        agent_store=agent_store,
        on_message=on_message,
        servicer=ResearcherServicer(agent_store=agent_store, on_message=on_message),
    )


@pytest.mark.asyncio
async def test_researcher_servicer_GetTaskUnary(servicer_env):
    node_agent = AsyncMock()
    node_agent.task_done = MagicMock()
    node_agent.get_task.return_value = [example_task, 0, time.time()]

    servicer_env.agent_store.retrieve.return_value = node_agent
    with patch("fedbiomed.transport.server.logger.debug") as logger_debug:
        async for r in servicer_env.servicer.GetTaskUnary(
            request=servicer_env.request, context=servicer_env.context
        ):
            assert r.iteration == 1
            assert r.size == 1

    assert any(
        "[WIRE][S->N][TX]" in call.args[0] for call in logger_debug.call_args_list
    )


@pytest.mark.asyncio
@patch("fedbiomed.transport.server.Serializer.loads")
async def test_researcher_servicer_ReplyTask(load, servicer_env):
    # Creates async iterator
    async def request_iterator():
        for i in [1, 2]:
            yield TaskResult(size=2, iteration=i, bytes_=b"test")

    node_agent = AsyncMock()
    servicer_env.agent_store.get.return_value = node_agent
    load.return_value = reply.to_dict()
    result = await servicer_env.servicer.ReplyTask(
        request_iterator=request_iterator(), unused_context=servicer_env.context
    )
    node_agent.on_reply.assert_called_once_with(load.return_value)
    assert result == Empty()


@pytest.mark.asyncio
async def test_researcher_servicer_Feedback(servicer_env):
    request = FeedbackMessage(
        researcher_id="test",
        log=FeedbackMessage.Log(node_id="test", level="DEBUG", msg="Error message"),
    )

    with patch("fedbiomed.transport.server.logger.debug") as logger_debug:
        result = await servicer_env.servicer.Feedback(
            request=request, unused_context=servicer_env.context
        )
    servicer_env.on_message.assert_called_once()
    assert result == Empty()
    assert any(
        "[WIRE][N->S][RX]" in call.args[0] for call in logger_debug.call_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [GeneratorExit, asyncio.CancelledError])
@patch("fedbiomed.transport.server.TaskResponse")
async def test_researcher_servicer_GetTaskUnary_exceptions(
    task_response, servicer_env, exception
):
    node_agent = AsyncMock()
    node_agent.task_done = MagicMock()
    node_agent.send_async = AsyncMock()
    servicer_env.agent_store.retrieve.return_value = node_agent

    node_agent.get_task.return_value = [example_task, 0, time.time()]
    task_response.side_effect = exception

    async for r in servicer_env.servicer.GetTaskUnary(
        request=servicer_env.request, context=servicer_env.context
    ):
        assert r is None
    node_agent.send_async.assert_called_once()


# -----------------------------------------------------------------------------
# _GrpcAsyncServer
# -----------------------------------------------------------------------------


@pytest.fixture
def async_server_env():
    with (
        patch("fedbiomed.transport.server.grpc.aio.server") as server_mock,
        patch("fedbiomed.transport.server.NodeAgent", autospec=True),
        patch("fedbiomed.transport.server.AgentStore", autospec=True) as agent_store,
    ):
        server_mock.return_value.start = AsyncMock()
        server_mock.return_value.wait_for_termination = AsyncMock()

        config_mock = MagicMock(spec=ResearcherConfig)
        config_mock.getint.return_value = 10

        on_message = MagicMock()
        yield SimpleNamespace(
            server=_GrpcAsyncServer(
                host="localhost",
                port="50051",
                ssl=_ssl_mock(),
                config=config_mock,
                on_message=on_message,
                debug=False,
            ),
            server_mock=server_mock,
            agent_store_mock=agent_store,
            config_mock=config_mock,
            on_message=on_message,
        )


def _mtls_server(bundle, env):
    ssl_credentials = _ssl_mock(key=b"key", cert=b"cert", mtls=True)
    ssl_credentials.trusted_node_certificates.return_value = bundle
    return _GrpcAsyncServer(
        host="localhost",
        port="50051",
        ssl=ssl_credentials,
        config=env.config_mock,
        on_message=env.on_message,
        debug=False,
    )


@pytest.mark.asyncio
async def test_grpc_async_server_start(async_server_env):
    await async_server_env.server.start()
    async_server_env.server_mock.return_value.start.assert_called_once()
    async_server_env.server_mock.return_value.wait_for_termination.assert_called_once()


@pytest.mark.asyncio
async def test_grpc_async_server_server_auth_only_credentials(async_server_env):
    # mtls disabled -> server-auth only, no client cert required
    with patch("fedbiomed.transport.server.grpc.ssl_server_credentials") as credentials:
        await async_server_env.server.start()
    credentials.assert_called_once_with(((b"test", b"test"),))


@pytest.mark.asyncio
async def test_grpc_async_server_mtls_requires_client_auth(async_server_env):
    server = _mtls_server(b"bundle", async_server_env)

    with (
        patch(
            "fedbiomed.transport.server.grpc.dynamic_ssl_server_credentials"
        ) as credentials,
        patch(
            "fedbiomed.transport.server.grpc.ssl_server_certificate_configuration"
        ) as cert_config,
    ):
        server._server_credentials()

    _, kwargs = credentials.call_args
    assert kwargs["require_client_authentication"]
    cert_config.assert_called_with(((b"key", b"cert"),), root_certificates=b"bundle")


@pytest.mark.asyncio
@pytest.mark.parametrize("bundle", [b"", None])
async def test_grpc_async_server_mtls_empty_bundle_raises(async_server_env, bundle):
    # An empty bundle cannot bind in gRPC, so it is reported before starting
    server = _mtls_server(bundle, async_server_env)
    with pytest.raises(FedbiomedCommunicationError):
        server._server_credentials()


@pytest.mark.asyncio
async def test_grpc_async_server_send(async_server_env):
    agent = AsyncMock(spec=NodeAgent)
    async_server_env.agent_store_mock.return_value.get.return_value = agent
    await async_server_env.server.start()
    await async_server_env.server.send(example_task, "node-id")
    agent.send_async.assert_called_once()

    # An unknown node id sends nothing
    agent.send_async.reset_mock()
    async_server_env.agent_store_mock.return_value.get.return_value = None
    await async_server_env.server.send(example_task, "node-id")
    agent.send_async.assert_not_called()


@pytest.mark.asyncio
async def test_grpc_async_server_broadcast(async_server_env):
    agents = {
        "node-1": AsyncMock(spec=NodeAgent),
        "node-2": AsyncMock(spec=NodeAgent),
    }
    async_server_env.agent_store_mock.return_value.get_all.return_value = agents
    await async_server_env.server.start()
    await async_server_env.server.broadcast(example_task)
    agents["node-1"].send_async.assert_called_once()
    agents["node-2"].send_async.assert_called_once()


@pytest.mark.asyncio
async def test_grpc_async_server_get_all_nodes(async_server_env):
    loop = asyncio.get_event_loop()

    agents = {
        "node-1": NodeAgent("node-1", loop, None, 10),
        "node-2": NodeAgent("node-2", loop, None, 10),
    }
    async_server_env.agent_store_mock.return_value.get_all.return_value = agents
    agents["node-1"]._status = NodeActiveStatus.DISCONNECTED
    agents["node-2"]._status = NodeActiveStatus.ACTIVE

    await async_server_env.server.start()
    nodes = await async_server_env.server.get_all_nodes()

    assert nodes[0]._status == NodeActiveStatus.DISCONNECTED
    assert nodes[1]._status == NodeActiveStatus.ACTIVE


@pytest.mark.asyncio
async def test_grpc_async_server_on_forward(async_server_env):
    agent = AsyncMock(spec=NodeAgent)
    async_server_env.agent_store_mock.return_value.get.return_value = agent
    await async_server_env.server.start()
    with patch("fedbiomed.transport.server.logger.debug") as logger_debug:
        await async_server_env.server._on_forward(overlay_message)
    agent.send_async.assert_called_once()
    debug_messages = [call.args[0] for call in logger_debug.call_args_list]
    assert any("Researcher relay forwarding overlay" in msg for msg in debug_messages)
    assert any("Researcher relay dispatching overlay" in msg for msg in debug_messages)


@pytest.mark.asyncio
async def test_grpc_async_server_get_node(async_server_env):
    agent = AsyncMock(spec=NodeAgent)
    async_server_env.agent_store_mock.return_value.get.return_value = agent
    await async_server_env.server.start()
    assert await async_server_env.server.get_node("node-id") == agent


# -----------------------------------------------------------------------------
# GrpcServer (threaded sync wrapper)
# -----------------------------------------------------------------------------


@pytest.fixture
def grpc_server_env():
    with (
        # Replace the async methods that GrpcServer's sync wrappers invoke via
        # super() with sync mocks, so they don't create un-awaited coroutines
        # when asyncio is mocked out. Patched before _GrpcAsyncServer so the
        # name still resolves to the real class the wrappers reach through
        # super().
        patch.multiple(
            "fedbiomed.transport.server._GrpcAsyncServer",
            send=MagicMock(),
            broadcast=MagicMock(),
            get_node=MagicMock(),
            get_all_nodes=MagicMock(),
        ),
        patch("fedbiomed.transport.server._GrpcAsyncServer"),
        patch("fedbiomed.transport.server.grpc.aio.server") as server_mock,
        patch("fedbiomed.transport.server.NodeAgent", autospec=True),
        patch("fedbiomed.transport.server.AgentStore", autospec=True),
    ):
        server_mock.return_value.start = AsyncMock()
        server_mock.return_value.wait_for_termination = AsyncMock()
        config_mock = MagicMock(spec=ResearcherConfig)
        config_mock.getint.return_value = 10
        yield SimpleNamespace(
            server=GrpcServer(
                host="localhost",
                port="50051",
                ssl=_ssl_mock(),
                config=config_mock,
                on_message=MagicMock(),
                debug=False,
            ),
            server_mock=server_mock,
        )


@pytest.fixture
def mocked_asyncio(grpc_server_env):
    with patch("fedbiomed.transport.server.asyncio") as asyncio_mock:
        yield asyncio_mock


@patch("fedbiomed.transport.server.GrpcServer.get_all_nodes")
def test_grpc_server_start(get_all_nodes, grpc_server_env):
    env = grpc_server_env
    server = env.server

    with patch("fedbiomed.transport.server.MAX_GRPC_SERVER_SETUP_TIMEOUT", 2):
        server.start()

    env.server_mock.return_value.start.assert_called_once()
    env.server_mock.return_value.wait_for_termination.assert_called_once()
    server._thread.join()

    env.server_mock.return_value.start.reset_mock()
    env.server_mock.return_value.wait_for_termination.reset_mock()
    server._debug = True

    get_all_nodes.side_effect = [[], [1, 2]]
    server.start()

    env.server_mock.return_value.start.assert_called_once()
    env.server_mock.return_value.wait_for_termination.assert_called_once()

    server._thread.join()


def test_grpc_server_send(grpc_server_env, mocked_asyncio):
    server = grpc_server_env.server

    # Invalid message
    with pytest.raises(FedbiomedCommunicationError):
        server.send(message="oops", node_id="node-1")

    # Started is unset
    with pytest.raises(FedbiomedCommunicationError):
        server.send(message=example_task, node_id="node-1")

    server._is_started.set()
    server.send(message=example_task, node_id="node-1")
    mocked_asyncio.run_coroutine_threadsafe.assert_called_once()


def test_grpc_server_broadcast(grpc_server_env, mocked_asyncio):
    server = grpc_server_env.server

    # Invalid message
    with pytest.raises(FedbiomedCommunicationError):
        server.broadcast(message="oops")

    # Started is unset
    with pytest.raises(FedbiomedCommunicationError):
        server.broadcast(message=example_task)

    server._is_started.set()
    server.broadcast(message=example_task)
    mocked_asyncio.run_coroutine_threadsafe.assert_called_once()


def test_grpc_server_get_all_nodes(grpc_server_env, mocked_asyncio):
    server = grpc_server_env.server

    # Started is unset
    with pytest.raises(FedbiomedCommunicationError):
        server.get_all_nodes()

    with pytest.raises(FedbiomedCommunicationError):
        server.get_node("node-id")

    server._is_started.set()
    mocked_asyncio.run_coroutine_threadsafe.return_value.result.return_value = "test"
    assert server.get_all_nodes() == "test"

    mocked_asyncio.run_coroutine_threadsafe.return_value.result.return_value = "test2"
    assert server.get_node("node-id") == "test2"


def test_grpc_server_is_alive(grpc_server_env, mocked_asyncio):
    server = grpc_server_env.server

    # Started is unset
    with pytest.raises(FedbiomedCommunicationError):
        server.is_alive()

    server._is_started.set()
    assert server.is_alive() is False

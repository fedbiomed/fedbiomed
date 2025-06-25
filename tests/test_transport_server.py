import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

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


class TestResearcherServicer(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.request = TaskRequest(node="node-1", protocol_version="x")

        self.context = MagicMock()

        self.agent_store = MagicMock(spec=AgentStore)
        self.on_message = MagicMock()

        self.servicer = ResearcherServicer(
            agent_store=self.agent_store,
            on_message=self.on_message,
        )
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    async def test_researcher_servicer_01_GetTaskUnary(self):
        node_agent = AsyncMock()
        node_agent.task_done = MagicMock()
        node_agent.get_task.return_value = [example_task, 0, time.time()]

        self.agent_store.retrieve.return_value = node_agent
        async for r in self.servicer.GetTaskUnary(
            request=self.request, context=self.context
        ):
            self.assertEqual(r.iteration, 1)
            self.assertEqual(r.size, 1)

    @patch("fedbiomed.transport.server.Serializer.loads")
    async def test_researcher_servicer_02_ReplyTask(self, load):
        # Creates async iterator
        async def request_iterator():
            for i in [1, 2]:
                yield TaskResult(size=2, iteration=i, bytes_=b"test")

        load.return_value = {"node_id": "test-node"}
        result = await self.servicer.ReplyTask(
            request_iterator=request_iterator(), unused_context=self.context
        )
        self.assertEqual(result, Empty())

    async def test_researcher_servicer_03_Feedback(self):
        request = FeedbackMessage(
            researcher_id="test",
            log=FeedbackMessage.Log(node_id="test", level="DEBUG", msg="Error message"),
        )

        result = await self.servicer.Feedback(
            request=request, unused_context=self.context
        )
        self.on_message.assert_called_once()
        self.assertEqual(result, Empty())

    @patch("fedbiomed.transport.server.TaskResponse")
    async def test_researcher_servicer_04_GetTaskUnary_exceptions(self, task_response):
        for exception in [GeneratorExit, asyncio.CancelledError]:
            node_agent = AsyncMock()
            node_agent.task_done = MagicMock()
            node_agent.send_async = AsyncMock()
            self.agent_store.retrieve.return_value = node_agent

            node_agent.get_task.return_value = [example_task, 0, time.time()]
            task_response.side_effect = exception

            async for r in self.servicer.GetTaskUnary(
                request=self.request, context=self.context
            ):
                self.assertEqual(r, None)
            node_agent.send_async.assert_called_once()


class TestGrpcAsyncServer(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        ssl_credentials = MagicMock()
        type(ssl_credentials).private_key = PropertyMock(return_value=b"test")
        type(ssl_credentials).certificate = PropertyMock(return_value=b"test")

        self.server_patch = patch("fedbiomed.transport.server.grpc.aio.server")
        self.node_agent_patch = patch(
            "fedbiomed.transport.server.NodeAgent", autospec=True
        )
        self.agent_store_patch = patch(
            "fedbiomed.transport.server.AgentStore", autospec=True
        )

        self.server_mock = self.server_patch.start()
        self.node_agent_mock = self.node_agent_patch.start()
        self.agent_store_mock = self.agent_store_patch.start()

        self.server_mock.return_value.start = AsyncMock()
        self.server_mock.return_value.wait_for_termination = AsyncMock()

        self.config_mock = MagicMock(spec=ResearcherConfig)
        self.config_mock.getint.return_value = 10

        self.on_message = MagicMock()
        self.grpc_server = _GrpcAsyncServer(
            host="localhost",
            port="50051",
            ssl=ssl_credentials,
            config=self.config_mock,
            on_message=self.on_message,
            debug=False,
        )

        return super().setUp()

    def tearDown(self) -> None:
        self.server_patch.stop()
        self.node_agent_patch.stop()
        self.agent_store_patch.stop()

        return super().tearDown()

    async def test_grpc_async_server_01_start(self):
        await self.grpc_server.start()
        self.server_mock.return_value.start.assert_called_once()
        self.server_mock.return_value.wait_for_termination.assert_called_once()

    async def test_grpc_async_server_02_send(self):
        agent = AsyncMock(spec=NodeAgent)
        self.agent_store_mock.return_value.get.return_value = agent
        await self.grpc_server.start()
        await self.grpc_server.send(example_task, "node-id")
        agent.send_async.assert_called_once()

        agent.send_async.reset_mock()
        self.agent_store_mock.return_value.get.return_value = None
        await self.grpc_server.start()
        await self.grpc_server.send(example_task, "node-id")
        agent.send_async.assert_not_called()

    async def test_grpc_async_server_03_broadcast(self):
        agents = {
            "node-1": AsyncMock(spec=NodeAgent),
            "node-2": AsyncMock(spec=NodeAgent),
        }
        self.agent_store_mock.return_value.get_all.return_value = agents
        await self.grpc_server.start()
        await self.grpc_server.broadcast(example_task)
        agents["node-2"].send_async.assert_called_once()
        agents["node-2"].send_async.assert_called_once()

    async def test_grpc_async_server_04_get_all_nodes(self):
        loop = asyncio.get_event_loop()

        agents = {
            "node-1": NodeAgent("node-1", loop, None, 10),
            "node-2": NodeAgent("node-2", loop, None, 10),
        }
        self.agent_store_mock.return_value.get_all.return_value = agents
        agents["node-1"]._status = NodeActiveStatus.DISCONNECTED
        agents["node-2"]._status = NodeActiveStatus.ACTIVE

        await self.grpc_server.start()
        nodes = await self.grpc_server.get_all_nodes()

        self.assertEqual(nodes[0]._status, NodeActiveStatus.DISCONNECTED)
        self.assertEqual(nodes[1]._status, NodeActiveStatus.ACTIVE)

    async def test_grpc_async_server_05_on_forward(self):
        agent = AsyncMock(spec=NodeAgent)
        self.agent_store_mock.return_value.get.return_value = agent
        await self.grpc_server.start()
        await self.grpc_server._on_forward(overlay_message)
        agent.send_async.assert_called_once()

    async def test_grpc_async_server_06_get_node(self):
        agent = AsyncMock(spec=NodeAgent)
        self.agent_store_mock.return_value.get.return_value = agent
        await self.grpc_server.start()
        test_node_get = await self.grpc_server.get_node("node-id")
        self.assertEqual(test_node_get, agent)


class TestGrpcServer(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.ssl_credentials = MagicMock()
        type(self.ssl_credentials).private_key = PropertyMock(return_value=b"test")
        type(self.ssl_credentials).certificate = PropertyMock(return_value=b"test")

        self.async_server_patch = patch("fedbiomed.transport.server._GrpcAsyncServer")
        self.server_patch = patch("fedbiomed.transport.server.grpc.aio.server")
        self.node_agent_patch = patch(
            "fedbiomed.transport.server.NodeAgent", autospec=True
        )
        self.agent_store_patch = patch(
            "fedbiomed.transport.server.AgentStore", autospec=True
        )
        self.asyncio_patch = patch("fedbiomed.transport.server.asyncio")

        self.async_server = self.async_server_patch.start()

        self.server_mock = self.server_patch.start()
        self.node_agent_mock = self.node_agent_patch.start()
        self.agent_store_mock = self.agent_store_patch.start()
        self.asyncio_mock = self.asyncio_patch.start()

        self.server_mock.return_value.start = AsyncMock()
        self.server_mock.return_value.wait_for_termination = AsyncMock()
        self.config_mock = MagicMock(spec=ResearcherConfig)
        self.config_mock.getint.return_value = 10
        self.on_message = MagicMock()
        self.grpc_server = GrpcServer(
            host="localhost",
            port="50051",
            ssl=self.ssl_credentials,
            config=self.config_mock,
            on_message=self.on_message,
            debug=False,
        )
        return super().setUp()

    def tearDown(self) -> None:
        self.server_patch.stop()
        self.node_agent_patch.stop()
        self.agent_store_patch.stop()
        self.asyncio_patch.stop()
        self.async_server_patch.stop()

        return super().tearDown()

    @patch("fedbiomed.transport.server.GrpcServer.get_all_nodes")
    def test_grpc_server_01_start(self, get_all_nodes):
        self.grpc_server = GrpcServer(
            host="localhost",
            port="50051",
            ssl=self.ssl_credentials,
            config=self.config_mock,
            on_message=self.on_message,
            debug=False,
        )

        self.asyncio_patch.stop()

        with patch("fedbiomed.transport.server.MAX_GRPC_SERVER_SETUP_TIMEOUT", 2):
            self.grpc_server.start()

        self.server_mock.return_value.start.assert_called_once()
        self.server_mock.return_value.wait_for_termination.assert_called_once()
        self.grpc_server._thread.join()

        self.server_mock.return_value.start.reset_mock()
        self.server_mock.return_value.wait_for_termination.reset_mock()
        self.grpc_server._debug = True

        get_all_nodes.side_effect = [[], [1, 2]]
        self.grpc_server.start()

        self.server_mock.return_value.start.assert_called_once()
        self.server_mock.return_value.wait_for_termination.assert_called_once()

        self.grpc_server._thread.join()

    def test_grpc_server_02_send(self):
        # Invalid message
        with self.assertRaises(FedbiomedCommunicationError):
            self.grpc_server.send(message="opps", node_id="node-1")

        # Started is unset
        with self.assertRaises(FedbiomedCommunicationError):
            self.grpc_server.send(message=example_task, node_id="node-1")

        self.grpc_server._is_started.set()
        self.grpc_server.send(message=example_task, node_id="node-1")
        self.asyncio_mock.run_coroutine_threadsafe.assert_called_once()

    def test_grpc_server_03_broadcast(self):
        # Invalid message
        with self.assertRaises(FedbiomedCommunicationError):
            self.grpc_server.broadcast(message="opps")

        # Started is unset
        with self.assertRaises(FedbiomedCommunicationError):
            self.grpc_server.broadcast(message=example_task)

        self.grpc_server._is_started.set()
        self.grpc_server.send(message=example_task, node_id="node-1")
        self.asyncio_mock.run_coroutine_threadsafe.assert_called_once()

    def test_grpc_server_04_get_all_nodes(self):
        # Started is unset
        with self.assertRaises(FedbiomedCommunicationError):
            self.grpc_server.get_all_nodes()

        with self.assertRaises(FedbiomedCommunicationError):
            self.grpc_server.get_node("node-id")

        self.grpc_server._is_started.set()
        self.asyncio_mock.run_coroutine_threadsafe.return_value.result.return_value = (
            "test"
        )
        result = self.grpc_server.get_all_nodes()
        self.assertEqual(result, "test")

        self.asyncio_mock.run_coroutine_threadsafe.return_value.result.return_value = (
            "test2"
        )
        result = self.grpc_server.get_node("node-id")
        self.assertEqual(result, "test2")

    def test_grpc_server_05_is_alive(self):
        # Started is unset
        with self.assertRaises(FedbiomedCommunicationError):
            self.grpc_server.is_alive()

        self.grpc_server._is_started.set()
        result = self.grpc_server.is_alive()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

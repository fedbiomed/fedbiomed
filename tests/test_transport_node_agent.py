import asyncio
import unittest
from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.message import OverlayMessage, SearchRequest
from fedbiomed.transport.node_agent import (
    AgentStore,
    NodeActiveStatus,
    NodeAgent,
    NodeAgentAsync,
)

message = MagicMock(spec=SearchRequest)


@pytest.fixture
def node_agent():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    fixture = TestNodeAgent()
    fixture.setUp()

    try:
        yield fixture
    finally:
        fixture.tearDown()
        loop.close()
        asyncio.set_event_loop(None)


class TestNodeAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop_policy().get_event_loop()
        self.node_agent = NodeAgent(
            id="node-1", loop=self.loop, on_forward=None, node_disconnection_timeout=10
        )

    def tearDown(self) -> None:
        return super().tearDown()

    def test_node_agent_01_id(self):
        id = self.node_agent.id
        self.assertEqual(id, "node-1")

    async def test_node_agent_02_status(self):
        status = await self.node_agent.status_async()
        self.assertEqual(status, NodeActiveStatus.ACTIVE)

    async def test_node_agent_03_set_active(self):
        status_task = MagicMock()

        self.node_agent._status_task = status_task
        self.node_agent._status = NodeActiveStatus.DISCONNECTED
        await self.node_agent.set_active()
        status_task.cancel.assert_called_once()

    async def test_node_agent_04_send(self):
        self.node_agent._status = NodeActiveStatus.DISCONNECTED
        r = await self.node_agent.send_async(message=message)
        self.assertIsNone(r)

        self.node_agent._status = NodeActiveStatus.WAITING
        r = await self.node_agent.send_async(message=message)
        item = await self.node_agent._queue.get()
        self.assertEqual(item[0], message)

        self.node_agent._status = NodeActiveStatus.ACTIVE
        r = await self.node_agent.send_async(message=message)
        item = await self.node_agent._queue.get()
        self.assertEqual(item[0], message)

        with patch("fedbiomed.transport.node_agent.asyncio.Queue.put") as put:
            put.side_effect = Exception
            with self.assertRaises(Exception):  # noqa: B017
                await self.node_agent.send_async(message=message)

    async def test_node_agent_06_get_task(self):
        await self.node_agent._queue.put(message)
        r = await self.node_agent.get_task()
        self.assertEqual(r, message)

    async def test_node_agent_07_task_done(self):
        with patch(
            "fedbiomed.transport.node_agent.asyncio.Queue.task_done"
        ) as task_done:
            self.node_agent.task_done()
            task_done.assert_called_once()

    async def test_node_agent_09_change_node_status_after_task(self):
        """Tests two methods

        - change_node_status_after_task
        - _change_node_status_disconnected

        """
        with patch("fedbiomed.transport.node_agent.asyncio.sleep") as _:
            await self.node_agent.change_node_status_after_task()
            await self.node_agent._status_task
            self.assertEqual(self.node_agent._status, NodeActiveStatus.DISCONNECTED)


class TestAgentStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop_policy().get_event_loop()
        self.agent_store = AgentStore(
            loop=self.loop, on_forward=None, node_disconnection_timeout=10
        )

    async def test_agent_store_01_retrieve(self):
        node_agent = await self.agent_store.retrieve(node_id="node-id")
        self.assertTrue("node-id" in self.agent_store._node_agents)
        self.assertIsInstance(node_agent, NodeAgent)

    async def test_agent_store_02_get_all(self):
        # Register node agents
        await self.agent_store.retrieve(node_id="node-id-1")
        await self.agent_store.retrieve(node_id="node-id-2")

        all_ = await self.agent_store.get_all()

        # Try to update
        all_["node-id-2"] = "opps"
        self.assertFalse(self.agent_store._node_agents["node-id-2"] == "opps")

    async def test_agent_store_03_get(self):
        # Register node agents
        await self.agent_store.retrieve(node_id="node-id-1")
        await self.agent_store.retrieve(node_id="node-id-2")

        result = await self.agent_store.get("node-id-1")
        self.assertEqual(result.id, "node-id-1")


def test_node_agent_flush_marks_stopped_request(node_agent):
    node_agent.node_agent._replies["req-1"] = {
        "reply": None,
        "callback": lambda msg: None,
    }

    node_agent.loop.run_until_complete(
        NodeAgentAsync.flush(node_agent.node_agent, "req-1", stopped=True)
    )

    assert "req-1" in node_agent.node_agent._replies
    assert "req-1" in node_agent.node_agent._stopped_request_ids


def test_node_agent_on_reply_pending_request(monkeypatch, node_agent):
    class DummyReply:
        request_id = "req-2"

    seen = {"reply": None}

    def callback(reply):
        seen["reply"] = reply

    node_agent.node_agent._replies["req-2"] = {"reply": None, "callback": callback}

    monkeypatch.setattr(
        "fedbiomed.transport.node_agent.Message.from_dict",
        lambda _: DummyReply(),
    )

    node_agent.loop.run_until_complete(node_agent.node_agent.on_reply({}))

    assert node_agent.node_agent._replies["req-2"]["reply"].request_id == "req-2"
    assert seen["reply"].request_id == "req-2"


def test_node_agent_on_reply_overlay_message(monkeypatch, node_agent):
    seen = {"reply": None}

    def callback(reply):
        seen["reply"] = reply

    overlay = object.__new__(OverlayMessage)
    overlay.request_id = "req-overlay"

    node_agent.node_agent._replies["req-overlay"] = {
        "reply": None,
        "callback": callback,
    }

    monkeypatch.setattr(
        "fedbiomed.transport.node_agent.Message.from_dict",
        lambda _: overlay,
    )

    node_agent.loop.run_until_complete(node_agent.node_agent.on_reply({}))

    assert node_agent.node_agent._replies["req-overlay"]["reply"] is overlay
    assert seen["reply"] is overlay


def test_node_agent_on_reply_multiple_reply_warning(monkeypatch, node_agent):
    """Covers the branch where a second reply arrives for the same request."""

    class DummyReply:
        request_id = "req-dup"

    warnings = {"count": 0}

    def fake_warning(*args, **kwargs):
        warnings["count"] += 1

    node_agent.node_agent._replies["req-dup"] = {
        "reply": object(),
        "callback": lambda msg: None,
    }

    monkeypatch.setattr(
        "fedbiomed.transport.node_agent.Message.from_dict",
        lambda _: DummyReply(),
    )
    monkeypatch.setattr("fedbiomed.transport.node_agent.logger.warning", fake_warning)

    node_agent.loop.run_until_complete(node_agent.node_agent.on_reply({}))

    assert warnings["count"] == 1


def test_node_agent_on_reply_none_request_id_unexpected_warning(
    monkeypatch, node_agent
):
    """In this implementation, request_id=None does not log error.
    It falls through to the unexpected-request warning branch.
    """

    class DummyReply:
        request_id = None

    warnings = {"count": 0}

    def fake_warning(*args, **kwargs):
        warnings["count"] += 1

    monkeypatch.setattr(
        "fedbiomed.transport.node_agent.Message.from_dict",
        lambda _: DummyReply(),
    )
    monkeypatch.setattr("fedbiomed.transport.node_agent.logger.warning", fake_warning)

    node_agent.loop.run_until_complete(node_agent.node_agent.on_reply({}))

    assert warnings["count"] == 1


def test_node_agent_on_reply_stopped_request(monkeypatch, node_agent):
    class DummyReply:
        request_id = "req-3"

    node_agent.node_agent._stopped_request_ids.append("req-3")

    monkeypatch.setattr(
        "fedbiomed.transport.node_agent.Message.from_dict",
        lambda _: DummyReply(),
    )

    node_agent.loop.run_until_complete(node_agent.node_agent.on_reply({}))

    assert "req-3" not in node_agent.node_agent._stopped_request_ids


def test_node_agent_on_reply_unexpected_request(monkeypatch, node_agent):
    class DummyReply:
        request_id = "req-404"

    warnings = {"count": 0}

    def fake_warning(*args, **kwargs):
        warnings["count"] += 1

    monkeypatch.setattr(
        "fedbiomed.transport.node_agent.Message.from_dict",
        lambda _: DummyReply(),
    )
    monkeypatch.setattr("fedbiomed.transport.node_agent.logger.warning", fake_warning)

    node_agent.loop.run_until_complete(node_agent.node_agent.on_reply({}))

    assert warnings["count"] >= 1


if __name__ == "__main__":
    unittest.main()

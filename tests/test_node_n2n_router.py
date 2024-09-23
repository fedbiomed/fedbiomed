import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import time

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase

#############################################################

from fedbiomed.common.message import OverlayMessage, KeyReply
from fedbiomed.node.environ import environ
from fedbiomed.node.requests._n2n_router import NodeToNodeRouter, _NodeToNodeAsyncRouter



class DummyException(Exception):
    pass


class TestNodeToNodeAsyncRouter(unittest.IsolatedAsyncioTestCase, NodeTestCase):
    """Test for node2node router module, _NodeToNodeAsyncRouter class"""

    def setUp(self):
        self.async_queue_patch = patch(
            "fedbiomed.node.requests._n2n_router.asyncio.Queue", autospec=True
        )
        self.async_queue_patch.side_effect = AsyncMock()
        self.n2n_controller_patch = patch(
            "fedbiomed.node.requests._n2n_router.NodeToNodeController", autospec=True
        )
        self.n2n_controller_patch.side_effect = AsyncMock()
        self.format_overlay_channel_patch = patch(
            'fedbiomed.node.requests._n2n_router.OverlayChannel', autospec=True
        )

        self.async_queue_patcher = self.async_queue_patch.start()
        self.n2n_controller_patcher = self.n2n_controller_patch.start()
        self.format_overlay_channel_patcher = self.format_overlay_channel_patch.start()

        self.grpc_controller_mock = MagicMock(autospec=True)
        self.pending_requests_mock = MagicMock(autospec=True)
        self.controller_data_mock = MagicMock(autospec=True)

        self.n2n_async_router = _NodeToNodeAsyncRouter(
            self.grpc_controller_mock,
            self.pending_requests_mock,
            self.controller_data_mock,
        )

    def tearDown(self):
        self.async_queue_patch.stop()
        self.n2n_controller_patch.stop()
        self.format_overlay_channel_patch.stop()

    async def test_n2n_async_router_01_remove_finished_tasks(self):
        """Remove finished tasks of a n2n async router"""

        # prepare
        async def dummy_task(*args, **kwargs):
            pass

        t1 = asyncio.create_task(dummy_task())
        t2 = asyncio.create_task(dummy_task())
        t3 = asyncio.create_task(dummy_task())

        # need to access private list of active tasks
        active_tasks = self.n2n_async_router._active_tasks
        # will not timeout
        active_tasks[t1.get_name()] = {
            "start_time": time.time(),
            "task": t1,
            "finally": False,
        }
        active_tasks[t2.get_name()] = {
            "start_time": time.time(),
            "task": t2,
            "finally": False,
        }

        # 1. remove task existing in active tasks

        # action
        self.n2n_async_router._remove_finished_task(t1)
        await asyncio.sleep(0.1)  # give time to complete task

        self.assertEqual(set(active_tasks.keys()), set([t2.get_name()]))

        # 1. remove task not existing in active tasks

        # action
        self.n2n_async_router._remove_finished_task(t3)
        await asyncio.sleep(0.1)  # give time to complete task

        self.assertEqual(set(active_tasks.keys()), set([t2.get_name()]))

    @patch("fedbiomed.node.requests._n2n_router.asyncio.sleep")
    @patch("fedbiomed.node.requests._n2n_router.OVERLAY_MESSAGE_PROCESS_TIMEOUT", 0.5)
    async def test_n2n_async_router_02_clean_active_tasks(
        self,
        asyncio_sleep,
    ):
        """Clean active tasks of a n2n async router"""
        # prepare
        asyncio_sleep.side_effect = [None, DummyException]

        async def dummy_task(*args, **kwargs):
            pass

        t1 = asyncio.create_task(dummy_task())
        t2 = asyncio.create_task(dummy_task())
        start_time = time.time()

        # need to access private list of active tasks
        active_tasks = self.n2n_async_router._active_tasks
        # will not timeout
        active_tasks[t1.get_name()] = {
            "start_time": start_time,
            "task": t1,
            "finally": False,
        }
        # will timeout
        active_tasks[t2.get_name()] = {
            "start_time": start_time - 1,
            "task": t2,
            "finally": False,
        }

        # action
        with self.assertRaises(DummyException):
            await self.n2n_async_router._clean_active_tasks()

        # checks
        self.assertEqual(active_tasks[t1.get_name()]["finally"], False)
        self.assertEqual(active_tasks[t2.get_name()]["finally"], True)

    @patch(
        "fedbiomed.node.requests._n2n_router._NodeToNodeAsyncRouter._overlay_message_process"
    )
    @patch(
        "fedbiomed.node.requests._n2n_router._NodeToNodeAsyncRouter._remove_finished_task"
    )
    @patch(
        "fedbiomed.node.requests._n2n_router._NodeToNodeAsyncRouter._clean_active_tasks"
    )
    async def test_n2n_async_router_03_run_async(
        self, clean_active_tasks, remove_finished_task, overlay_message_process
    ):
        """Run async of a n2n async router"""

        # prepare
        async def dummy_task(*args, **kwargs):
            pass

        clean_active_tasks.return_value = asyncio.create_task(dummy_task())
        remove_finished_task.return_value = asyncio.create_task(dummy_task())
        overlay_message_process.return_value = asyncio.create_task(dummy_task())

        message = {
            "dest_node_id": environ["NODE_ID"],
            "overlay": "dummy content",
        }
        self.async_queue_patcher.return_value.get.side_effect = [
            message,
            DummyException,
        ]

        # action
        with self.assertRaises(DummyException):
            await self.n2n_async_router._run_async()

        # check
        clean_active_tasks.assert_called_once()
        overlay_message_process.assert_called_once()

    async def test_n2n_async_router_04_submit(self):
        """Submit message to a n2n async router"""
        # successful
        await self.n2n_async_router._submit("dummy_message")
        self.async_queue_patcher.return_value.put_nowait.assert_called_once()
        self.async_queue_patcher.return_value.put_nowait.reset_mock()

        # failed
        self.async_queue_patcher.return_value.put_nowait.side_effect = asyncio.QueueFull
        await self.n2n_async_router._submit("dummy_messagfe")
        self.async_queue_patcher.return_value.put_nowait.assert_called_once()
        self.async_queue_patcher.return_value.put_nowait.reset_mock()

    async def test_n2n_async_router_05_overlay_process(self):
        """Process overlay message in n2n async router"""
        #
        # 1. successful call
        #

        # prepare
        self.format_overlay_channel_patcher.return_value.format_incoming_overlay.return_value = KeyReply(
            request_id="request",
            dest_node_id=environ["ID"],
            node_id="test",
            public_key=b"test",
            secagg_id="test",
        )
        self.n2n_controller_patcher.return_value.handle.return_value = None
        self.n2n_controller_patcher.return_value.final.return_value = None

        msg = OverlayMessage(
            **{
                "node_id": "n1",
                "dest_node_id": environ["NODE_ID"],
                "overlay": [b"dummy content"],
                "researcher_id": "r1",
                'setup': False,
                'salt': b'my dummy salt',
            }
        )
        # need to initialize private variable (store current active task)
        self.n2n_async_router._active_tasks[asyncio.current_task().get_name()] = {}

        # action
        await self.n2n_async_router._overlay_message_process(msg)

        # check
        self.format_overlay_channel_patcher.return_value.format_incoming_overlay.assert_called_once()
        self.n2n_controller_patcher.return_value.handle.assert_called_once()
        self.n2n_controller_patcher.return_value.final.assert_called_once()

        # reset
        self.format_overlay_channel_patcher.reset_mock()
        self.n2n_controller_patcher.reset_mock()

        #
        # 2. bad message failure call
        #

        # prepare
        msg2 = OverlayMessage(
            **{
                "node_id": "n1",
                "dest_node_id": "incorrect node id",
                "overlay": [b"dummy content"],
                "researcher_id": "r1",
                'setup': False,
                'salt': b'my dummy salt',
            }
        )

        # action
        await self.n2n_async_router._overlay_message_process(msg2)

        # check
        self.format_overlay_channel_patcher.return_value.format_incoming_overlay.assert_not_called()
        self.n2n_controller_patcher.return_value.handle.assert_not_called()
        self.n2n_controller_patcher.return_value.final.assert_not_called()

        # reset
        self.format_overlay_channel_patcher.reset_mock()
        self.n2n_controller_patcher.reset_mock()

        #
        # 3. cancelled handling failure call
        #

        # prepare
        self.n2n_controller_patcher.return_value.handle.side_effect = (
            asyncio.CancelledError
        )

        # action
        await self.n2n_async_router._overlay_message_process(msg)

        # check
        self.format_overlay_channel_patcher.return_value.format_incoming_overlay.assert_called_once()
        self.n2n_controller_patcher.return_value.handle.assert_called_once()
        self.n2n_controller_patcher.return_value.final.assert_not_called()

        # reset
        self.format_overlay_channel_patcher.reset_mock()
        self.n2n_controller_patcher.reset_mock()

        #
        # 4. other error failure call
        #

        # prepare
        self.n2n_controller_patcher.return_value.handle.side_effect = DummyException

        # action
        await self.n2n_async_router._overlay_message_process(msg)

        # check
        self.format_overlay_channel_patcher.return_value.format_incoming_overlay.assert_called_once()
        self.n2n_controller_patcher.return_value.handle.assert_called_once()
        self.n2n_controller_patcher.return_value.final.assert_not_called()

        # reset
        self.format_overlay_channel_patcher.reset_mock()
        self.n2n_controller_patcher.reset_mock()


class TestNodeToNodeRouter(unittest.IsolatedAsyncioTestCase, NodeTestCase):
    """Test for node2node router module, NodeToNodeRouter class"""

    def setUp(self):

        self.async_patch = patch(
            "fedbiomed.node.requests._n2n_router.asyncio", autospec=True
        )
        self.thread_patch = patch(
            "fedbiomed.node.requests._n2n_router.Thread", autospec=True
        )
        self.n2n_controller_patch = patch(
            "fedbiomed.node.requests._n2n_router.NodeToNodeController", autospec=True
        )
        self.n2n_controller_patch.side_effect = AsyncMock()

        self.async_patcher = self.async_patch.start()
        self.thread_patcher = self.thread_patch.start()
        self.n2n_controller_patcher = self.n2n_controller_patch.start()

        self.grpc_controller_mock = MagicMock(autospec=True)
        self.pending_requests_mock = MagicMock(autospec=True)
        self.controller_data_mock = MagicMock(autospec=True)

        self.n2n_router = NodeToNodeRouter(
            self.grpc_controller_mock,
            self.pending_requests_mock,
            self.controller_data_mock,
        )

    def tearDown(self):

        self.async_patch.stop()
        self.thread_patch.stop()
        self.n2n_controller_patch.stop()

    def test_n2n_router_01_start(self):
        """Start a n2n router"""
        self.n2n_router.start()
        self.thread_patcher.return_value.start.assert_called_once()

    def test_n2n_router_02_private_run(self):
        """Call `_run()` from n2n router"""
        # successful run
        self.n2n_router._run()
        self.async_patcher.run.assert_called_once()

        # failed run
        self.async_patcher.run.side_effect = DummyException
        with self.assertRaises(DummyException):
            self.n2n_router._run()

    def test_n2n_router_03_submit(self):
        """Submit message to a n2n router"""
        # correct message
        message = OverlayMessage(
            node_id="test",
            dest_node_id="test",
            overlay=[b"test"],
            researcher_id="researcher",
            setup=False,
            salt=b'my dummy salt',
        )

        self.n2n_router.submit(message)
        self.async_patcher.run_coroutine_threadsafe.assert_called_once()

        # failed submission
        self.async_patcher.run_coroutine_threadsafe.side_effect = DummyException
        with self.assertRaises(DummyException):
            self.n2n_router.submit(message)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

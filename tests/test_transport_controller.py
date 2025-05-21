import unittest
import asyncio

from unittest.mock import MagicMock, patch

from fedbiomed.transport.controller import (
    GrpcAsyncTaskController,
    ResearcherCredentials,
    GrpcController,
)
from fedbiomed.common.message import SearchReply
from fedbiomed.common.exceptions import FedbiomedCommunicationError

message = SearchReply(
    researcher_id="researcher-id",
    node_id="node-id",
    databases=[],
    count=0,
)


class TestGrpcAsyncTaskController(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.r_id = "researcher-id"
        self.client_patch = patch(
            "fedbiomed.transport.controller.GrpcClient", autospec=True
        )
        self.client_mock = self.client_patch.start()

        self.on_message = MagicMock()
        self.controller = GrpcAsyncTaskController(
            node_id="test-id",
            on_message=self.on_message,
            researchers=[ResearcherCredentials(host="localhost", port="50051")],
            debug=False,
        )

        self.controller._ip_id_map = {self.r_id: "localhost:50051"}

        f = asyncio.Future()
        f.set_result(True)

        self.client_mock.return_value.start.return_value = f
        self.start = self.controller.start()

    def tearDown(self) -> None:
        self.client_patch.stop()

    async def test_grpc_async_controller_01_start(self):
        f = asyncio.Future()
        f.set_result(True)

        self.client_mock.return_value.start.return_value = f
        await self.controller.start()
        self.client_mock.return_value.start.assert_called_once()

    async def test_grpc_async_controller_02_send(self):
        await self.start
        self.client_mock.return_value.start.assert_called_once()
        await self.controller.send(message=message)
        self.client_mock.return_value.send.assert_called_once_with(message)

        # Broadcast
        self.client_mock.return_value.send.reset_mock()
        await self.controller.send(message=message, broadcast=True)
        self.client_mock.return_value.send.assert_called_once_with(message)

    async def test_grpc_async_controller_03_update_id_map(self):
        await self.start
        await self.controller._update_id_ip_map(id_=self.r_id, ip="localhost:50052")
        self.assertEqual(self.controller._ip_id_map[self.r_id], "localhost:50052")

    async def test_grpc_async_controller_04_is_connected(self):
        task = MagicMock()
        task.done.return_value = False
        type(self.client_mock.return_value).tasks = [task]
        await self.start
        self.assertTrue(await self.controller.is_connected())


class TestGrpcController(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.async_patch = patch(
            "fedbiomed.transport.controller.asyncio", autospec=True
        )
        self.thread_patch = patch(
            "fedbiomed.transport.controller.threading.Thread", autospec=True
        )
        self.send_patch = patch(
            "fedbiomed.transport.controller.GrpcAsyncTaskController.send", autospec=True
        )

        self.async_mock = self.async_patch.start()
        self.thread_mock = self.thread_patch.start()
        self.send_mock = self.send_patch.start()

        self.on_message = MagicMock()

        self.controller = GrpcController(
            node_id="test-id",
            on_message=self.on_message,
            researchers=[ResearcherCredentials(host="localhost", port="50051")],
            debug=False,
        )

    def tearDown(self) -> None:
        self.async_patch.stop()
        self.thread_patch.stop()
        self.send_patch.stop()

        return super().tearDown()

    def test_grpc_controller_01_start(self):
        self.controller.start()
        self.thread_mock.return_value.start.assert_called_once()

    def test_grpc_controller_02_private_run(self):
        on_finish = MagicMock()
        self.controller._run(on_finish=on_finish)
        self.async_mock.run.assert_called_once()

        self.async_mock.run.side_effect = Exception
        self.controller._run(on_finish=on_finish)
        on_finish.assert_called_once()

    def test_grpc_controller_03_send(self):
        invalid_message = {}
        with self.assertRaises(FedbiomedCommunicationError):
            self.controller.send(invalid_message)

        # is_started not set
        with self.assertRaises(FedbiomedCommunicationError):
            self.controller.send(message)

        # is_started is set

        self.controller._is_started.set()
        self.controller.send(message)
        self.send_mock.assert_called_once()
        self.async_mock.run_coroutine_threadsafe.assert_called_once()

    def test_grpc_controller_04_is_connected(self):
        with self.assertRaises(FedbiomedCommunicationError):
            self.controller.is_connected()

        self.controller.start()
        self.controller._is_started.set()
        self.thread_mock.return_value.is_alive.return_value = False
        self.assertFalse(self.controller.is_connected())

        self.controller.start()
        self.controller._is_started.set()
        self.thread_mock.return_value.is_alive.return_value = True
        future_mock = MagicMock()
        future_mock.result.return_value = True
        self.async_mock.run_coroutine_threadsafe.return_value = future_mock
        with patch(
            "fedbiomed.transport.controller.GrpcAsyncTaskController.is_connected"
        ) as is_connected:
            self.assertTrue(self.controller.is_connected())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

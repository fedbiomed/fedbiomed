import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
from testsupport.mock import AsyncMock

from fedbiomed.common.constants import MAX_RETRIEVE_ERROR_RETRIES, MAX_SEND_RETRIES
from fedbiomed.common.exceptions import FedbiomedCommunicationError
from fedbiomed.common.message import FeedbackMessage, Log, Scalar, SearchReply
from fedbiomed.transport.client import (
    Channels,
    ClientStatus,
    GrpcClient,
    ResearcherCredentials,
    Sender,
    TaskListener,
    _StubType,
)
from fedbiomed.transport.protocols.researcher_pb2 import TaskResponse
from fedbiomed.transport.protocols.researcher_pb2_grpc import ResearcherServiceStub


class TestGrpcClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create patches
        self.stub_patch = patch(
            "fedbiomed.transport.client.ResearcherServiceStub", autospec=True
        )
        self.sender_patch = patch("fedbiomed.transport.client.Sender", autospec=True)

        # Start patched
        self.task_listener_patch = patch(
            "fedbiomed.transport.client.TaskListener", autospec=True
        )
        self.stub_mock = self.stub_patch.start()
        self.sender_mock = self.sender_patch.start()
        self.task_listener_mock = self.task_listener_patch.start()

        self.update_id_map = AsyncMock()

        credentials = ResearcherCredentials(port="50051", host="localhost")

        self.client = GrpcClient(
            node_id="test-node-id",
            researcher=credentials,
            update_id_map=self.update_id_map,
        )

    def tearDown(self):
        self.stub_patch.stop()
        self.sender_patch.stop()
        self.task_listener_patch.stop()
        pass

    async def test_grpc_client_01_start(self):
        on_task = MagicMock()
        task = self.client.start(on_task=on_task)
        self.assertIsInstance(task, asyncio.Future)

    async def test_grpc_client_02_send(self):
        message = {"test": "test"}
        await self.client.send(message)
        self.sender_mock.return_value.send.assert_called_once_with(message)

    async def test_grpc_client_03_on_status_change(self):
        await self.client._on_status_change(ClientStatus.CONNECTED)
        self.assertEqual(self.client._status, ClientStatus.CONNECTED)

    async def test_grpc_client_05__update_id(self):
        await self.client._update_id(id_="test")
        self.assertEqual(self.client._id, "test")
        self.update_id_map.assert_called_once_with(
            f"{self.client._researcher.host}:{self.client._researcher.port}", "test"
        )

        with self.assertRaises(FedbiomedCommunicationError):
            await self.client._update_id(id_="test-malicious")

        pass


class TestTaskListener(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.serializer_patch = patch("fedbiomed.transport.client.Serializer")
        self.serializer_mock = self.serializer_patch.start()
        self.node_id = "test-node-id"
        self.on_status_change = AsyncMock()
        self.update_id = AsyncMock()
        self.callback = MagicMock()
        self.channels = MagicMock()
        self.channels.connect = AsyncMock()
        self.task_listener = TaskListener(
            channels=self.channels,
            node_id=self.node_id,
            on_status_change=self.on_status_change,
            update_id=self.update_id,
        )

    def tearDown(self) -> None:
        self.serializer_patch.stop()
        pass

    async def test_task_listener_01_listen(self):
        self.serializer_mock.load.return_value = {"researcher_id": "test-researcher-id"}

        async def async_iterator(items):
            for item in items:
                yield item

        request_stub = MagicMock()
        self.channels.stub = AsyncMock()
        self.channels.stub.return_value = request_stub
        # Run with cancel to be able to stop the loop ---------------------
        request_stub.GetTaskUnary.side_effect = [
            async_iterator(
                [
                    TaskResponse(bytes_=b"test-1", iteration=0, size=1),
                    TaskResponse(bytes_=b"test-2", iteration=1, size=1),
                ]
            ),
            asyncio.CancelledError,
        ]
        task = self.task_listener.listen(self.callback)

        with self.assertRaises(asyncio.CancelledError):
            await task

        # self.callback.assert_called_once()
        self.serializer_mock.loads.assert_called_once()
        self.assertEqual(request_stub.GetTaskUnary.call_count, 2)
        self.update_id.assert_called_once()

        # Cancel the task for next test
        task.cancel()

    @patch("fedbiomed.transport.client.asyncio.sleep")
    async def test_task_listener_02_listen_grpc_exceptions(self, sleep):
        request_stub = MagicMock()
        self.channels.stub = AsyncMock()
        self.channels.stub.return_value = request_stub

        # deadline exceeded
        request_stub.GetTaskUnary.side_effect = [
            grpc.aio.AioRpcError(
                code=grpc.StatusCode.DEADLINE_EXCEEDED,
                trailing_metadata=grpc.aio.Metadata(("test", "test")),
                initial_metadata=grpc.aio.Metadata(("test", "test")),
            ),
            asyncio.CancelledError,
        ]

        task = self.task_listener.listen(self.callback)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(request_stub.GetTaskUnary.call_count, 2)
        # Cancel and reset the task for next test
        task.cancel()
        request_stub.reset_mock()

        # unavailable
        request_stub.GetTaskUnary.side_effect = [
            grpc.aio.AioRpcError(
                code=grpc.StatusCode.UNAVAILABLE,
                trailing_metadata=grpc.aio.Metadata(("test", "test")),
                initial_metadata=grpc.aio.Metadata(("test", "test")),
            ),
            asyncio.CancelledError,
        ]

        task = self.task_listener.listen(self.callback)
        with self.assertRaises(asyncio.CancelledError):
            await task
        sleep.assert_called_once()
        self.assertEqual(request_stub.GetTaskUnary.call_count, 2)
        # Cancel and reset the task for next test
        task.cancel()
        request_stub.reset_mock()
        sleep.reset_mock()

        # unknown
        request_stub.GetTaskUnary.side_effect = [
            grpc.aio.AioRpcError(
                code=grpc.StatusCode.UNKNOWN,
                trailing_metadata=grpc.aio.Metadata(("test", "test")),
                initial_metadata=grpc.aio.Metadata(("test", "test")),
            ),
            asyncio.CancelledError,
        ]

        task = self.task_listener.listen(self.callback)
        with self.assertRaises(asyncio.CancelledError):
            await task
        sleep.assert_called_once()
        self.assertEqual(request_stub.GetTaskUnary.call_count, 2)
        # Cancel and reset the task for next test
        task.cancel()
        request_stub.reset_mock()
        sleep.reset_mock()

        # For all others gRPC errors
        request_stub.GetTaskUnary.side_effect = [
            grpc.aio.AioRpcError(
                code=grpc.StatusCode.ABORTED,
                trailing_metadata=grpc.aio.Metadata(("test", "test")),
                initial_metadata=grpc.aio.Metadata(("test", "test")),
            ),
            asyncio.CancelledError,
        ]

        task = self.task_listener.listen(self.callback)
        with self.assertRaises(asyncio.CancelledError):
            await task
        sleep.assert_called_once()
        self.assertEqual(request_stub.GetTaskUnary.call_count, 2)

        # Cancel and reset the task for next test
        task.cancel()
        request_stub.reset_mock()
        sleep.reset_mock()

    @patch("fedbiomed.transport.client.asyncio.sleep")
    async def test_task_listener_03_listen_non_grpc_exceptions(self, sleep):
        request_stub = MagicMock()
        self.channels.stub = AsyncMock()
        self.channels.stub.return_value = request_stub

        async def async_iterator(items):
            for item in items:
                yield item

        # Wrap to count calls
        self.task_listener._post_handle_raise = MagicMock(
            wraps=self.task_listener._post_handle_raise
        )

        # Test with increasing number of error until over the maximum authorized (MAX + 3 to test more cases)
        for exception in [RuntimeError, Exception, GeneratorExit]:
            for nb_errors in range(1, MAX_RETRIEVE_ERROR_RETRIES + 5):
                request_stub.GetTaskUnary.side_effect = [exception] * nb_errors + [
                    asyncio.CancelledError
                ]

                task = self.task_listener.listen(self.callback)
                if nb_errors <= MAX_RETRIEVE_ERROR_RETRIES:
                    signal = asyncio.CancelledError
                else:
                    signal = FedbiomedCommunicationError
                with self.assertRaises(signal):
                    await task
                self.assertEqual(
                    sleep.call_count, min(nb_errors, MAX_RETRIEVE_ERROR_RETRIES)
                )
                self.assertEqual(
                    request_stub.GetTaskUnary.call_count,
                    min(nb_errors + 1, MAX_RETRIEVE_ERROR_RETRIES + 1),
                )
                self.assertEqual(
                    self.task_listener._post_handle_raise.call_count,
                    max(0, nb_errors - MAX_RETRIEVE_ERROR_RETRIES),
                )

                # Cancel the task
                task.cancel()

                # Need a successful task retrieve to reset the retry counters
                request_stub.GetTaskUnary.side_effect = [
                    async_iterator(
                        [
                            TaskResponse(bytes_=b"test-1", iteration=0, size=1),
                            TaskResponse(bytes_=b"test-2", iteration=1, size=1),
                        ]
                    ),
                    asyncio.CancelledError,
                ]
                task = self.task_listener.listen(self.callback)

                with self.assertRaises(asyncio.CancelledError):
                    await task

                # Cancel the task and reset for next test
                task.cancel()
                request_stub.reset_mock()
                sleep.reset_mock()
            self.task_listener._post_handle_raise.reset_mock()


class TestSender(unittest.IsolatedAsyncioTestCase):
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

    def setUp(self):
        self.serializer_patch = patch("fedbiomed.transport.client.Serializer")
        self.serializer_mock = self.serializer_patch.start()
        self.channels = MagicMock()
        self.channels.stub = AsyncMock()
        self.channels.connect = AsyncMock()
        self.channels.feedback_stub.Feedback = MagicMock(
            spec=grpc.aio.UnaryUnaryMultiCallable
        )
        self.channels.task_stub.ReplyTask = MagicMock(
            spec=grpc.aio.StreamUnaryMultiCallable
        )
        self.on_status_change = AsyncMock()
        self.sender = Sender(
            channels=self.channels, on_status_change=self.on_status_change
        )

    def tearDown(self) -> None:
        self.serializer_patch.stop()
        pass

    async def test_sender_01_send(self):
        await self.sender.send(message=self.message_search)
        item = await self.sender._queue.get()
        self.assertEqual(
            item, {"stub": _StubType.SENDER_TASK_STUB, "message": self.message_search}
        )

        await self.sender.send(message=self.message_log)
        item = await self.sender._queue.get()
        self.assertEqual(
            item, {"stub": _StubType.SENDER_FEEDBACK_STUB, "message": self.message_log}
        )

        await self.sender.send(message=self.message_scalar)
        item = await self.sender._queue.get()
        self.assertEqual(
            item,
            {"stub": _StubType.SENDER_FEEDBACK_STUB, "message": self.message_scalar},
        )

    async def test_sender_02_listen(self):
        self.serializer_patch.stop()

        future = asyncio.Future()
        future.set_result("x")

        self.channels.feedback_stub.Feedback.side_effect = [
            future,
            asyncio.CancelledError,
        ]
        self.channels.stub.return_value = self.channels.feedback_stub
        await self.sender.send(message=self.message_log)
        await self.sender.send(message=self.message_log)

        task = self.sender.listen()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.channels.feedback_stub.Feedback.call_count, 2)

        task.cancel()

        stream_call = AsyncMock()
        self.channels.task_stub.ReplyTask.side_effect = [
            stream_call,
            asyncio.CancelledError,
        ]
        self.channels.stub.return_value = self.channels.task_stub
        await self.sender.send(message=self.message_search)
        await self.sender.send(message=self.message_search)

        task = self.sender.listen()
        with self.assertRaises(asyncio.CancelledError):
            await task

        task.cancel()
        self.assertEqual(self.channels.task_stub.ReplyTask.call_count, 2)
        stream_call.write.assert_called_once()
        stream_call.done_writing.assert_called_once()

    @patch("fedbiomed.transport.client.asyncio.sleep")
    async def test_sender_02_listen_exceptions(self, sleep):
        # Unknown error
        codes = [
            grpc.aio.AioRpcError(
                code=grpc.StatusCode.UNKNOWN,
                trailing_metadata=grpc.aio.Metadata(("test", "test")),
                initial_metadata=grpc.aio.Metadata(("test", "test")),
            ),
            grpc.aio.AioRpcError(
                code=grpc.StatusCode.ABORTED,
                trailing_metadata=grpc.aio.Metadata(("test", "test")),
                initial_metadata=grpc.aio.Metadata(("test", "test")),
            ),
        ]
        for message in [self.message_log, self.message_scalar]:
            for code in codes:
                for retry in range(1, MAX_SEND_RETRIES + 5):
                    self.channels.stub.return_value = self.channels.feedback_stub
                    self.channels.feedback_stub.Feedback.side_effect = [
                        code
                    ] * retry + [asyncio.CancelledError]

                    await self.sender.send(message=message)
                    await self.sender.send(message=message)
                    task = self.sender.listen()

                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertEqual(
                        self.channels.feedback_stub.Feedback.call_count, retry + 1
                    )
                    self.assertEqual(
                        sleep.call_count, retry - int((retry - 1) / MAX_SEND_RETRIES)
                    )

                    # Cancel the task
                    task.cancel()

                    # Need a successful task retrieve to reset the retry counters
                    future = asyncio.Future()
                    future.set_result("x")
                    self.channels.feedback_stub.Feedback.side_effect = [
                        future,
                        asyncio.CancelledError,
                    ]
                    self.channels.stub.return_value = self.channels.feedback_stub
                    await self.sender.send(message=message)
                    await self.sender.send(message=message)
                    task = self.sender.listen()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

                    # Cancel and reset the task for next test
                    task.cancel()
                    self.channels.feedback_stub.reset_mock()
                    sleep.reset_mock()

        # Unavailable
        for message in [self.message_log, self.message_scalar]:
            for retry in range(1, MAX_SEND_RETRIES + 5):
                self.channels.stub.return_value = self.channels.feedback_stub
                self.channels.feedback_stub.Feedback.side_effect = [
                    grpc.aio.AioRpcError(
                        code=grpc.StatusCode.UNAVAILABLE,
                        trailing_metadata=grpc.aio.Metadata(("test", "test")),
                        initial_metadata=grpc.aio.Metadata(("test", "test")),
                    )
                ] * retry + [asyncio.CancelledError]

                await self.sender.send(message=message)
                await self.sender.send(message=message)
                task = self.sender.listen()

                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(
                    self.channels.feedback_stub.Feedback.call_count, retry + 1
                )
                self.assertEqual(
                    sleep.call_count, retry - int((retry - 1) / MAX_SEND_RETRIES)
                )

                # Cancel the task
                task.cancel()

                # Need a successful task retrieve to reset the retry counters
                future = asyncio.Future()
                future.set_result("x")
                self.channels.feedback_stub.Feedback.side_effect = [
                    future,
                    asyncio.CancelledError,
                ]
                self.channels.stub.return_value = self.channels.feedback_stub
                await self.sender.send(message=message)
                await self.sender.send(message=message)
                task = self.sender.listen()
                with self.assertRaises(asyncio.CancelledError):
                    await task

                # Cancel the task
                task.cancel()
                self.channels.feedback_stub.reset_mock()
                sleep.reset_mock()

        # Deadline
        for message in [self.message_log, self.message_scalar]:
            for retry in range(1, MAX_SEND_RETRIES + 5):
                self.channels.stub.return_value = self.channels.feedback_stub
                self.channels.feedback_stub.Feedback.side_effect = [
                    grpc.aio.AioRpcError(
                        code=grpc.StatusCode.DEADLINE_EXCEEDED,
                        trailing_metadata=grpc.aio.Metadata(("test", "test")),
                        initial_metadata=grpc.aio.Metadata(("test", "test")),
                    )
                ] * retry + [asyncio.CancelledError]

                await self.sender.send(message=message)
                await self.sender.send(message=message)
                task = self.sender.listen()

                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(
                    self.channels.feedback_stub.Feedback.call_count, retry + 1
                )
                sleep.assert_not_called()

                # Cancel the task
                task.cancel()

                # Need a successful task retrieve to reset the retry counters
                future = asyncio.Future()
                future.set_result("x")
                self.channels.feedback_stub.Feedback.side_effect = [
                    future,
                    asyncio.CancelledError,
                ]
                self.channels.stub.return_value = self.channels.feedback_stub
                await self.sender.send(message=message)
                await self.sender.send(message=message)
                task = self.sender.listen()
                with self.assertRaises(asyncio.CancelledError):
                    await task

                task.cancel()
                self.channels.feedback_stub.reset_mock()
                sleep.reset_mock()

        # Other exceptions
        codes = [
            RuntimeError,
            Exception,
            GeneratorExit,
        ]
        for message in [self.message_log, self.message_scalar]:
            for code in codes:
                for retry in range(1, MAX_SEND_RETRIES + 5):
                    self.channels.stub.return_value = self.channels.feedback_stub
                    self.channels.feedback_stub.Feedback.side_effect = [
                        code
                    ] * retry + [asyncio.CancelledError]

                    await self.sender.send(message=message)
                    await self.sender.send(message=message)
                    task = self.sender.listen()
                    if retry <= MAX_SEND_RETRIES:
                        signal = asyncio.CancelledError
                    else:
                        signal = FedbiomedCommunicationError

                    with self.assertRaises(signal):
                        await task
                    self.assertEqual(
                        self.channels.feedback_stub.Feedback.call_count,
                        min(retry + 1, MAX_SEND_RETRIES + 1),
                    )
                    self.assertEqual(sleep.call_count, min(retry, MAX_SEND_RETRIES))

                    # Cancel the task
                    task.cancel()

                    # Need a successful task retrieve to reset the retry counters
                    future = asyncio.Future()
                    future.set_result("x")
                    self.channels.feedback_stub.Feedback.side_effect = [
                        future,
                        asyncio.CancelledError,
                    ]
                    self.channels.stub.return_value = self.channels.feedback_stub
                    await self.sender.send(message=message)
                    await self.sender.send(message=message)
                    task = self.sender.listen()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

                    # Cancel and reset the task for next test
                    task.cancel()
                    self.channels.feedback_stub.reset_mock()
                    sleep.reset_mock()


class TestChannels(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.create_channel_patch = patch(
            "fedbiomed.transport.client.Channels._create_channel", autospec=True
        )
        self.stub_patch = patch(
            "fedbiomed.transport.client.ResearcherServiceStub", autospec=True
        )

        self.create_channel_mock = self.create_channel_patch.start()

        self.create_channel_mock.return_value.close = AsyncMock()

        self.stub_mock = self.stub_patch.start()

        r = ResearcherCredentials(host="localhost", port="50051", certificate=b"test")

        self.channels = Channels(researcher=r)

        pass

    def tearDown(self):
        self.create_channel_patch.stop()
        self.stub_patch.stop()
        pass

    async def test_channels_02_connect_and_stub(self):
        await self.channels.connect()
        for stub in [
            _StubType.LISTENER_TASK_STUB,
            _StubType.SENDER_TASK_STUB,
            _StubType.SENDER_FEEDBACK_STUB,
        ]:
            self.assertIsInstance(await self.channels.stub(stub), ResearcherServiceStub)

        # Recall connect
        await self.channels.connect()
        for stub in [
            _StubType.LISTENER_TASK_STUB,
            _StubType.SENDER_TASK_STUB,
            _StubType.SENDER_FEEDBACK_STUB,
        ]:
            self.assertIsInstance(await self.channels.stub(stub), ResearcherServiceStub)

        # test non existing stub
        self.assertEqual(await self.channels.stub("dummy"), None)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

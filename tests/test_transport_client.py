import unittest
import asyncio
import threading
import grpc 


from unittest.mock import patch, MagicMock, AsyncMock
from fedbiomed.transport.client import GrpcClient, \
    ClientStatus, \
    ResearcherCredentials, \
    TaskListener, \
    Sender, \
    Channels
from fedbiomed.common.exceptions import FedbiomedCommunicationError
from fedbiomed.common.message import SearchReply, FeedbackMessage, Log
from fedbiomed.transport.protocols.researcher_pb2 import TaskResponse
from fedbiomed.transport.protocols.researcher_pb2_grpc import ResearcherServiceStub
from testsupport.mock import AsyncMock


class TestGrpcClient(unittest.IsolatedAsyncioTestCase):


    def setUp(self):

        # Create patches
        self.stub_patch = patch("fedbiomed.transport.client.ResearcherServiceStub", autospec=True)
        self.sender_patch = patch("fedbiomed.transport.client.Sender", autospec=True)

        # Start patched
        self.task_listener_patch = patch("fedbiomed.transport.client.TaskListener", autospec=True)
        self.stub_mock = self.stub_patch.start()
        self.sender_mock = self.sender_patch.start()
        self.task_listener_mock = self.task_listener_patch.start()


        self.update_id_map = AsyncMock()

        credentials = ResearcherCredentials(
            port="50051",
            host="localhost"
        )


        self.client = GrpcClient(
            node_id="test-node-id",
            researcher=credentials,
            update_id_map=self.update_id_map
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

        self.client._on_status_change(ClientStatus.CONNECTED)
        self.assertEqual(self.client._status, ClientStatus.CONNECTED)

    async def test_grpc_client_05__update_id(self):



        await self.client._update_id(id_='test')
        self.assertEqual(self.client._id, 'test')
        self.update_id_map.assert_called_once_with(
            f"{self.client._researcher.host}:{self.client._researcher.port}", 'test'
        )

        with self.assertRaises(FedbiomedCommunicationError):
            await self.client._update_id(id_='test-malicious')

        pass 


class TestTaskListener(unittest.IsolatedAsyncioTestCase):


    def setUp(self):
        
        self.serializer_patch = patch('fedbiomed.transport.client.Serializer')
        self.serializer_mock = self.serializer_patch.start()
        self.node_id = "test-node-id"
        self.on_status_change = MagicMock()
        self.update_id = MagicMock()
        self.callback = MagicMock()
        self.channels = MagicMock()
        self.channels.connect = AsyncMock()
        self.task_listener = TaskListener(
            channels=self.channels,
            node_id=self.node_id,
            on_status_change=self.on_status_change,
            update_id=self.update_id
        )

    def tearDown(self) -> None:
        self.serializer_patch.stop()
        pass 


    async def test_task_listener_01_listen(self):


        self.on_status_change.side_effect = [None, asyncio.CancelledError]
        self.serializer_mock.load.return_value = {'researcher_id': 'test-researcher-id'}

        async def async_iterator(items):
            for item in items:
                yield item

        # Run with cancel to be able to stop the loop ---------------------
        self.on_status_change.side_effect = [None, asyncio.CancelledError]
        task = self.task_listener.listen(self.callback)
        self.channels.task_stub.GetTaskUnary.return_value = async_iterator([
            TaskResponse(bytes_= b'test-1', iteration=0, size=1),
            TaskResponse(bytes_= b'test-2', iteration=1, size=1)
        ])
        with self.assertRaises(asyncio.CancelledError):
            await task

        # self.callback.assert_called_once()
        self.serializer_mock.loads.assert_called_once()
        self.channels.task_stub.GetTaskUnary.assert_called_once()
        self.update_id.assert_called_once()

        # Cancel the task for next test
        task.cancel()


    @patch('fedbiomed.transport.client.asyncio.sleep')
    async def test_task_listener_02_listen_exceptions(self, sleep):  
        
        # deadline exceeded
        self.channels.task_stub.GetTaskUnary.side_effect = [grpc.aio.AioRpcError(code=grpc.StatusCode.DEADLINE_EXCEEDED,
                                                                   trailing_metadata=grpc.aio.Metadata(('test', 'test')),
                                                                   initial_metadata=grpc.aio.Metadata(('test', 'test'))), 
                                              asyncio.CancelledError]

        task = self.task_listener.listen(self.callback)
        with self.assertRaises(asyncio.CancelledError):
            await task
        # Cancel the task for next test
        task.cancel()

        # unavailable
        self.channels.task_stub.GetTaskUnary.side_effect = [grpc.aio.AioRpcError(code=grpc.StatusCode.UNAVAILABLE,
                                                                   trailing_metadata=grpc.aio.Metadata(('test', 'test')),
                                                                   initial_metadata=grpc.aio.Metadata(('test', 'test'))), 
                                              asyncio.CancelledError]

        task = self.task_listener.listen(self.callback)
        with self.assertRaises(asyncio.CancelledError):
            await task
        # Cancel the task for next test
        task.cancel()
        sleep.assert_called()

        # unknown
        self.channels.task_stub.GetTaskUnary.side_effect = [
            grpc.aio.AioRpcError(code=grpc.StatusCode.UNKNOWN,
                                 trailing_metadata=grpc.aio.Metadata(('test', 'test')),
                                 initial_metadata=grpc.aio.Metadata(('test', 'test'))),
            asyncio.CancelledError]

        task = self.task_listener.listen(self.callback)
        with self.assertRaises(asyncio.CancelledError):
            await task
        sleep.assert_called()
        # Cancel the task for next test
        task.cancel()

        # For all others
        self.channels.task_stub.GetTaskUnary.side_effect = [
            grpc.aio.AioRpcError(
                code=grpc.StatusCode.ABORTED,
                trailing_metadata=grpc.aio.Metadata(('test', 'test')),
                initial_metadata=grpc.aio.Metadata(('test', 'test'))),
            asyncio.CancelledError]

        task = self.task_listener.listen(self.callback)
        with self.assertRaises(asyncio.CancelledError):
            await task
        sleep.assert_called()
        # Cancel the task for next test
        task.cancel()


class TestSender(unittest.IsolatedAsyncioTestCase):


    message_search = SearchReply(
            researcher_id='test',
            success=True,
            databases=[],
            node_id='node-id',
            count=1,
            command='search',
        )

    message_log = FeedbackMessage(
            researcher_id='test',
            log=Log(
                node_id='test',
                level='DEBUG',
                msg="Error message"
            )
    )

    def setUp(self):

        self.serializer_patch = patch('fedbiomed.transport.client.Serializer')
        self.serializer_mock = self.serializer_patch.start()
        self.channels = MagicMock()
        self.channels.connect = AsyncMock()
        self.channels.feedback_stub.Feedback = MagicMock(spec=grpc.aio.UnaryUnaryMultiCallable)
        self.channels.task_stub.ReplyTask = MagicMock(spec=grpc.aio.StreamUnaryMultiCallable)
        self._task_stub = MagicMock()
        self._task_stub.ReplyTask = MagicMock(spec=grpc.aio.StreamUnaryMultiCallable)
        self.on_status_change = MagicMock()
        self.sender = Sender(
            channels=self.channels,
            on_status_change=self.on_status_change
        )

    def tearDown(self) -> None:
        self.serializer_patch.stop()
        pass 


    async def test_sender_01_send(self):

        await self.sender.send(message=self.message_search)
        item = await self.sender._queue.get()
        self.assertEqual(item, {'stub': self.channels.task_stub.ReplyTask, 'message': self.message_search})


        await self.sender.send(message=self.message_log)
        item = await self.sender._queue.get()
        self.assertEqual(item, {'stub': self.channels.feedback_stub.Feedback, 'message': self.message_log.to_proto()})


    async def test_sender_02_listen(self):
        self.serializer_patch.stop()

        future = asyncio.Future()
        future.set_result('x')

        self.channels.feedback_stub.Feedback.side_effect = [future, asyncio.CancelledError]
        await self.sender.send(message=self.message_log)
        await self.sender.send(message=self.message_log)

        task = self.sender.listen()
        with self.assertRaises(asyncio.CancelledError):
            await task
        task.cancel()

        stream_call = AsyncMock()
        self.channels.task_stub.ReplyTask.side_effect = [stream_call, asyncio.CancelledError]
        await self.sender.send(message=self.message_search)
        await self.sender.send(message=self.message_search)

        task = self.sender.listen()
        with self.assertRaises(asyncio.CancelledError):
            await task

        task.cancel()
        stream_call.write.assert_called()
        stream_call.done_writing.assert_called()

    @patch('fedbiomed.transport.client.asyncio.sleep')
    async def test_sender_02_listen_exceptions(self, sleep):

        # Unknown error
        self.channels.feedback_stub.Feedback.side_effect = [
            grpc.aio.AioRpcError(code=grpc.StatusCode.UNKNOWN,
                                 trailing_metadata=grpc.aio.Metadata(('test', 'test')),
                                 initial_metadata=grpc.aio.Metadata(('test', 'test'))),
            asyncio.CancelledError]

        await self.sender.send(message=self.message_log)
        await self.sender.send(message=self.message_log)

        task = self.sender.listen()
        with self.assertRaises(asyncio.CancelledError):
            await task
        task.cancel()

        # Unavailable
        retry = self.sender._retry_count 
        self.channels.feedback_stub.Feedback.side_effect = [
            grpc.aio.AioRpcError(code=grpc.StatusCode.UNAVAILABLE,
                                 trailing_metadata=grpc.aio.Metadata(('test', 'test')),
                                 initial_metadata=grpc.aio.Metadata(('test', 'test'))),
            asyncio.CancelledError]
        
        await self.sender.send(message=self.message_log)
        await self.sender.send(message=self.message_log)

        task = self.sender.listen()
        with self.assertRaises(asyncio.CancelledError):
            await task
        task.cancel()
        sleep.assert_called()
        self.assertEqual(self.sender._retry_count, retry+1)


        # Deadline
        retry = self.sender._retry_count 
        self.channels.feedback_stub.Feedback.side_effect = [
            grpc.aio.AioRpcError(code=grpc.StatusCode.DEADLINE_EXCEEDED,
                                 trailing_metadata=grpc.aio.Metadata(('test', 'test')),
                                 initial_metadata=grpc.aio.Metadata(('test', 'test'))),
            asyncio.CancelledError]

        await self.sender.send(message=self.message_log)
        await self.sender.send(message=self.message_log)

        task = self.sender.listen()
        with self.assertRaises(asyncio.CancelledError):
            await task
        task.cancel()


class TestChannels(unittest.IsolatedAsyncioTestCase):

    def setUp(self):

        self.create_channel_patch = patch("fedbiomed.transport.client.Channels._create_channel", autospec=True)
        self.stub_patch = patch("fedbiomed.transport.client.ResearcherServiceStub", autospec=True)

        self.create_channel_mock = self.create_channel_patch.start()

        self.create_channel_mock.return_value.close = AsyncMock()

        self.stub_mock = self.stub_patch.start()

        r = ResearcherCredentials(
            host='localhost',
            port='50051',
            certificate=b'test')

        self.channels = Channels(researcher=r)

        pass

    def tearDown(self):
        self.create_channel_patch.stop()
        self.stub_patch.stop()
        pass


    async def test_channels_02_connect(self):

        await self.channels.connect()
        self.assertIsInstance(self.channels.feedback_stub, ResearcherServiceStub)
        self.assertIsInstance(self.channels.task_stub, ResearcherServiceStub)

        # Recall connect
        await self.channels.connect()
        self.assertIsInstance(self.channels.feedback_stub, ResearcherServiceStub)
        self.assertIsInstance(self.channels.task_stub, ResearcherServiceStub)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()




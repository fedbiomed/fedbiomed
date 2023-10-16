import unittest

from unittest.mock import patch, MagicMock
from fedbiomed.transport.client import GrpcClient, \
    ClientStatus, \
    ResearcherCredentials 
from fedbiomed.common.exceptions import FedbiomedCommunicationError

from testsupport.mock import AsyncMock


class TestGrpcClient(unittest.IsolatedAsyncioTestCase):


    def setUp(self):

        # Create patches
        self.create_channel_patch = patch("fedbiomed.transport.client.create_channel", autospec=True)
        self.stub_patch = patch("fedbiomed.transport.client.ResearcherServiceStub", autospec=True)
        self.sender_patch = patch("fedbiomed.transport.client.Sender", autospec=True)

        # Start patched
        self.task_listener_patch = patch("fedbiomed.transport.client.TaskListener", autospec=True)
        self.create_channel_mock = self.create_channel_patch.start()
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

        self.create_channel_patch.stop()
        self.stub_patch.stop()
        self.sender_patch.stop()
        self.task_listener_patch.stop()
        pass 

    
    def test_grpc_client_01_start(self):
        
        on_task = MagicMock()
        tasks = self.client.start(on_task=on_task)

        self.task_listener_mock.return_value.listen.assert_called_once_with(on_task)
        self.sender_mock.return_value.listen.assert_called_once()
        self.assertIsInstance(tasks, list)
        self.assertEqual(len(tasks), 2)

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
            f"{self.client._host}:{self.client._port}", 'test'
        )


        with self.assertRaises(FedbiomedCommunicationError):
            await self.client._update_id(id_='test-malicious')

        pass 

if __name__ == '__main__':  # pragma: no cover
    unittest.main()




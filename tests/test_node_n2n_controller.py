import unittest
from unittest.mock import MagicMock, AsyncMock

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from fedbiomed.common.message import (
    KeyRequest,
    KeyReply,
    PingRequest,
    OverlayMessage
)
from fedbiomed.node.requests._n2n_controller import NodeToNodeController


async def func_return_argument(message, researcher_id, setup = False):
    return [b"x", b"y"], b'salty salty'


class TestNodeToNodeController(unittest.IsolatedAsyncioTestCase, NodeTestCase):
    """Test for node2node controller module, NodeToNodeRouter class"""

    def setUp(self):
        self.overlay_channel_mock = AsyncMock(autospec=True)
        self.overlay_channel_mock.format_outgoing_overlay.side_effect = func_return_argument
        self.overlay_channel_mock.get_local_public_key.return_value = b'a dummy key'

        self.grpc_controller_mock = MagicMock(autospec=True)
        self.pending_requests_mock = MagicMock(autospec=True)
        self.controller_data_mock = MagicMock(autospec=True)

        self.n2n_controller = NodeToNodeController(
            self.grpc_controller_mock, self.overlay_channel_mock, self.pending_requests_mock, self.controller_data_mock)

        self.overlay_msg = OverlayMessage(
            researcher_id='dummy researcher',
            node_id='dummy source overlay node',
            dest_node_id='dummy dest overlay node',
            overlay=['dummy overlay content'],
            setup=False,
            salt=b'my salt',
        )

        self.inner_msg = KeyRequest(
            node_id='dummy source inner node',
            dest_node_id='dummy dest inner node',
            request_id='dummy request id',
            secagg_id='dummy secagg id',
        )


    def tearDown(self):
        pass

    async def test_n2n_controller_01_handle_key_request(self):
        """Handle incoming message KeyRequest in node to node controller
        """
        # 1. public key is available

        # prepare
        public_key = b'my dummy pubkey'
        self.controller_data_mock.wait.return_value = [True, [{'public_key': public_key}]]


        # action
        result = await self.n2n_controller.handle(self.overlay_msg, self.inner_msg)

        # check
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(['overlay_resp']))
        self.assertEqual(result['overlay_resp'].researcher_id, self.overlay_msg.researcher_id)


        # 2. public key is not available

        # prepare
        self.controller_data_mock.wait.return_value = [False, [{'any': 'dummy'}]]

        # action
        result = await self.n2n_controller.handle(self.overlay_msg, self.inner_msg)

        # check
        self.assertIsNone(result)

    async def test_n2n_controller_02_handle_key_reply(self):
        """Handle incoming message KeyReply in node to node controller
        """
        inner_msg = KeyReply(**{
            'node_id': 'dummy source inner node',
            'dest_node_id': 'dummy dest inner node',
            'request_id': 'dummy request id',
            'public_key': b'dummy public bytes',
            'secagg_id': 'dummy secagg id',
        })

        # action
        result = await self.n2n_controller.handle(self.overlay_msg, inner_msg)

        # check
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(['inner_msg']))
        self.assertEqual(result['inner_msg'].public_key, inner_msg.public_key)

    async def test_n2n_controller_03_handle_key_default(self):
        """Handle incoming message non existing in node to node controller
        """
        # None exssting message/request
        inner_msg = PingRequest(researcher_id='x')
        result = await self.n2n_controller.handle(self.overlay_msg, inner_msg)
        self.assertIsNone(result)

    async def test_n2n_controller_04_final_key_request(self):
        """Final handler for key request in node to node controller
        """
        # prepare
        overlay_msg = OverlayMessage(**{
            'researcher_id': 'dummy researcher',
            'node_id': 'dummy source overlay node',
            'dest_node_id': 'dummy dest overlay node',
            'overlay': ['dummy overlay content'],
            'setup': False,
            'salt': b'a dummy salt',
        })
        await self.n2n_controller.final('KeyRequest', overlay_resp=overlay_msg)
        self.grpc_controller_mock.send.assert_called_once()

    async def test_n2n_controller_04_final_key_reply(self):
        """Final handler for key reply in node to node controller
        """
        await self.n2n_controller.final(KeyReply.__name__, inner_msg=self.inner_msg)
        self.pending_requests_mock.event.assert_called_once()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

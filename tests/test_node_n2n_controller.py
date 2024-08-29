import unittest
from unittest.mock import patch, MagicMock

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from fedbiomed.common.message import NodeMessages
from fedbiomed.node.requests._n2n_controller import NodeToNodeController

from testsupport.fake_message import FakeMessages


def func_return_argument(message):
    return message


class TestNodeToNodeController(unittest.IsolatedAsyncioTestCase, NodeTestCase):
    """Test for node2node controller module, NodeToNodeRouter class"""

    def setUp(self):
        self.format_outgoing_patch = patch('fedbiomed.node.requests._n2n_controller.format_outgoing_overlay', autospec=True)

        self.format_outgoing_patcher = self.format_outgoing_patch.start()
        def fake_format_outgoing_message(message):
            return message
        self.format_outgoing_patcher.side_effect = func_return_argument

        self.grpc_controller_mock = MagicMock(autospec=True)
        self.pending_requests_mock = MagicMock(autospec=True)
        self.controller_data_mock = MagicMock(autospec=True)

        self.n2n_controller = NodeToNodeController(
            self.grpc_controller_mock, self.pending_requests_mock, self.controller_data_mock)

        self.overlay_msg = {
            'researcher_id': 'dummy researcher',
            'node_id': 'dummy source overlay node',
            'dest_node_id': 'dummy dest overlay node',
            'overlay': ['dummy overlay content'],
            'command': 'overlay',
        }
        self.inner_msg = FakeMessages({
            'node_id': 'dummy source inner node',
            'dest_node_id': 'dummy dest inner node',
            'request_id': 'dummy request id',
            'secagg_id': 'dummy secagg id',
            'command': 'key-request',
        })


    def tearDown(self):
        self.format_outgoing_patch.stop()

    @patch('fedbiomed.node.requests._n2n_controller.NodeToNodeMessages.format_outgoing_message')
    @patch('fedbiomed.node.requests._n2n_controller.NodeMessages.format_outgoing_message')
    async def test_n2n_controller_01_handle_key_request(self, nm_format_outgoing, n2nm_format_outgoing):
        """Handle incoming message KeyRequest in node to node controller
        """
        # 1. public key is available

        # prepare
        public_key = 'my dummy pubkey'
        self.controller_data_mock.wait.return_value = [True, [{'public_key': public_key}]]

        nm_format_outgoing.side_effect = func_return_argument
        n2nm_format_outgoing.side_effect = func_return_argument

        # action
        result = await self.n2n_controller.handle(self.overlay_msg, self.inner_msg)

        # check
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(['overlay_resp']))
        self.assertEqual(result['overlay_resp']['researcher_id'], self.overlay_msg['researcher_id'])
        self.assertEqual(result['overlay_resp']['overlay']['request_id'], self.inner_msg.get_param('request_id'))
        self.assertEqual(result['overlay_resp']['overlay']['public_key'], public_key)


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
        # prepare
        inner_msg = FakeMessages({
            'node_id': 'dummy source inner node',
            'dest_node_id': 'dummy dest inner node',
            'request_id': 'dummy request id',
            'public-bytes': b'dummy public bytes',
            'secagg_id': 'dummy secagg id',
            'command': 'key-reply',
        })

        # action
        result = await self.n2n_controller.handle(self.overlay_msg, inner_msg)

        # check
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(['inner_msg']))
        self.assertEqual(result['inner_msg'].get_param('public-bytes'), inner_msg.get_param('public-bytes'))

    async def test_n2n_controller_03_handle_key_default(self):
        """Handle incoming message non existing in node to node controller
        """
        # prepare
        inner_msg = FakeMessages({
            'command': 'not-existing',
        })

        # action
        result = await self.n2n_controller.handle(self.overlay_msg, inner_msg)

        # check
        self.assertIsNone(result)

    async def test_n2n_controller_04_final_key_request(self):
        """Final handler for key request in node to node controller
        """
        # prepare
        overlay_msg = NodeMessages.format_outgoing_message({
            'researcher_id': 'dummy researcher',
            'node_id': 'dummy source overlay node',
            'dest_node_id': 'dummy dest overlay node',
            'overlay': ['dummy overlay content'],
            'command': 'overlay',
        })

        # action
        await self.n2n_controller.final('key-request', overlay_resp=overlay_msg)

        # check
        self.grpc_controller_mock.send.assert_called_once()

    async def test_n2n_controller_04_final_key_reply(self):
        """Final handler for key reply in node to node controller
        """
        # prepare

        # action
        await self.n2n_controller.final('key-reply', inner_msg=self.inner_msg)

        # check
        self.pending_requests_mock.event.assert_called_once()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

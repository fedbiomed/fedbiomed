import unittest
from unittest.mock import MagicMock, AsyncMock

from fedbiomed.common.message import (
    ChannelSetupRequest,
    ChannelSetupReply,
    KeyRequest,
    KeyReply,
    AdditiveSSharingRequest,
    AdditiveSSharingReply,
    PingRequest,
    OverlayMessage
)
from fedbiomed.node.requests._n2n_controller import NodeToNodeController


async def func_return_argument(message, researcher_id, setup = False):
    return b"payload payload", b'salty salty', b'noncy noncy'


class TestNodeToNodeController(unittest.IsolatedAsyncioTestCase, unittest.TestCase):
    """Test for node2node controller module, NodeToNodeRouter class"""

    def setUp(self):
        self.overlay_channel_mock = AsyncMock(autospec=True)
        self.overlay_channel_mock.format_outgoing_overlay.side_effect = func_return_argument
        self.overlay_channel_mock.get_local_public_key.return_value = b'a dummy key'

        self.grpc_controller_mock = MagicMock(autospec=True)
        self.pending_requests_mock = MagicMock(autospec=True)
        self.controller_data_mock = MagicMock(autospec=True)

        self.n2n_controller = NodeToNodeController(
            'test-node-id',
            self.grpc_controller_mock,
            self.overlay_channel_mock,
            self.pending_requests_mock,
            self.controller_data_mock
        )

        self.overlay_msg = OverlayMessage(
            researcher_id='dummy researcher',
            node_id='dummy source overlay node',
            dest_node_id='dummy dest overlay node',
            overlay=b'dummy overlay content',
            setup=False,
            salt=b'my salt',
            nonce=b'my own nonce',
        )

        self.inner_msg = KeyRequest(
            node_id='dummy source inner node',
            dest_node_id='dummy dest inner node',
            request_id='dummy request id',
            secagg_id='dummy secagg id',
        )


    def tearDown(self):
        pass

    async def test_n2n_controller_01_handle_key_ASS_request(self):
        """Handle incoming message KeyRequest AdditiveSSharingRequest in node to node controller
        """
        # prepare
        node_id = 'dummy source inner node'
        for inner_msg, controller_data in zip([
            self.inner_msg,
            AdditiveSSharingRequest(
                node_id=node_id,
                dest_node_id='dummy dest inner node',
                request_id='dummy request id',
                secagg_id='dummy secagg id',
            ),
            AdditiveSSharingRequest(
                node_id=node_id,
                dest_node_id='dummy dest inner node',
                request_id='dummy request id',
                secagg_id='dummy secagg id',
            ),
        ], [
            [{'public_key': b'my dummy pubkey'}],
            [{'shares': {node_id: 12345}}],
            [{'shares': {node_id: ['shares shares']}}],
        ]
        ):
            # 1. public key is available

            # prepare
            self.controller_data_mock.wait.return_value = [True, controller_data]


            # action
            result = await self.n2n_controller.handle(self.overlay_msg, inner_msg)

            # check
            self.assertIsInstance(result, dict)
            self.assertEqual(set(result.keys()), set(['overlay_resp']))
            self.assertEqual(result['overlay_resp'].researcher_id, self.overlay_msg.researcher_id)


            # 2. public key is not available

            # prepare
            self.controller_data_mock.wait.return_value = [False, [{'any': 'dummy'}]]

            # action
            result = await self.n2n_controller.handle(self.overlay_msg, inner_msg)

            # check
            self.assertIsNone(result)

    async def test_n2n_controller_02_handle_key_ASS_reply(self):
        """Handle incoming message KeyReply AdditiveSSharingReply in node to node controller
        """
        for inner_msg in [
            KeyReply(**{
                'node_id': 'dummy source inner node',
                'dest_node_id': 'dummy dest inner node',
                'request_id': 'dummy request id',
                'public_key': b'dummy public bytes',
                'secagg_id': 'dummy secagg id',
            }),
            AdditiveSSharingReply(
                node_id='dummy source inner node',
                dest_node_id='dummy dest inner node',
                request_id='dummy request id',
                secagg_id='another dummy secagg id',
                share=12345,
            ),
        ]:

            # action
            result = await self.n2n_controller.handle(self.overlay_msg, inner_msg)

            # check
            self.assertIsInstance(result, dict)
            self.assertEqual(set(result.keys()), set(['inner_msg']))
            self.assertEqual(result['inner_msg'].secagg_id, inner_msg.secagg_id)

    async def test_n2n_controller_03_handle_key_default(self):
        """Handle incoming message non existing in node to node controller
        """
        # None exssting message/request
        inner_msg = PingRequest(researcher_id='x')
        result = await self.n2n_controller.handle(self.overlay_msg, inner_msg)
        self.assertIsNone(result)

    async def test_n2n_controller_04_final_key_ASS_request(self):
        """Final handler for key/ASS request in node to node controller
        """
        # prepare
        overlay_msg = OverlayMessage(**{
            'researcher_id': 'dummy researcher',
            'node_id': 'dummy source overlay node',
            'dest_node_id': 'dummy dest overlay node',
            'overlay': b'dummy overlay content',
            'setup': False,
            'salt': b'a dummy salt',
            'nonce': b'my own nonce',
        })

        for message_name in ['KeyRequest', 'AdditiveSSharingRequest']:
            # action + test
            await self.n2n_controller.final(message_name, overlay_resp=overlay_msg)
            self.grpc_controller_mock.send.assert_called_once()

            self.grpc_controller_mock.reset_mock()

    async def test_n2n_controller_04_final_key_ASS_reply(self):
        """Final handler for key/ASS reply in node to node controller
        """
        for message_name in [
            KeyReply.__name__,
            AdditiveSSharingReply.__name__,
        ]:
            await self.n2n_controller.final(message_name, inner_msg=self.inner_msg)
            self.pending_requests_mock.event.assert_called_once()

            self.pending_requests_mock.reset_mock()

    async def test_n2n_controller_05_handle_channel_request(self):
        """Handle incoming message channel setup request in node to node controller
        """
        # prepare
        self.overlay_channel_mock.get_local_public_key.return_value = b'my public key'
        inner_msg = ChannelSetupRequest(
            node_id = 'dummy source inner node',
            dest_node_id = 'dummy dest inner node',
            request_id = 'dummy request id',
        )

        # action
        result = await self.n2n_controller.handle(self.overlay_msg, inner_msg)

        # check
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(['overlay_resp']))
        self.assertEqual(result['overlay_resp'].researcher_id, self.overlay_msg.researcher_id)

    async def test_n2n_controller_06_final_channel_reply(self):
        """Final handler for channel setup request in node to node controller
        """
        # prepare
        inner_msg = ChannelSetupReply(
            node_id = 'dummy source inner node',
            dest_node_id = 'dummy dest inner node',
            request_id = 'dummy request id',
            public_key = b'my public key',
        )

        for r in [True, False]:
            self.overlay_channel_mock.set_distant_key.return_value = r

            # test
            await self.n2n_controller.final(ChannelSetupReply.__name__, inner_msg=inner_msg)

            # test
            self.overlay_channel_mock.set_distant_key.assert_called_once()

            # reset
            self.overlay_channel_mock.reset_mock()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

import unittest
from unittest.mock import MagicMock, PropertyMock

from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.message import InnerMessage, InnerRequestReply
from fedbiomed.node.requests._send_nodes import send_nodes


class TestNodeRequestsSendNodes(unittest.TestCase):
    """Test for node overlay send_nodes utilility"""

    def setUp(self):
        self.n2n_router_mock = MagicMock(autospec=True)
        self.grpc_controller_mock = MagicMock(autospec=True)
        self.pending_requests_mock = MagicMock(autospec=True)

        type(self.n2n_router_mock).node_id = PropertyMock(return_value='test-node-id')

    def tearDown(self):
        pass

    def test_requests_01_send_nodes_success(self):
        """Test send_nodes function successful call
        """
        # prepare
        self.n2n_router_mock.format_outgoing_overlay.return_value = (
            b'overlay bytes dummmy', b'salt dummy', b'nonce dummy'
        )

        def pending_requests_wait(request_ids, timeout):
            return True, request_ids
        self.pending_requests_mock.wait.side_effect = pending_requests_wait

        nodes = [ 'node1', 'node2', 'node3' ]
        request_id = 'my dummy id for inner request reply'
        messages = [
            InnerRequestReply(
                node_id= 'my node id',
                dest_node_id= nodes[0],
                request_id= request_id,
            ),
            InnerMessage(
                node_id= 'my 2nd node id',
                dest_node_id= nodes[1],
            ),
            InnerRequestReply(
                node_id= 'my 3rd node id',
                dest_node_id= nodes[2],
                request_id= request_id,
            )
        ]

        # action
        all_received, replies = send_nodes(
            self.n2n_router_mock,
            self.grpc_controller_mock,
            self.pending_requests_mock,
            'dummy_researcher_id',
            nodes,
            messages,
        )

        # check
        self.assertEqual(all_received, True)
        self.assertEqual(tuple(replies), tuple([request_id, request_id]))

    def test_requests_02_send_nodes_failed(self):
        """Test send_nodes function failed call
        """
        # prepare
        self.n2n_router_mock.format_outgoing_overlay.return_value = (
            b'overlay bytes dummmy', b'salt dummy', b'nonce dummy'
        )
        reply_message = InnerRequestReply(
            node_id= 'my 3rd node id',
            dest_node_id= 'any node',
            request_id= 'any request'
        )

        def pending_requests_wait(request_ids, timeout):
            reply_messages = []
            for r in request_ids:
                reply_messages.append(reply_message)
            return False, reply_messages
        self.pending_requests_mock.wait.side_effect = pending_requests_wait

        nodes = [ 'node1', 'node2', 'node3' ]
        request_id = 'my dummy id for inner request reply'
        messages = [
            InnerRequestReply(
                node_id= 'my node id',
                dest_node_id= nodes[0],
                request_id= request_id,
            ),
            InnerMessage(
                node_id= 'my 2nd node id',
                dest_node_id= nodes[1],
            ),
            InnerRequestReply(
                node_id= 'my 3rd node id',
                dest_node_id= nodes[2],
                request_id= request_id,
            )
        ]

        # 1. don't raise exception if failed

        # action
        all_received, replies = send_nodes(
            self.n2n_router_mock,
            self.grpc_controller_mock,
            self.pending_requests_mock,
            'dummy_researcher_id',
            nodes,
            messages,
        )

        # check
        self.assertEqual(all_received, False)
        self.assertEqual(tuple(replies), tuple([reply_message, reply_message]))

        # 2. raise exception if failed

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            all_received, replies = send_nodes(
                self.n2n_router_mock,
                self.grpc_controller_mock,
                self.pending_requests_mock,
                'dummy_researcher_id',
                nodes,
                messages,
                raise_if_not_all_received=True,
            )




if __name__ == '__main__':  # pragma: no cover
    unittest.main()

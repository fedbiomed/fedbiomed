import unittest
from unittest.mock import MagicMock

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from fedbiomed.common.message import InnerMessage, InnerRequestReply
from fedbiomed.node.requests._send_nodes import send_nodes


class TestNodeRequestsSendNodes(NodeTestCase):
    """Test for node overlay send_nodes utilility"""

    def setUp(self):
        self.n2n_router_mock = MagicMock(autospec=True)
        self.grpc_controller_mock = MagicMock(autospec=True)
        self.pending_requests_mock = MagicMock(autospec=True)

    def tearDown(self):
        pass

    def test_requests_01_send_nodes(self):
        """Test send_nodes function
        """
        # prepare
        self.n2n_router_mock.format_outgoing_overlay.return_value = ([b'overlay bytes dummmy'], b'salt dummy')

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
        _, replies = send_nodes(
            self.n2n_router_mock,
            self.grpc_controller_mock,
            self.pending_requests_mock,
            'dummy_researcher_id',
            nodes,
            messages,
        )

        # check
        self.assertEqual(tuple(replies), tuple([request_id, request_id]))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

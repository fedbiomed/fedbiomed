import unittest
from unittest.mock import patch, MagicMock

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.message import Message, KeyRequest, InnerMessage, InnerRequestReply
from fedbiomed.common.serializer import Serializer
from fedbiomed.node.requests._overlay import format_outgoing_overlay, format_incoming_overlay, send_nodes, \
    _default_n2n_key, _CHUNK_SIZE, load_default_n2n_key


class TestNodeRequestsOverlay(NodeTestCase):
    """Test for node overlay communications module"""

    def setUp(self):
        self.grpc_controller_mock = MagicMock(autospec=True)
        self.pending_requests_mock = MagicMock(autospec=True)

        self.inner_message = InnerMessage(
            node_id= 'my node id',
            dest_node_id= 'my dest node id',
        )

    def tearDown(self):
        pass

    def test_overlay_01_format_out_in(self):
        """Test outgoing + incoming formatting function
        """
        # prepare
        src_message = KeyRequest(
            node_id= 'my node id',
            dest_node_id= 'my dest node id',
            request_id='my request id',
            secagg_id='my secagg id',
        )

        # action
        dest_message = format_incoming_overlay(format_outgoing_overlay(src_message))

        # check
        self.assertIsInstance(dest_message, InnerMessage)
        self.assertEqual(set(src_message.get_dict().keys()), set(dest_message.get_dict().keys()))
        for k in src_message.get_dict().keys():
            self.assertEqual(src_message.get_param(k), dest_message.get_param(k))


    def test_overlay_02_send_nodes(self):
        """Test send_nodes function
        """
        # prepare
        def pending_requests_wait(request_ids, timeout):
            return True, request_ids
        self.pending_requests_mock.wait.side_effect = pending_requests_wait

        nodes = [ 'node1', 'node2']
        request_id = 'my dummy id for inner request reply'
        messages = [
            InnerMessage(
                node_id= 'my node id',
                dest_node_id= nodes[0],
            ),
            InnerRequestReply(
                node_id= 'my node id',
                dest_node_id= nodes[0],
                request_id= request_id,
            )
        ]

        # action
        _, replies = send_nodes(
            self.grpc_controller_mock,
            self.pending_requests_mock,
            'dummy_researcher_id',
            nodes,
            messages,
        )

        # check
        self.assertEqual(set(replies), set([request_id]))

    def test_overlay_03_format_outgoing_failure_arguments(self):
        """Test outgoing formatting function failure because of bad arguments
        """
        # prepare
        messages = [
            Message(),
            {'command': 'key-request'},
            False,
            3
        ]

        # action + check
        for m in messages:
            with self.assertRaises(FedbiomedNodeToNodeError):
                format_outgoing_overlay(m)

    @patch('fedbiomed.node.requests._overlay._CHUNK_SIZE', 10**6)
    def test_overlay_04_format_outgoing_failure_key_size(self):
        """Test outgoing formatting function failure because of bad key size
        """
        # prepare

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            format_outgoing_overlay(self.inner_message)

    def test_overlay_05_format_incoming_failure_arguments(self):
        """Test incoming formatting function failure because of bad arguments
        """
        # prepare
        payloads = [
            [3],
            [3, b'4'],
            [b'4', 3],
            [b'5',1, b'5'],
            [[b'4']],
        ]

        # action + check
        for p in payloads:
            with self.assertRaises(FedbiomedNodeToNodeError):
                format_incoming_overlay(p)

    @patch('fedbiomed.node.requests._overlay._CHUNK_SIZE', 10**6)
    def test_overlay_06_format_incoming_failure_key_size(self):
        """Test incoming formatting function failure because of bad key size
        """
        # prepare
        payload = [b'123456123456123456']

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            format_incoming_overlay(payload)

    def test_overlay_07_format_incoming_failure_decrypt(self):
        """Test incoming formatting function failure at decryption
        """
        # prepare
        payload = [b'123456123456123456']

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            format_incoming_overlay(payload)

    def test_overlay_08_format_incoming_failure_bad_message_content(self):
        """Test incoming formatting function failure bad encrypted message content
        """
        # prepare
        signed = Serializer.dumps({
            'message': self.inner_message.get_dict(),
            # intentionally forget to add signature field
        })
        payload = [
            _default_n2n_key.public_key().encrypt(
                signed[i:i + _CHUNK_SIZE],
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            for i in range(0, len(signed), _CHUNK_SIZE)
        ]

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            format_incoming_overlay(payload)

    def test_overlay_09_format_incoming_failure_bad_message_signature(self):
        """Test incoming formatting function failure bad encrypted message signature
        """
        # prepare
        inner_message_modified = {
            'node_id': 'different node id',
            'dest_node_id': 'my dest node id',
        }
        signed = Serializer.dumps({
            'message': self.inner_message.get_dict(),
            'signature': _default_n2n_key.sign(
                Serializer.dumps(inner_message_modified),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()

            )
        })
        payload = [
            _default_n2n_key.public_key().encrypt(
                signed[i:i + _CHUNK_SIZE],
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            for i in range(0, len(signed), _CHUNK_SIZE)
        ]

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            format_incoming_overlay(payload)

    @patch('fedbiomed.node.requests._overlay._DEFAULT_N2N_KEY_FILE', 'non_existing_filename')
    def test_overlay_10_load_default_key_failure(self):
        """Test loading default key failure
        """
        # prepare
        with self.assertRaises(FedbiomedNodeToNodeError):
            load_default_n2n_key()

if __name__ == '__main__':  # pragma: no cover
    unittest.main()

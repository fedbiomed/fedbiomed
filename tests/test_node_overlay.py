import unittest
from unittest.mock import patch, MagicMock
import copy

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.message import Message, KeyRequest, InnerMessage, OverlayMessage
from fedbiomed.common.secagg._dh import DHKeyAgreement
from fedbiomed.common.serializer import Serializer
from fedbiomed.transport.controller import GrpcController
from fedbiomed.node.requests._overlay import OverlayChannel


class TestNodeRequestsOverlay(unittest.IsolatedAsyncioTestCase, NodeTestCase):
    """Test for node overlay communications module"""

    def setUp(self):
        self.asyncio_event_patch = patch('fedbiomed.node.requests._overlay.asyncio.Event', autospec=True)
        self.channel_manager_patch = patch('fedbiomed.node.requests._overlay.ChannelManager', autospec=True)
        self.dhkey_agreement_patch = patch('fedbiomed.node.requests._overlay.DHKeyAgreement.agree')

        self.asyncio_event_mock = self.asyncio_event_patch.start()
        self.channel_manager_mock = self.channel_manager_patch.start()
        self.dhkey_agreement_mock = self.dhkey_agreement_patch.start()

        self.grpc_controller_mock = MagicMock(spec=GrpcController)

        self.inner_message = InnerMessage(
            node_id= 'my node id',
            dest_node_id= 'my dest node id',
        )

        self.overlay_channel = OverlayChannel(self.grpc_controller_mock)
        self.default_private_key, _ = self.overlay_channel._load_default_n2n_key()

        # needs to be 32 bytes long for ChaCha20
        self.dhkey_agreement_mock.return_value = b'k' * 32

    def tearDown(self):
        self.channel_manager_patch.stop()
        self.asyncio_event_patch.stop()
        self.dhkey_agreement_patch.stop()

    async def test_overlay_01_format_out_in(self):
        """Test outgoing + incoming formatting function
        """
        # prepare
        researcher_id = 'my dummy researched id'
        node_id = 'my node id'
        dest_node_id = 'my dest node id'
        src_message = KeyRequest(
            node_id=node_id,
            dest_node_id=dest_node_id,
            request_id='my request id',
            secagg_id='my secagg id',
        )

        # action
        payload, salt, nonce = await self.overlay_channel.format_outgoing_overlay(src_message, researcher_id, setup=True)
        overlay_message = OverlayMessage(
            researcher_id=researcher_id,
            node_id=node_id,
            dest_node_id=dest_node_id,
            overlay=payload,
            setup=True,
            salt=salt,
            nonce=nonce,
        )
        dest_message = await self.overlay_channel.format_incoming_overlay(overlay_message)

        # check
        self.assertIsInstance(dest_message, InnerMessage)
        self.assertEqual(set(src_message.get_dict().keys()), set(dest_message.get_dict().keys()))
        for k in src_message.get_dict().keys():
            self.assertEqual(src_message.get_param(k), dest_message.get_param(k))

    async def test_overlay_03_format_outgoing_failure_arguments(self):
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
                await self.overlay_channel.format_outgoing_overlay(m, 'dummy_researcher_id')

    async def test_overlay_05_format_incoming_failure_arguments(self):
        """Test incoming formatting function failure because of bad arguments
        """
        # prepare
        messages = [
            3,
            [b'4'],
            {'3': 4},
            InnerMessage(
                node_id='src node',
                dest_node_id='dest node',
            )
        ]

        # action + check
        for m in messages:
            with self.assertRaises(FedbiomedNodeToNodeError):
                await self.overlay_channel.format_incoming_overlay(m)

    async def test_overlay_07_format_incoming_failure_decrypt(self):
        """Test incoming formatting function failure at decryption with bad payload
        """
        # prepare
        payload = b'123456123456123456'
        overlay_message = OverlayMessage(
            researcher_id='any unique id',
            node_id='another unique id',
            dest_node_id='dest node unique id',
            overlay=payload,
            setup=True,
            salt=b'12345abcde',
            nonce=b'nopqrst',
        )

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_incoming_overlay(overlay_message)

    async def test_overlay_08_format_incoming_failure_bad_message_content(self):
        """Test incoming formatting function failure bad encrypted message content
        """
        # prepare
        signed = Serializer.dumps({
            'message': self.inner_message.get_dict(),
            # intentionally forget to add signature field
        })
        nonce = b'0123456789abcdef'
        salt = b'12345abcde'
        node_id = 'another unique id'
        dest_node_id = 'dest node unique id'

        derived_key = DHKeyAgreement(
            node_id, self.default_private_key, salt
        ).agree(dest_node_id, self.default_private_key.export_public_key())
        encryptor = Cipher(
            algorithms.ChaCha20(derived_key, nonce),
            mode=None,
            backend=default_backend()
        ).encryptor()
        payload = encryptor.update(signed) + encryptor.finalize()
        overlay_message = OverlayMessage(
            researcher_id='any unique id',
            node_id=node_id,
            dest_node_id=dest_node_id,
            overlay=payload,
            setup=True,
            salt=salt,
            nonce=nonce,
        )

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_incoming_overlay(overlay_message)

    async def test_overlay_09_format_incoming_failure_bad_message_signature(self):
        """Test incoming formatting function failure bad encrypted message signature
        """
        # prepare
        nonce = b'0123456789abcdef'
        salt = b'12345abcde'
        inner_message_modified = copy.deepcopy(self.inner_message)
        inner_message_modified.node_id = 'different node id'

        signed = Serializer.dumps({
            'message': self.inner_message.to_dict(),
            'signature': self.default_private_key.private_key.sign(
                Serializer.dumps(inner_message_modified.to_dict()),
                ec.ECDSA(hashes.SHA256()),
            )
        })
        #payload = [
        #    self.default_private_key.public_key.encrypt(
        #        signed[i:i + _CHUNK_SIZE],
        #        padding.OAEP(
        #            mgf=padding.MGF1(algorithm=hashes.SHA256()),
        #            algorithm=hashes.SHA256(),
        #            label=None
        #        )
        #    )
        #    for i in range(0, len(signed), _CHUNK_SIZE)
        #]
        derived_key = DHKeyAgreement(
            self.inner_message.node_id, self.default_private_key, salt
        ).agree(self.inner_message.dest_node_id, self.default_private_key.export_public_key())
        encryptor = Cipher(
            algorithms.ChaCha20(derived_key, nonce),
            mode=None,
            backend=default_backend()
        ).encryptor()
        payload = encryptor.update(signed) + encryptor.finalize()
        overlay_message = OverlayMessage(
            researcher_id='any unique id',
            node_id=self.inner_message.node_id,
            dest_node_id=self.inner_message.dest_node_id,
            overlay=payload,
            setup=True,
            salt=salt,
            nonce=nonce,
        )

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_incoming_overlay(overlay_message)

    @patch('fedbiomed.node.requests._overlay._DEFAULT_N2N_KEY_FILE', 'non_existing_filename')
    async def test_overlay_10_load_default_key_failure(self):
        """Test loading default key failure
        """
        # prepare
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel._load_default_n2n_key()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

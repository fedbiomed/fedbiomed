import unittest
from unittest.mock import patch, MagicMock

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.message import Message, KeyRequest, InnerMessage, OverlayMessage
from fedbiomed.common.serializer import Serializer
from fedbiomed.transport.controller import GrpcController
from fedbiomed.node.requests._overlay import OverlayChannel, _CHUNK_SIZE


class TestNodeRequestsOverlay(unittest.IsolatedAsyncioTestCase, NodeTestCase):
    """Test for node overlay communications module"""

    def setUp(self):
        self.asyncio_event_patch = patch('fedbiomed.node.requests._overlay.asyncio.Event', autospec=True)
        self.channel_manager_patch = patch('fedbiomed.node.requests._overlay.ChannelManager', autospec=True)

        self.asyncio_event_mock = self.asyncio_event_patch.start()
        self.channel_manager_mock = self.channel_manager_patch.start()

        self.grpc_controller_mock = MagicMock(spec=GrpcController)

        self.inner_message = InnerMessage(
            node_id= 'my node id',
            dest_node_id= 'my dest node id',
        )

        self.overlay_channel = OverlayChannel(self.grpc_controller_mock)
        self.default_private_key, _ = self.overlay_channel._load_default_n2n_key()

    def tearDown(self):
        self.channel_manager_patch.stop()
        self.asyncio_event_patch.stop()

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
        payload, salt = await self.overlay_channel.format_outgoing_overlay(src_message, researcher_id, setup=True)
        overlay_message = OverlayMessage(
            researcher_id=researcher_id,
            node_id=node_id,
            dest_node_id=dest_node_id,
            overlay=payload,
            setup=True,
            salt=salt,
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

    @patch('fedbiomed.node.requests._overlay._CHUNK_SIZE', 10**6)
    async def test_overlay_04_format_outgoing_failure_key_size(self):
        """Test outgoing formatting function failure because of bad key size
        """
        # prepare

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_outgoing_overlay(self.inner_message, 'dummy_researcher_id')

    async def test_overlay_05_format_incoming_failure_arguments(self):
        """Test incoming formatting function failure because of bad arguments
        """
        # prepare
        payloads = [
            [3],
            [3, b'4'],
            [b'4', 3],
            [b'5', 1, b'5'],
            [[b'4']],
        ]

        # action + check
        for p in payloads:
            overlay_message = OverlayMessage(
                researcher_id='any unique id',
                node_id='another unique id',
                dest_node_id='dest node unique id',
                overlay=p,
                setup=True,
                salt=b'12345abcde',
            )

            with self.assertRaises(FedbiomedNodeToNodeError):
                await self.overlay_channel.format_incoming_overlay(overlay_message)

    @patch('fedbiomed.node.requests._overlay._CHUNK_SIZE', 10**6)
    async def test_overlay_06_format_incoming_failure_key_size(self):
        """Test incoming formatting function failure because of bad key size
        """
        # prepare
        payload = [b'123456123456123456']
        overlay_message = OverlayMessage(
            researcher_id='any unique id',
            node_id='another unique id',
            dest_node_id='dest node unique id',
            overlay=payload,
            setup=True,
            salt=b'12345abcde',
        )

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_incoming_overlay(overlay_message)

    async def test_overlay_07_format_incoming_failure_decrypt(self):
        """Test incoming formatting function failure at decryption
        """
        # prepare
        payload = [b'123456123456123456']
        overlay_message = OverlayMessage(
            researcher_id='any unique id',
            node_id='another unique id',
            dest_node_id='dest node unique id',
            overlay=payload,
            setup=True,
            salt=b'12345abcde',
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
        payload = [
            self.default_private_key.public_key.encrypt(
                signed[i:i + _CHUNK_SIZE],
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            for i in range(0, len(signed), _CHUNK_SIZE)
        ]
        overlay_message = OverlayMessage(
            researcher_id='any unique id',
            node_id='another unique id',
            dest_node_id='dest node unique id',
            overlay=payload,
            setup=True,
            salt=b'12345abcde',
        )

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_incoming_overlay(overlay_message)

    async def test_overlay_09_format_incoming_failure_bad_message_signature(self):
        """Test incoming formatting function failure bad encrypted message signature
        """
        # prepare
        inner_message_modified = {
            'node_id': 'different node id',
            'dest_node_id': 'my dest node id',
        }
        signed = Serializer.dumps({
            'message': self.inner_message.get_dict(),
            'signature': self.default_private_key.private_key.sign(
                Serializer.dumps(inner_message_modified),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()

            )
        })
        payload = [
            self.default_private_key.public_key.encrypt(
                signed[i:i + _CHUNK_SIZE],
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            for i in range(0, len(signed), _CHUNK_SIZE)
        ]
        overlay_message = OverlayMessage(
            researcher_id='any unique id',
            node_id='another unique id',
            dest_node_id='dest node unique id',
            overlay=payload,
            setup=True,
            salt=b'12345abcde',
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

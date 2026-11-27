import copy
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.message import (
    ChannelSetupReply,
    ChannelSetupRequest,
    InnerMessage,
    KeyRequest,
    Message,
    OverlayMessage,
)
from fedbiomed.common.secagg._dh import DHKey
from fedbiomed.common.serializer import Serializer
from fedbiomed.node.requests._overlay import OverlayChannel, _ChannelKeys
from fedbiomed.transport.controller import GrpcController


class TestNodeRequestsChannelKeys(unittest.IsolatedAsyncioTestCase, unittest.TestCase):
    """Test for node overlay communications module channel keys handling"""

    def setUp(self):
        self.asyncio_event_patch = patch(
            "fedbiomed.node.requests._overlay.asyncio.Event", autospec=True
        )
        self.channel_manager_patch = patch(
            "fedbiomed.node.requests._overlay.ChannelManager", autospec=True
        )
        self.asyncio_waitfor_patch = patch(
            "fedbiomed.node.requests._overlay.asyncio.wait_for"
        )

        self.asyncio_event_mock = self.asyncio_event_patch.start()
        self.channel_manager_mock = self.channel_manager_patch.start()
        self.asyncio_waitfor_mock = self.asyncio_waitfor_patch.start()

        # `asyncio.wait_for` is mocked and so never awaits its argument; make the
        # mocked `Event.wait()` return a plain value instead of a coroutine so it
        # is not reported as an un-awaited coroutine.
        self.asyncio_event_mock.return_value.wait = MagicMock()

        self.grpc_controller_mock = MagicMock(spec=GrpcController)

        self.inner_message = InnerMessage(
            node_id="my node id",
            dest_node_id="my dest node id",
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.temp_dir.name, "test.json")

        self.overlay_channel = OverlayChannel(
            db=self.db, node_id="test-node-id", grpc_client=self.grpc_controller_mock
        )

        self.private_key = DHKey()
        # needs to be 32 bytes long for ChaCha20
        self.derived_key = b"k" * 32

        self.channel_keys = _ChannelKeys(db=self.db)

    def tearDown(self):
        self.temp_dir.cleanup()
        self.channel_manager_patch.stop()
        self.asyncio_event_patch.stop()
        self.asyncio_waitfor_patch.stop()

    async def test_channel_keys_01_basic(self):
        """Basic operations on channel keys object"""

        # prepare
        node1 = "node1"
        node2 = "node2"
        request1 = "request1"
        request2 = "request2"
        channel_keys2 = self.overlay_channel._channel_keys

        # action
        with patch("fedbiomed.node.requests._overlay.logger.debug") as logger_debug:
            local_key1a, distant_key1a = await self.channel_keys.get_keys(node1)
            await self.channel_keys.add_pending_request(node1, request1)
            set1 = await self.channel_keys.set_distant_key(
                node1, self.private_key.export_public_key(), request1
            )
            local_key1b, distant_key1b = await self.channel_keys.get_keys(node1)
            local_public1 = await self.channel_keys.get_local_public_key(node1)

            local_key2a, distant_key2a = await channel_keys2.get_keys(node2)
            await channel_keys2.add_pending_request(node2, request2)
            set2 = await self.overlay_channel.set_distant_key(
                node2, self.private_key.export_public_key(), request2
            )
            local_key2b, distant_key2b = await channel_keys2.get_keys(node2)
            local_public2 = await self.overlay_channel.get_local_public_key(node2)

            set3 = await self.channel_keys.set_distant_key(
                "non existing node",
                self.private_key.export_public_key(),
                "any request",
            )

        # test
        self.assertEqual(local_key1a, local_key1b)
        self.assertEqual(distant_key1a, None)
        self.assertTrue(set1)
        self.assertEqual(
            distant_key1b.export_public_key(),
            self.private_key.export_public_key(),
        )
        self.assertEqual(local_public1, local_key1a.export_public_key())

        self.assertEqual(local_key2a, local_key2b)
        self.assertEqual(distant_key2a, None)
        self.assertTrue(set2)
        self.assertEqual(
            distant_key2b.export_public_key(),
            self.private_key.export_public_key(),
        )
        self.assertEqual(local_public2, local_key2a.export_public_key())

        self.assertFalse(set3)
        debug_messages = [call.args[0] for call in logger_debug.call_args_list]
        self.assertTrue(
            any("Creating node-to-node channel state" in msg for msg in debug_messages)
        )
        self.assertTrue(
            any(
                "Registered pending node-to-node channel request" in msg
                for msg in debug_messages
            )
        )
        self.assertTrue(any("Stored distant node key" in msg for msg in debug_messages))
        self.assertTrue(
            any("Ignoring distant node key update" in msg for msg in debug_messages)
        )

    async def test_channel_keys_02_wait_ready_channel(self):
        """Test channel keys object wait_ready_channel"""
        # prepare
        node_id = "any node"
        await self.channel_keys.get_keys(node_id)

        # 1. completes before timeout
        is_ready = await self.channel_keys.wait_ready_channel(node_id)
        self.assertTrue(is_ready)

        # 2. timeout before completes
        self.asyncio_waitfor_mock.side_effect = TimeoutError
        is_ready = await self.channel_keys.wait_ready_channel(node_id)
        self.assertFalse(is_ready)

    async def test_channel_keys_03_init_load_keys(self):
        """Test channel keys object constructor reloading keys"""
        # prepare
        node1 = "node1"
        node2 = "node2"
        self.channel_manager_mock.return_value.list.return_value = [node1, node2]
        self.channel_manager_mock.return_value.get.return_value = {
            "local_key": self.private_key.export_private_key()
        }

        # need to re-instantiate after mocking
        channel_keys = _ChannelKeys(self.db)

        # action
        kn1 = await channel_keys.get_local_public_key(node1)
        kn2 = await channel_keys.get_local_public_key(node2)
        kn3 = await channel_keys.get_local_public_key("other node")

        # check
        self.assertEqual(kn1, self.private_key.export_public_key())
        self.assertEqual(kn2, self.private_key.export_public_key())
        self.assertNotEqual(kn3, self.private_key.export_public_key())


class TestNodeRequestsOverlayChannel(
    unittest.IsolatedAsyncioTestCase, unittest.TestCase
):
    """Test for node overlay communications module"""

    def setUp(self):
        self.asyncio_event_patch = patch(
            "fedbiomed.node.requests._overlay.asyncio.Event", autospec=True
        )
        self.channel_manager_patch = patch(
            "fedbiomed.node.requests._overlay.ChannelManager", autospec=True
        )
        self.dhkey_agreement_patch = patch(
            "fedbiomed.node.requests._overlay.DHKeyAgreement.agree"
        )
        self.asyncio_waitfor_patch = patch(
            "fedbiomed.node.requests._overlay.asyncio.wait_for"
        )

        self.asyncio_event_mock = self.asyncio_event_patch.start()
        self.channel_manager_mock = self.channel_manager_patch.start()
        self.dhkey_agreement_mock = self.dhkey_agreement_patch.start()
        self.asyncio_waitfor_mock = self.asyncio_waitfor_patch.start()

        self.grpc_controller_mock = MagicMock(spec=GrpcController)

        self.inner_message = InnerMessage(
            node_id="my node id",
            dest_node_id="my dest node id",
        )

        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.temp_dir.name, "test.json")

        self.overlay_channel = OverlayChannel(
            node_id="test-node-id", db=self.db, grpc_client=self.grpc_controller_mock
        )
        self.private_key = DHKey()
        self.public_key = DHKey(public_key_pem=self.private_key.export_public_key())

        # needs to be 32 bytes long for ChaCha20
        self.derived_key = b"k" * 32
        self.dhkey_agreement_mock.return_value = self.derived_key
        self.overlay_channel._setup_or_use_channel_keys = AsyncMock(
            return_value=(self.private_key, self.public_key, self.derived_key)
        )

    def tearDown(self):
        self.temp_dir.cleanup()
        self.channel_manager_patch.stop()
        self.asyncio_event_patch.stop()
        self.dhkey_agreement_patch.stop()
        self.asyncio_waitfor_patch.stop()

    async def test_overlay_01_format_out_in(self):
        """Test outgoing + incoming plaintext channel setup formatting."""
        # prepare
        researcher_id = "my dummy researched id"
        node_id = "my node id"
        dest_node_id = "my dest node id"
        src_message = ChannelSetupReply(
            node_id=node_id,
            dest_node_id=dest_node_id,
            request_id="my request id",
            public_key=self.private_key.export_public_key(),
        )

        # 1. correct message

        # action
        with patch("fedbiomed.node.requests._overlay.logger.debug") as logger_debug:
            payload, salt, nonce = await self.overlay_channel.format_outgoing_overlay(
                src_message, researcher_id, setup=True
            )
            overlay_message = OverlayMessage(
                researcher_id=researcher_id,
                node_id=node_id,
                dest_node_id=dest_node_id,
                overlay=payload,
                setup=True,
                salt=salt,
                nonce=nonce,
            )
            dest_message = await self.overlay_channel.format_incoming_overlay(
                overlay_message
            )

        # check
        self.assertEqual(Serializer.loads(payload), src_message.to_dict())
        self.assertEqual(salt, b"")
        self.assertEqual(nonce, b"")
        self.overlay_channel._setup_or_use_channel_keys.assert_not_awaited()
        self.assertIsInstance(dest_message, InnerMessage)
        self.assertEqual(
            set(src_message.get_dict().keys()), set(dest_message.get_dict().keys())
        )
        for k in src_message.get_dict().keys():
            self.assertEqual(src_message.get_param(k), dest_message.get_param(k))
        debug_messages = [call.args[0] for call in logger_debug.call_args_list]
        self.assertTrue(
            any("Formatting outgoing overlay message" in msg for msg in debug_messages)
        )
        self.assertTrue(
            any(
                "Formatted outgoing plaintext channel setup payload" in msg
                for msg in debug_messages
            )
        )
        self.assertTrue(
            any("Formatting incoming overlay message" in msg for msg in debug_messages)
        )
        self.assertTrue(
            any("Validated incoming overlay message" in msg for msg in debug_messages)
        )

        # 2. node_id mismatch
        overlay_message.node_id = "another node id"
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_incoming_overlay(overlay_message)

        # 3. dest_node_id mismatch
        overlay_message.node_id = node_id
        overlay_message.dest_node_id = "another dest node id"
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_incoming_overlay(overlay_message)

    async def test_overlay_02_format_outgoing_failure_arguments(self):
        """Test outgoing formatting function failure because of bad arguments"""
        # prepare
        messages = [Message(), {"command": "key-request"}, False, 3]

        # action + check
        for m in messages:
            with self.assertRaises(FedbiomedNodeToNodeError):
                await self.overlay_channel.format_outgoing_overlay(
                    m, "dummy_researcher_id"
                )

        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_outgoing_overlay(
                KeyRequest(
                    node_id="node-1",
                    dest_node_id="node-2",
                    request_id="request-1",
                    secagg_id="secagg-1",
                ),
                "dummy_researcher_id",
                setup=True,
            )

    async def test_overlay_03_format_incoming_failure_arguments(self):
        """Test incoming formatting function failure because of bad arguments"""
        # prepare
        messages = [
            3,
            [b"4"],
            {"3": 4},
            InnerMessage(
                node_id="src node",
                dest_node_id="dest node",
            ),
        ]

        # action + check
        for m in messages:
            with self.assertRaises(FedbiomedNodeToNodeError):
                await self.overlay_channel.format_incoming_overlay(m)

    async def test_overlay_04_format_incoming_failure_decrypt(self):
        """Test incoming formatting failure for malformed plaintext setup."""
        # prepare
        payload = b"123456123456123456"
        overlay_message = OverlayMessage(
            researcher_id="any unique id",
            node_id="another unique id",
            dest_node_id="dest node unique id",
            overlay=payload,
            setup=True,
            salt=b"",
            nonce=b"",
        )

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_incoming_overlay(overlay_message)

    async def test_overlay_05_format_incoming_failure_bad_message_content(self):
        """Test incoming formatting function failure bad encrypted message content"""
        # prepare
        signed = Serializer.dumps(
            {
                "message": self.inner_message.get_dict(),
                # intentionally forget to add signature field
            }
        )
        nonce = b"0123456789abcdef"
        salt = b"12345abcde"
        node_id = "another unique id"
        dest_node_id = "dest node unique id"

        encryptor = Cipher(
            algorithms.ChaCha20(self.derived_key, nonce),
            mode=None,
            backend=default_backend(),
        ).encryptor()
        payload = encryptor.update(signed) + encryptor.finalize()
        overlay_message = OverlayMessage(
            researcher_id="any unique id",
            node_id=node_id,
            dest_node_id=dest_node_id,
            overlay=payload,
            setup=False,
            salt=salt,
            nonce=nonce,
        )

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_incoming_overlay(overlay_message)

    async def test_overlay_06_format_incoming_failure_bad_message_signature(self):
        """Test incoming formatting function failure bad encrypted message signature"""
        # prepare
        nonce = b"0123456789abcdef"
        salt = b"12345abcde"
        inner_message_modified = copy.deepcopy(self.inner_message)
        inner_message_modified.node_id = "different node id"

        signed = Serializer.dumps(
            {
                "message": self.inner_message.to_dict(),
                "signature": self.private_key.private_key.sign(
                    Serializer.dumps(inner_message_modified.to_dict()),
                    ec.ECDSA(hashes.SHA256()),
                ),
            }
        )
        encryptor = Cipher(
            algorithms.ChaCha20(self.derived_key, nonce),
            mode=None,
            backend=default_backend(),
        ).encryptor()
        payload = encryptor.update(signed) + encryptor.finalize()
        overlay_message = OverlayMessage(
            researcher_id="any unique id",
            node_id=self.inner_message.node_id,
            dest_node_id=self.inner_message.dest_node_id,
            overlay=payload,
            setup=False,
            salt=salt,
            nonce=nonce,
        )

        # action + check
        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_incoming_overlay(overlay_message)

    async def test_overlay_07_format_out_in_encrypted(self):
        """Test formatting an application message over established channel keys."""
        src_message = KeyRequest(
            node_id="my node id",
            dest_node_id="my dest node id",
            request_id="my request id",
            secagg_id="my secagg id",
        )

        payload, salt, nonce = await self.overlay_channel.format_outgoing_overlay(
            src_message, "my researcher"
        )
        overlay_message = OverlayMessage(
            researcher_id="my researcher",
            node_id=src_message.node_id,
            dest_node_id=src_message.dest_node_id,
            overlay=payload,
            setup=False,
            salt=salt,
            nonce=nonce,
        )
        dest_message = await self.overlay_channel.format_incoming_overlay(
            overlay_message
        )

        self.assertEqual(len(salt), 32)
        self.assertEqual(len(nonce), 16)
        self.assertEqual(dest_message.to_dict(), src_message.to_dict())
        self.assertEqual(self.overlay_channel._setup_or_use_channel_keys.await_count, 2)

    async def test_overlay_08_setup_or_use_channel_keys(self):
        """Test key setup function for n2n channels"""

        # prepare
        distant_node_id = "my distant node"
        researcher_id = "my_researcher"
        salt = b"1" * 32

        # 1. receive distant node's key

        channel_keys_patch = patch(
            "fedbiomed.node.requests._overlay._ChannelKeys", autospec=True
        )
        channel_keys_mock = channel_keys_patch.start()
        channel_keys_mock.return_value.get_keys.side_effect = [
            # for case 1.
            (self.private_key, None),
            (self.private_key, self.private_key),
            # for case 2.
            (self.private_key, None),
            (self.private_key, self.private_key),
        ]
        # need to re-instantiate for mock to be active
        self.overlay_channel = OverlayChannel(
            self.db, "test-node", self.grpc_controller_mock
        )

        # action
        (
            local_key,
            distant_key,
            derived_key,
        ) = await self.overlay_channel._setup_or_use_channel_keys(
            distant_node_id, researcher_id, salt
        )

        # test
        self.assertEqual(
            local_key.export_public_key(), self.private_key.export_public_key()
        )
        self.assertEqual(
            distant_key.export_public_key(),
            self.private_key.export_public_key(),
        )
        self.assertEqual(derived_key, self.derived_key)

        # 2. don't receive distant node's key

        # prepare
        channel_keys_mock.return_value.wait_ready_channel.return_value = False

        # action
        with self.assertRaises(FedbiomedNodeToNodeError):
            (
                local_key,
                distant_key,
                derived_key,
            ) = await self.overlay_channel._setup_or_use_channel_keys(
                distant_node_id, researcher_id, salt
            )

        # clean
        channel_keys_patch.stop()

    async def test_overlay_10_setup_rejects_encryption_parameters(self):
        """Test plaintext setup rejects legacy encrypted-envelope parameters."""
        message = ChannelSetupRequest(
            node_id="node-1", dest_node_id="node-2", request_id="request-1"
        )
        overlay_message = OverlayMessage(
            researcher_id="researcher",
            node_id=message.node_id,
            dest_node_id=message.dest_node_id,
            overlay=Serializer.dumps(message.to_dict()),
            setup=True,
            salt=b"legacy salt",
            nonce=b"legacy nonce",
        )

        with self.assertRaises(FedbiomedNodeToNodeError):
            await self.overlay_channel.format_incoming_overlay(overlay_message)


@pytest.fixture
def overlay_channel(mocker, tmp_path):
    mocker.patch("fedbiomed.node.requests._overlay.ChannelManager", autospec=True)
    return OverlayChannel(
        node_id="node-2",
        db=str(tmp_path / "node.db"),
        grpc_client=MagicMock(spec=GrpcController),
    )


def _overlay_message(overlay: bytes, setup: bool, nonce: bytes = b"") -> OverlayMessage:
    return OverlayMessage(
        researcher_id="researcher",
        node_id="node-1",
        dest_node_id="node-2",
        overlay=overlay,
        setup=setup,
        salt=b"salt" if not setup else b"",
        nonce=nonce,
    )


def _setup_request_without_protocol_version() -> dict:
    payload = ChannelSetupRequest(
        node_id="node-1", dest_node_id="node-2", request_id="request-1"
    ).to_dict()
    del payload["protocol_version"]
    return payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, error",
    [
        ([1, 2], "not a mapping"),
        (_setup_request_without_protocol_version(), "KeyError.*protocol_version"),
        ({"node_id": "node-1"}, "does not include valid '__type_message__'"),
        (
            KeyRequest(
                node_id="node-1",
                dest_node_id="node-2",
                request_id="request-1",
                secagg_id="secagg-1",
            ).to_dict(),
            "invalid channel setup message type KeyRequest",
        ),
    ],
    ids=["not-a-dict", "missing-field", "no-message-type", "not-a-setup-message"],
)
async def test_plaintext_setup_rejects_invalid_payload(overlay_channel, payload, error):
    message = _overlay_message(Serializer.dumps(payload), setup=True)

    with pytest.raises(FedbiomedNodeToNodeError, match=error):
        await overlay_channel.format_incoming_overlay(message)


def _encrypt(payload: bytes, key: bytes, nonce: bytes) -> bytes:
    encryptor = Cipher(
        algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend()
    ).encryptor()
    return encryptor.update(payload) + encryptor.finalize()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overlay, nonce",
    [
        # ChaCha20 needs a 16-byte nonce
        (b"payload", b"short"),
        # 0xc1 is never used in MsgPack
        (_encrypt(b"\xc1", b"k" * 32, b"n" * 16), b"n" * 16),
    ],
    ids=["bad-nonce", "not-msgpack"],
)
async def test_encrypted_overlay_rejects_undecryptable_payload(
    overlay_channel, overlay, nonce
):
    overlay_channel._setup_or_use_channel_keys = AsyncMock(
        return_value=(None, None, b"k" * 32)
    )
    message = _overlay_message(overlay, setup=False, nonce=nonce)

    with pytest.raises(FedbiomedNodeToNodeError, match="cannot decrypt payload"):
        await overlay_channel.format_incoming_overlay(message)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Optional
import os
from dataclasses import dataclass
import uuid
import secrets
import asyncio

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

from fedbiomed.common.channel_manager import ChannelManager
from fedbiomed.common.constants import ErrorNumbers, TIMEOUT_NODE_TO_NODE_REQUEST, REQUEST_PREFIX
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import Message, InnerMessage, \
    OverlayMessage, ChannelSetupRequest
from fedbiomed.common.secagg import DHKey as DHKeyECC
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.utils import ROOT_DIR

from fedbiomed.transport.controller import GrpcController

from fedbiomed.node.environ import environ

_DEFAULT_KEY_DIR = os.path.join(ROOT_DIR, "envs", "common", "default_keys")
_DEFAULT_N2N_KEY_FILE = "default_n2n_key.pem"

# chunk size must be less or equal (in bits) the smallest RSA key length used
_SMALLEST_KEY = 2048
_CHUNK_SIZE = int(_SMALLEST_KEY / 8)


# Mimic `common.secagg.DHKey` to ease future migration to symmetric encryption


class DHKey(DHKeyECC):
    """Temporarily adapts the key management for using RSA asymmetric encryption

    Currently cannot use elliptic curve crypto `DHkey` with asymmetric encryption.

    Attributes:
        private_key: The user's private RSA key.
        public_key: The user's public RSA key.
    """

    def __init__(
        self,
        private_key_pem: bytes | None = None,
        public_key_pem: bytes | None = None
    ) -> None:
        if private_key_pem:
            self.private_key = self._import_key(
                serialization.load_pem_private_key,
                data=private_key_pem,
                password=None,
                backend=default_backend()
            )
        elif not public_key_pem:
            # Specific to RSA vs ECC
            self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        else:
            # Means that only public key is loaded
            self.private_key = None

        if public_key_pem:
            self.public_key = self._import_key(serialization.load_pem_public_key,
                                               data=public_key_pem,
                                               backend=default_backend()
                                               )
        else:
            self.public_key = self.private_key.public_key()


@dataclass
class _N2nKeysEntry:
    "Stores description of one node to node channel key status"
    local_key: DHKey
    ready_event: asyncio.Event
    distant_key: Optional[DHKey] = None


class _ChannelKeys:
    """Manages encryption/signing keys for each node to node channel

    Attributes:
        _channel_keys: (Dict[str, _N2nKeysEntry]) key status for node to node channels
        _channel_keys_lock: (asyncio.Lock) lock to ensure exclusive access to _channel_keys
    """
    def __init__(self):
        """Class constructor"""
        self._channel_keys = {}
        self._channel_keys_lock = asyncio.Lock()
        self._channel_manager = ChannelManager(environ['DB_PATH'])

        for distant_node_id in self._channel_manager.list():
            channel = self._channel_manager.get(distant_node_id)

            # don't need to acquire _channel_keys_lock in the constructor
            self._channel_keys[distant_node_id] = _N2nKeysEntry(
                local_key=DHKey(private_key_pem=channel['local_key']),
                ready_event=asyncio.Event()
            )

    def _create_channel(self, distant_node_id: str) -> None:
        """Create channel for peering with a given distant peer node if it does not exist yet.

        Assumes it is called with already acquired _channel_keys_lock

        If channel does not exist, create it and generate local key.
        If channel already exists, do nothing.

        Args:
            distant_node_id: unique ID of the peer node
        """
        if distant_node_id not in self._channel_keys:
            local_key = DHKey()
            self._channel_manager.add(distant_node_id, local_key.export_private_key())

            self._channel_keys[distant_node_id] = _N2nKeysEntry(local_key=local_key, ready_event=asyncio.Event())

    async def get_keys(self, distant_node_id: str) -> Tuple[DHKey, DHKey]:
        """Gets keys for peering with a given distant peer node on a channel

        Args:
            distant_node_id: unique ID of the peer node

        Returns:
            A tuple consisting of the public key object for the channel and the private key
                object for the channel (or None if not known yet)
        """
        async with self._channel_keys_lock:
            self._create_channel(distant_node_id)
            return self._channel_keys[distant_node_id].local_key, self._channel_keys[distant_node_id].distant_key

    async def set_distant_key(self, distant_node_id: str, public_key_pem: bytes) -> None:
        """Sets distant (public) key of a channel for peering with a given distant peer node.

        Args:
            distant_node_id: unique ID of the peer node
            public_key_pem: public key in PEM format
        """
        async with self._channel_keys_lock:
            # This case should not happen, but this ensures robustness/integrity
            self._create_channel(distant_node_id)
            self._channel_keys[distant_node_id].distant_key = DHKey(public_key_pem=public_key_pem)
            self._channel_keys[distant_node_id].ready_event.set()

    async def get_local_public_key(self, distant_node_id: str) -> bytes:
        """Gets local public key of a channel for peering with a given distant peer node.

        Args:
            distant_node_id: unique ID of the peer node

        Returns:
            Local node's public key ID for this peer node as bytes
        """
        local_key, _ = await self.get_keys(distant_node_id)
        return local_key.export_public_key()

    async def is_ready_channel(self, distant_node_id: str) -> bool:
        """Checks if keys of a channel to a given distant node are ready for usage, and return immediately

        Args:
            distant_node_id: unique ID of the peer node

        Returns:
            True if keys for the channel are ready for usage, False if keys are not ready.
        """
        async with self._channel_keys_lock:
            # New node to node channel
            self._create_channel(distant_node_id)

            # Channel not fully setup
            #
            # note: there may be an ongoing KeyChannelRequest. In that case, we send
            # another (redundant) request. Each answer updates (completes) the distant_key
            if not self._channel_keys[distant_node_id].distant_key:
                return False

        return True

    async def wait_ready_channel(self, distant_node_id: str) -> bool:
        """Waits until keys of a channel to a given distant node are ready for usage, or timeout is reached.

        Args:
            distant_node_id: unique ID of the peer node

        Returns:
            True if keys for the channel are ready for usage, False if keys are not ready.
        """
        async with self._channel_keys_lock:
            # New node to node channel
            self._create_channel(distant_node_id)

        try:
            await asyncio.wait_for(self._channel_keys[distant_node_id].ready_event.wait(), TIMEOUT_NODE_TO_NODE_REQUEST)
            return True
        except TimeoutError:
            return False


class OverlayChannel:
    """Provides asyncio safe layer for sending and receiving overlay messages.

    This class is not thread safe, all calls must be done within the same thread (except constructor).
    """

    def __init__(self, grpc_client: GrpcController):
        """Class constructor

        Args:
            grpc_client: object managing the communication with other components
        """
        self._grpc_client = grpc_client

        self._channel_keys = _ChannelKeys()

        # Issue #1142 in "Crypto material management" will optionally replace current default key published with the
        # library and used for each node for setup by a keypair generated securely for each node.
        # Caveat: though encrypted, current implementation does not ensure a secure overlay node2node channel ...

        # Default keys
        default_private_key, default_public_key = self._load_default_n2n_key()
        self._default_n2n_key = _N2nKeysEntry(
            local_key=default_private_key,
            ready_event=asyncio.Event(),
            distant_key=default_public_key
        )

        # Default key generation:
        #
        # from cryptography.hazmat.primitives.asymmetric import rsa
        # from cryptography.hazmat.primitives import serialization
        # private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        # private_bytes = private_key.private_bytes(
        #     encoding = serialization.Encoding.PEM,
        #     format = serialization.PrivateFormat.PKCS8,
        #     encryption_algorithm = serialization.NoEncryption()
        # )
        # with open(os.path.join(_DEFAULT_KEY_DIR, _DEFAULT_N2N_KEY_FILE), 'w') as file:
        #     file.write(private_bytes.decode('utf-8'))


    @staticmethod
    def _load_default_n2n_key() -> Tuple[DHKey, DHKey]:
        """Read default node to node private key from file.

        Currently uses the same keypair, published with library, for each node pair.
            To be replaced by a securely generated keypair per node, public key shared
            with other nodes

        Returns:
            A tuple of the default private key object and the public key object

        Raises:
            FedbiomedNodeToNodeError: cannot read key from file
        """
        default_key_file = os.path.join(_DEFAULT_KEY_DIR, _DEFAULT_N2N_KEY_FILE)
        try:
            with open(default_key_file, 'r', encoding="UTF-8") as file:
                private_key_pem = bytes(file.read(), 'utf-8')
        except (FileNotFoundError, PermissionError, ValueError) as e:
            raise FedbiomedNodeToNodeError(
                f'{ErrorNumbers.FB324.value}: cannot read default node to node key '
                f'from file {default_key_file}') from e
        private_key = DHKey(private_key_pem=private_key_pem)
        public_key = DHKey(public_key_pem=private_key.public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo
        ))
        return private_key, public_key


    async def get_local_public_key(self, distant_node_id: str) -> bytes:
        """Gets local public key for peering with a given distant peer node

        Args:
            distant_node_id: unique ID of the peer node

        Returns:
            Local node's public key ID for this peer node
        """
        return await self._channel_keys.get_local_public_key(distant_node_id)


    async def set_distant_key(self, distant_node_id: str, public_key_pem: bytes) -> None:
        """Sets distant (public) key of a channel for peering with a given distant peer node.

        Args:
            distant_node_id: unique ID of the peer node
            public_key_pem: public key in PEM format
        """
        await self._channel_keys.set_distant_key(distant_node_id, public_key_pem)


    async def _setup_use_channel_keys(self, distant_node_id: str, researcher_id: str, setup: bool) \
            -> Tuple[DHKey, DHKey]:
        """Returns channel key objects for

        If key object for local node does not yet exist, generate it.
        If key object for distant node is not yet known, request it.

        Attributes:
            distant_node: unique ID of node at the distant side of the n2n channel
            researcher_id: unique ID of researcher connecting the nodes
            setup: False for sending a message over the channel, True for a message
                setting up the channel

        Returns:
            A tuple consisting of
                - private key object for local node
                - public key object for distant node

        Raises:
            FedbiomedNodeToNodeError: distant node does not answer during channel setup
        """
        # If we are doing channel setup exchange, then use the default or "master" keys for the channel
        if setup:
            return self._default_n2n_key.local_key, self._default_n2n_key.distant_key

        if not await self._channel_keys.is_ready_channel(distant_node_id):
            # Contact node to node channel peer to get its public key
            #
            # nota: channel setup is sequential per peer-node (not optimal in setup time,
            # but more simple implementation, plus acceptable because executed only once for channel setup)
            distant_node_message = ChannelSetupRequest(
                request_id=REQUEST_PREFIX + str(uuid.uuid4()),
                node_id=environ['NODE_ID'],
                dest_node_id=distant_node_id,
            )

            received = await self.send_node_setup(
                researcher_id,
                distant_node_id,
                distant_node_message,
            )
            logger.debug(f"Completed node to node channel setup with success={received} "
                         f"node_id='{environ['NODE_ID']}' distant_node_id='{distant_node_id}")
            if not received:
                raise FedbiomedNodeToNodeError(
                    f"{ErrorNumbers.FB324.value}: A node did not answer during channel setup: {distant_node_id}."
                )

        return await self._channel_keys.get_keys(distant_node_id)


    async def format_outgoing_overlay(self, message: Message, researcher_id: str, setup: bool = False) -> \
            Tuple[List[bytes], bytes]:
        """Creates an overlay message payload from an inner message.

        Serialize, crypt, sign the inner message

        Args:
            message: Inner message to send as overlay payload
            researcher_id: unique ID of researcher connecting the nodes
            setup: False for sending a message over the channel, True for a message
                setting up the channel

        Returns:
            A tuple consisting of: payload for overlay message, salt for inner message encryption

        Raises:
            FedbiomedNodeToNodeError: key is too short
            FedbiomedNodeToNodeError: bad message type
        """
        # robustify from developer error (try to encapsulate a bad message type)
        if not isinstance(message, InnerMessage):
            raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: not an inner message')

        local_node_private_key, distant_node_public_key = await self._setup_use_channel_keys(
            message.get_param('dest_node_id'),
            researcher_id,
            setup
        )

        # `salt` value is unused for now, will be used when moving to symmetric encryption of overlay messages
        # Adjust length of `salt` depending on algorithm (eg: 16 bytes for ChaCha20)
        salt = secrets.token_bytes(16)

        # consider encrypt-sign([message,node_id]) or other see
        # https://theworld.com/~dtd/sign_encrypt/sign_encrypt7.html

        if _CHUNK_SIZE * 8 > min(
                local_node_private_key.private_key.key_size,
                distant_node_public_key.public_key.key_size
        ):
            raise FedbiomedNodeToNodeError(
                f'{ErrorNumbers.FB324.value}: cannot use key shorter than {_CHUNK_SIZE} bits')

        # sign inner payload
        signed = Serializer.dumps({
            'message': message.to_dict(),
            'signature': local_node_private_key.private_key.sign(
                Serializer.dumps(message.to_dict()),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()

            )
        })

        # split to chunks and encrypt
        return [
            distant_node_public_key.public_key.encrypt(
                signed[i:i + _CHUNK_SIZE],
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
            for i in range(0, len(signed), _CHUNK_SIZE)
        ], salt


    async def format_incoming_overlay(self, overlay_msg: OverlayMessage) -> InnerMessage:
        """Retrieves inner message from overlay message payload.

        Check signature, decrypt, deserialize the inner message

        Args:
            overlay_msg: Overlay message.

        Returns:
            Inner message retrieved from overlay payload

        Raises:
            FedbiomedNodeToNodeError: key is too short
            FedbiomedNodeToNodeError: cannot decrypt payload
            FedbiomedNodeToNodeError: bad payload format
            FedbiomedNodeToNodeError: cannot verify payload integrity
            FedbiomedNodeToNodeError: sender/dest node ID don't match in overlay and inner message
        """
        payload = overlay_msg.overlay
        # check payload types (not yet done by message type checks, only checks it's a list)
        if not all(isinstance(p, bytes) for p in payload):
            raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: bad type for node to node payload')

        local_node_private_key, distant_node_public_key = await self._setup_use_channel_keys(
            overlay_msg.node_id,
            overlay_msg.researcher_id,
            overlay_msg.setup
        )

        # decode and ensure only node2node (inner) messages are received

        if _CHUNK_SIZE * 8 > min(
            local_node_private_key.private_key.key_size,
            distant_node_public_key.public_key.key_size
        ):
            raise FedbiomedNodeToNodeError(
                f'{ErrorNumbers.FB324.value}: cannot use key shorter than {_CHUNK_SIZE} bits')

        # decrypt outer payload
        # caveat: decryption can be long for long messages (~10s for 1MB cleartext message)
        try:
            decrypted_chunks = [
                local_node_private_key.private_key.decrypt(
                    chunk,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                for chunk in payload
            ]
        except ValueError as e:
            raise FedbiomedNodeToNodeError(
                f'{ErrorNumbers.FB324.value}: cannot decrypt payload: {e}') from e

        decrypted = Serializer.loads(bytes().join(decrypted_chunks))

        if not isinstance(decrypted, dict) or not set(('message', 'signature')) <= set(decrypted):
            raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: bad inner payload format '
                                           f"in received message")

        # verify inner payload
        try:
            distant_node_public_key.public_key.verify(
                decrypted['signature'],
                Serializer.dumps(decrypted['message']),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        except InvalidSignature as e:
            raise FedbiomedNodeToNodeError(
                f'{ErrorNumbers.FB324.value}: cannot verify payload integrity: {e}') from e

        inner_msg = Message.from_dict(decrypted['message'])

        # Node ID mismatch reveals either (1) malicious peer forging message (2) application internal error
        if inner_msg.node_id != overlay_msg.node_id:
            raise FedbiomedNodeToNodeError(
                f'{ErrorNumbers.FB324.value}: Source node ID mismatch for overlay message '
                f'inner_node_id={inner_msg.node_id} overlay_node_id={overlay_msg.node_id}')
        if inner_msg.dest_node_id != overlay_msg.dest_node_id:
            raise FedbiomedNodeToNodeError(
                f'{ErrorNumbers.FB324.value}: Destination node ID mismatch for overlay message '
                f'inner_node_id={inner_msg.dest_node_id} overlay_node_id={overlay_msg.dest_node_id}')

        return inner_msg

    async def send_node_setup(
            self,
            researcher_id: str,
            node: str,
            message: InnerMessage,
    ) -> bool:
        """Send a channel setup message to another node using overlay communications and wait for its reply.

            Args:
                researcher_id: unique ID of researcher connecting the nodes
                node: unique node ID of the destination node
                message: inner message for the destination node

            Returns:
                True if channel is ready, False if channel not ready after timeout
        """
        overlay, salt = await self.format_outgoing_overlay(message, researcher_id, True)
        message_overlay = OverlayMessage(
            researcher_id=researcher_id,
            node_id=environ['NODE_ID'],
            dest_node_id=node,
            overlay=overlay,
            setup=True,
            salt=salt,
        )

        self._grpc_client.send(message_overlay)

        return await self._channel_keys.wait_ready_channel(node)

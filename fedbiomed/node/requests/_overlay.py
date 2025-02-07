# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Optional
import os
from dataclasses import dataclass, field
import uuid
import secrets
import asyncio

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.exceptions import InvalidSignature

from fedbiomed.common.channel_manager import ChannelManager
from fedbiomed.common.constants import ErrorNumbers, TIMEOUT_NODE_TO_NODE_REQUEST, REQUEST_PREFIX
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import Message, InnerMessage, \
    OverlayMessage, ChannelSetupRequest
from fedbiomed.common.secagg import DHKey, DHKeyAgreement
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.utils import SHARE_DIR

from fedbiomed.transport.controller import GrpcController

_DEFAULT_KEY_DIR = os.path.join(SHARE_DIR, "envs", "common", "default_keys")
_DEFAULT_N2N_KEY_FILE = "default_n2n_key.pem"


@dataclass
class _N2nKeysEntry:
    "Stores description of one node to node channel key status"
    local_key: DHKey
    ready_event: asyncio.Event
    pending_requests: list[str] = field(default_factory=list)
    distant_key: Optional[DHKey] = None


class _ChannelKeys:
    """Manages encryption/signing keys for each node to node channel

    Attributes:
        _channel_keys: (Dict[str, _N2nKeysEntry]) key status for node to node channels
        _channel_keys_lock: (asyncio.Lock) lock to ensure exclusive access to _channel_keys
    """
    def __init__(self, db: str):
        """Class constructor

        Args:
            db: Path to database file of the node.
        """
        self._channel_keys = {}
        self._channel_keys_lock = asyncio.Lock()
        self._channel_manager = ChannelManager(db)

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

            self._channel_keys[distant_node_id] = _N2nKeysEntry(
                local_key=local_key,
                ready_event=asyncio.Event()
            )

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
            return self._channel_keys[distant_node_id].local_key, \
                   self._channel_keys[distant_node_id].distant_key

    async def add_pending_request(
            self,
            distant_node_id: str,
            request_id: str
    ) -> None:
        """Adds a pending request for a channel with a given distant peer node

        Args:
            distant_node_id: unique ID of the peer node
            request_id: unique ID of the request
        """
        async with self._channel_keys_lock:
            self._channel_keys[distant_node_id].pending_requests.append(request_id)

    async def set_distant_key(
            self,
            distant_node_id: str,
            public_key_pem: bytes,
            request_id: str,
    ) -> bool:
        """Sets distant (public) key of a channel for peering with a given
            distant peer node.

        Distant key is not set if no channel exists for that `distant_node_id`
            or if the `request_id` does not match a pending request

        Args:
            distant_node_id: unique ID of the peer node
            public_key_pem: public key in PEM format
            request_id: unique ID of the request

        Returns:
            True if the distant key was set, False if it was not set
        """
        dh_key = DHKey(public_key_pem=public_key_pem)

        async with self._channel_keys_lock:
            if distant_node_id not in self._channel_keys \
                    or request_id not in \
                    self._channel_keys[distant_node_id].pending_requests:
                return False

            self._channel_keys[distant_node_id].distant_key = dh_key
            r = self._channel_keys[distant_node_id].pending_requests
            r.pop(r.index(request_id))
            self._channel_keys[distant_node_id].ready_event.set()
            return True

    async def get_local_public_key(self, distant_node_id: str) -> bytes:
        """Gets local public key of a channel for peering with a given distant peer node.

        Args:
            distant_node_id: unique ID of the peer node

        Returns:
            Local node's public key ID for this peer node as bytes
        """
        local_key, _ = await self.get_keys(distant_node_id)
        return local_key.export_public_key()

    async def wait_ready_channel(self, distant_node_id: str) -> bool:
        """Waits until keys of a channel to a given distant node are ready for usage, or timeout is reached.

        Args:
            distant_node_id: unique ID of the peer node

        Returns:
            True if keys for the channel are ready for usage, False if keys are not ready.
        """
        try:
            await asyncio.wait_for(self._channel_keys[distant_node_id].ready_event.wait(), TIMEOUT_NODE_TO_NODE_REQUEST)
            return True
        except TimeoutError:
            return False


class OverlayChannel:
    """Provides asyncio safe layer for sending and receiving overlay messages.

    This class is not thread safe, all calls must be done within the same thread (except constructor).
    """

    def __init__(
        self,
        node_id: str,
        db: str,
        grpc_client: GrpcController
    ) -> None:
        """Class constructor

        Args:
            node_id: ID of the active node.
            db: Path to database file.
            grpc_client: object managing the communication with other components
        """
        self._node_id = node_id
        self._grpc_client = grpc_client

        self._channel_keys = _ChannelKeys(db)

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
        # from cryptography.hazmat.primitives.asymmetric import ec
        # from cryptography.hazmat.primitives import serialization
        # from cryptography.hazmat.backends import default_backend
        # private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
        # private_byest = private_key.private_bytes(
        #     encoding=serialization.Encoding.PEM,
        #     format=serialization.PrivateFormat.PKCS8,
        #     encryption_algorithm=serialization.NoEncryption(),
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


    async def set_distant_key(
            self,
            distant_node_id: str,
            public_key_pem: bytes,
            request_id: str,
    ) -> bool:
        """Sets distant (public) key of a channel for peering with a given
            distant peer node.

        Distant key is not set if no channel exists for that `distant_node_id`
            or if the `request_id` does not match a pending request

        Args:
            distant_node_id: unique ID of the peer node
            public_key_pem: public key in PEM format
            request_id: unique ID of the request

        Returns:
            True if the distant key was set, False if it was not set
        """
        return await self._channel_keys.set_distant_key(
            distant_node_id,
            public_key_pem,
            request_id
        )


    async def _setup_use_channel_keys(
            self, distant_node_id: str, researcher_id: str, setup: bool, salt: bytes) \
            -> Tuple[DHKey, DHKey, bytes]:
        """Returns channel key objects for

        If key object for local node does not yet exist, generate it.
        If key object for distant node is not yet known, request it.

        Attributes:
            distant_node: unique ID of node at the distant side of the n2n channel
            researcher_id: unique ID of researcher connecting the nodes
            setup: False for sending a message over the channel, True for a message
                setting up the channel
            salt: salt for symmetric encryption key generation

        Returns:
            A tuple consisting of
                - private key object for local node
                - public key object for distant node
                - derived key for symmetric encryption

        Raises:
            FedbiomedNodeToNodeError: distant node does not answer during channel setup
        """

        if setup:
            # If we are doing channel setup exchange, then use the default or "master" keys
            # for the channel
            local_key = self._default_n2n_key.local_key
            distant_key = self._default_n2n_key.distant_key
        else:
            local_key, distant_key = await self._channel_keys.get_keys(distant_node_id)

            if not distant_key:
                # Contact node to node channel peer to get its public key
                #
                # nota: channel setup is sequential per peer-node (not optimal in setup time,
                # but more simple implementation, plus acceptable because executed only once for channel setup)
                distant_node_message = ChannelSetupRequest(
                    request_id=REQUEST_PREFIX + str(uuid.uuid4()),
                    node_id=self._node_id,
                    dest_node_id=distant_node_id,
                )

                await self._channel_keys.add_pending_request(
                    distant_node_id,
                    distant_node_message.request_id
                )
                received = await self.send_node_setup(
                    researcher_id,
                    distant_node_id,
                    distant_node_message,
                )
                logger.debug(f"Completed node to node channel setup with success={received} "
                             f"node_id='{self._node_id}' distant_node_id='{distant_node_id}")
                if not received:
                    raise FedbiomedNodeToNodeError(
                        f"{ErrorNumbers.FB324.value}: A node did not answer during channel setup: {distant_node_id}."
                    )

                local_key, distant_key = await self._channel_keys.get_keys(distant_node_id)

        derived_key = DHKeyAgreement(
            node_u_id=self._node_id,
            node_u_dh_key=local_key,
            session_salt=salt,
        ).agree(
            node_v_id=distant_node_id,
            public_key_pem=distant_key.export_public_key(),
        )
        return local_key, distant_key, derived_key


    async def format_outgoing_overlay(self, message: InnerMessage, researcher_id: str, setup: bool = False) -> \
            Tuple[bytes, bytes, bytes]:
        """Creates an overlay message payload from an inner message.

        Serialize, crypt, sign the inner message

        Args:
            message: Inner message to send as overlay payload
            researcher_id: unique ID of researcher connecting the nodes
            setup: False for sending a message over the channel, True for a message
                setting up the channel

        Returns:
            A tuple consisting of: payload for overlay message, salt for inner message
                encryption key, nonce for the inner message encryption

        Raises:
            FedbiomedNodeToNodeError: bad message type
        """
        # robustify from developer error (try to encapsulate a bad message type)
        if not isinstance(message, InnerMessage):
            raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: not an inner message')

        # Value for salting the symmetric encryption key generation for this message
        # Adjust length of `salt` depending on algorithm
        salt = secrets.token_bytes(32)

        # Value for noncing the symmetric encryption for this message
        # This is normally not needed as we generate different key for each message due to `salt`
        # but provides another layer of security
        # Adjust the length of `nonce` depending on algotrithm
        nonce = secrets.token_bytes(16)

        local_node_private_key, _, derived_key = await self._setup_use_channel_keys(
            message.get_param('dest_node_id'),
            researcher_id,
            setup,
            salt
        )

        # consider encrypt-sign([message,node_id]) or other see
        # https://theworld.com/~dtd/sign_encrypt/sign_encrypt7.html

        # sign inner payload
        signed = Serializer.dumps({
            'message': message.to_dict(),
            'signature': local_node_private_key.private_key.sign(
                Serializer.dumps(message.to_dict()),
                ec.ECDSA(hashes.SHA256()),
            )
        })

        encryptor = Cipher(
            algorithms.ChaCha20(derived_key, nonce),
            mode=None,
            backend=default_backend()
        ).encryptor()
        return encryptor.update(signed) + encryptor.finalize(), salt, nonce


    async def format_incoming_overlay(self, overlay_msg: OverlayMessage) -> InnerMessage:
        """Retrieves inner message from overlay message payload.

        Check signature, decrypt, deserialize the inner message

        Args:
            overlay_msg: Overlay message.

        Returns:
            Inner message retrieved from overlay payload

        Raises:
            FedbiomedNodeToNodeError: bad message type
            FedbiomedNodeToNodeError: cannot decrypt payload
            FedbiomedNodeToNodeError: bad inner payload format
            FedbiomedNodeToNodeError: cannot verify payload integrity
            FedbiomedNodeToNodeError: sender/dest node ID don't match in overlay and inner message
        """
        # robustify from developer error (try to encapsulate a bad message type)
        if not isinstance(overlay_msg, OverlayMessage):
            raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: not an overlay message')


        _, distant_node_public_key, derived_key = await self._setup_use_channel_keys(
            overlay_msg.node_id,
            overlay_msg.researcher_id,
            overlay_msg.setup,
            overlay_msg.salt,
        )

        # decrypt outer payload
        try:
            decryptor = Cipher(
                algorithms.ChaCha20(derived_key, overlay_msg.nonce),
                mode=None,
                backend=default_backend()
            ).decryptor()
            decrypted_serial = decryptor.update(overlay_msg.overlay) + decryptor.finalize()
        except ValueError as e:
            raise FedbiomedNodeToNodeError(
                f'{ErrorNumbers.FB324.value}: cannot decrypt payload: {e}') from e

        decrypted = Serializer.loads(decrypted_serial)
        if not isinstance(decrypted, dict) or not set(('message', 'signature')) <= set(decrypted):
            raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: bad inner payload format '
                                           f"in received message")

        # verify inner payload
        try:
            distant_node_public_key.public_key.verify(
                decrypted['signature'],
                Serializer.dumps(decrypted['message']),
                ec.ECDSA(hashes.SHA256()),
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
        overlay, salt, nonce = await self.format_outgoing_overlay(message, researcher_id, True)
        message_overlay = OverlayMessage(
            researcher_id=researcher_id,
            node_id=self._node_id,
            dest_node_id=node,
            overlay=overlay,
            setup=True,
            salt=salt,
            nonce=nonce,
        )

        self._grpc_client.send(message_overlay)

        return await self._channel_keys.wait_ready_channel(node)

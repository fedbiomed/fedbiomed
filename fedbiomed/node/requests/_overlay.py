# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
from typing import Any, List, Tuple, Optional
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

from fedbiomed.common.utils import ROOT_DIR
from fedbiomed.common.message import Message, InnerMessage, InnerRequestReply, \
    NodeMessages, NodeToNodeMessages
from fedbiomed.common.constants import ErrorNumbers, TIMEOUT_NODE_TO_NODE_REQUEST
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.secagg import DHKey
from fedbiomed.common.singleton import SingletonMeta
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.synchro import EventWaitExchange

from fedbiomed.transport.controller import GrpcController

from fedbiomed.node.environ import environ

_DEFAULT_KEY_DIR = os.path.join(ROOT_DIR, 'envs', 'common', 'default_keys')
_DEFAULT_N2N_KEY_FILE = 'default_n2n_key.pem'

# chunk size must be less or equal (in bits) the smallest RSA key length used
_SMALLEST_KEY = 2048
_CHUNK_SIZE = int(_SMALLEST_KEY / 8)


# Mimic `common.secagg.DHKey` to ease future migration to symmetric encryption


class DHKeyRSA(DHKey):
    """Temporarily adapts the key management for using RSA asymmetric encryption

    Currently cannot use elliptic curve crypto `DHkey` with asymetric encryption.

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
    local_key: DHKeyRSA
    distant_key: Optional[DHKeyRSA] = None
    pending: bool = False


class Overlay(metaclass=SingletonMeta):
    """Provides thread safe layer for sending and receiving overlay messages.

    Attributes:
        _channel_keys: (Dict[str, _N2nKeysEntry]) key status for node to node channels
        _pending_channel_keys: (EventWaitExchange) synchronize for channel keys negotiation
    """

    def __init__(self, grpc_client: GrpcController, pending_requests: EventWaitExchange):
        """Class constructor

        Args:
            grpc_client: object managing the communication with other components
        """
        self._grpc_client = grpc_client
        self._pending_requests = pending_requests

        self._channel_keys = {}
        # TODO: read saved keys from DB

        self._pending_channel_keys = EventWaitExchange(remove_delivered=False)

        # Issue #1142 in "Crypto material management" will optionally replace current default key published with the library
        # and used for each node for setup by a keypair generated securely for each node.
        # Caveat: though encrypted, current implementation does not ensure a secure overlay node2node channel ...

        # Default keys
        default_private_key, default_public_key = self._load_default_n2n_key()
        self._default_n2n_key = _N2nKeysEntry(
            local_key=default_private_key,
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
    def _load_default_n2n_key() -> Tuple[DHKeyRSA, DHKeyRSA]:
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
        private_key = DHKeyRSA(private_key_pem=private_key_pem)
        public_key = DHKeyRSA(public_key_pem=private_key.public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo
        ))
        return private_key, public_key

    def _setup_use_channel_keys(self, distant_node_id: str) -> Tuple[DHKeyRSA, DHKeyRSA]:
        """Returns channel key objects for

        If key object for local node does not yet exist, generate it.
        If key object for distant node is not yet known, request it.

        Attributes:
            distant_node: unique ID of node at the distant side of the n2n channel

        Returns:
            A tuple consisting of
                - private key object for local node
                - public key object for distant node
        """

        # TODO implement
        return self._default_n2n_key.local_key, self._default_n2n_key.distant_key


    def format_outgoing_overlay(self, message: Message) -> List[bytes]:
        """Creates an overlay message payload from an inner message.

        Serialize, crypt, sign the inner message

        Args:
            message: Inner message to send as overlay payload

        Returns:
            Payload for overlay message

        Raises:
            FedbiomedNodeToNodeError: key is too short
            FedbiomedNodeToNodeError: bad message type
        """
        # robustify from developer error (try to encapsulate a bad message type)
        if not isinstance(message, InnerMessage):
            raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: not an inner message')

        local_node_private_key, distant_node_public_key = self._setup_use_channel_keys(
            message.get_param('dest_node_id'))

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
            'message': message.get_dict(),
            'signature': local_node_private_key.private_key.sign(
                Serializer.dumps(message.get_dict()),
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
                    label=None
                )
            )
            for i in range(0, len(signed), _CHUNK_SIZE)
        ]


    def format_incoming_overlay(self, payload: List[bytes], src_node_id: str) -> InnerMessage:
        """Retrieves inner message from overlay message payload.

        Check signature, decrypt, deserialize the inner message

        Args:
            payload: Payload of overlay message.
            src_node_id: Unique ID of sender (distant) peer node

        Returns:
            Inner message retrieved from overlay payload

        Raises:
            FedbiomedNodeToNodeError: key is too short
            FedbiomedNodeToNodeError: cannot decrypt payload
            FedbiomedNodeToNodeError: bad payload format
            FedbiomedNodeToNodeError: cannot verify payload integrity
        """
        # check payload types (not yet done by message type checks, only checks it's a list)
        if not all(isinstance(p, bytes) for p in payload):
            raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: bad type for node to node payload')

        local_node_private_key, distant_node_public_key = self._setup_use_channel_keys(src_node_id)

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

        return NodeToNodeMessages.format_incoming_message(decrypted['message'])


    def send_nodes(
            self,
            researcher_id: str,
            nodes: List[str],
            messages: List[InnerMessage],
    ) -> Tuple[bool, List[Any]]:
        """Send message to some other nodes using overlay communications.

            Args:
                researcher_id: unique ID of researcher connecting the nodes
                nodes: list of node IDs of the destination nodes
                messages: list of the inner messages for the destination nodes
            Returns:
                status: True if all messages are received
                replies: List of replies from each node.
        """
        request_ids = []

        for node, message in zip(nodes, messages):
            message_overlay = NodeMessages.format_outgoing_message(
                {
                    'researcher_id': researcher_id,
                    'node_id': environ['NODE_ID'],
                    'dest_node_id': node,
                    'overlay': self.format_outgoing_overlay(message),
                    'setup': False,
                    # `salt` value is unused for now, will be used when moving to symetric encryption of overlay messages
                    # Adjust length of `salt` depending on algorithm (eg: 16 bytes for ChaCha20)
                    # secrets.token_bytes(16)
                    'salt': b'',  # returned by format_outgoing_overlay
                    'command': 'overlay',
                })

            self._grpc_client.send(message_overlay)

            if isinstance(message, InnerRequestReply):
                request_ids += [message.get_param('request_id')]

        return self._pending_requests.wait(request_ids, TIMEOUT_NODE_TO_NODE_REQUEST)


# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import os

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

from fedbiomed.common.utils import ROOT_DIR
from fedbiomed.common.message import Message, InnerMessage, InnerRequestReply, \
    NodeMessages, NodeToNodeMessages
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.serializer import Serializer

from fedbiomed.transport.controller import GrpcController

from fedbiomed.node.environ import environ
from ._pending_requests import PendingRequests


_DEFAULT_KEY_DIR = os.path.join(ROOT_DIR, 'envs', 'common', 'default_keys')
_DEFAULT_N2N_KEY_FILE = 'default_n2n_key.pem'

# TODO: replace default key published with the library and used for each node
# by a keypair generated securely for each node.
# Caveat: though encrypted, current implementation does not ensure a secure overlay node2node channel ...
#
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


def load_default_n2n_key() -> rsa.RSAPrivateKey:
    """Read default node to node private key from file.

    Currently uses the same keypair, published with library, for each node pair.
        To be replaced by a securely generated keypair per node, public key shared
        with other nodes

    Returns:
        The default private key object

    Raises:
        FedbiomedNodeToNodeError: cannot read key from file
    """
    default_key_file = os.path.join(_DEFAULT_KEY_DIR, _DEFAULT_N2N_KEY_FILE)
    try:
        with open(default_key_file, 'r') as file:
            private_key = serialization.load_pem_private_key(bytes(file.read(), 'utf-8'), None)
    except (FileNotFoundError, PermissionError, ValueError):
        raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: cannot read default node to node key '
                                       f'from file {default_key_file}')
    return private_key


_default_n2n_key = load_default_n2n_key()

# chunk size must be less or equal (in bits) the smallest RSA key length used
_SMALLEST_KEY = 2048
_CHUNK_SIZE = int(_SMALLEST_KEY / 8)


def format_outgoing_overlay(message: Message) -> List[bytes]:
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
    # robustify from developper error (try to encapsulate a bad message type)
    if not isinstance(message, InnerMessage):
        raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: not an inner message')

    # consider encrypt-sign([message,node_id]) or other see https://theworld.com/~dtd/sign_encrypt/sign_encrypt7.html

    local_node_private_key = _default_n2n_key
    distant_node_public_key = _default_n2n_key.public_key()

    if _CHUNK_SIZE * 8 > min(local_node_private_key.key_size, distant_node_public_key.key_size):
        raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: cannot use key shorter than {_CHUNK_SIZE} bits')

    # sign inner payload
    signed = Serializer.dumps({
        'message': message.get_dict(),
        'signature': local_node_private_key.sign(
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
        distant_node_public_key.encrypt(
            signed[i:i + _CHUNK_SIZE],
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        for i in range(0, len(signed), _CHUNK_SIZE)
    ]


def format_incoming_overlay(payload: List[bytes]) -> InnerMessage:
    """Retrieves inner message from overlay message payload.

    Check signature, decrypt, deserialize the inner message

    Args:
        payload: Payload of overlay message.

    Returns:
        Inner message retrieved from overlay payload

    Raises:
        FedbiomedNodeToNodeError: key is too short
        FedbiomedNodeToNodeError: cannot decrypt payload
        FedbiomedNodeToNodeError: bad payload format
        FedbiomedNodeToNodeError: cannot verify payload integrity
    """
    # decode and ensure only node2node (inner) messages are received

    local_node_private_key = _default_n2n_key
    distant_node_public_key = _default_n2n_key.public_key()

    if _CHUNK_SIZE * 8 > min(local_node_private_key.key_size, distant_node_public_key.key_size):
        raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: cannot use key shorter than {_CHUNK_SIZE} bits')

    # decrypt outer payload
    try:
        decrypted_chunks = [
            local_node_private_key.decrypt(
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
        raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: cannot decrypt payload: {e}')

    decrypted = Serializer.loads(bytes().join(decrypted_chunks))

    if not isinstance(decrypted, dict) or not set(('message', 'signature')) <= set(decrypted):
        raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: bad inner payload format '
                                       f"in received message")

    # verify inner payload
    try:
        distant_node_public_key.verify(
            decrypted['signature'],
            Serializer.dumps(decrypted['message']),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
    except ValueError as e:
        raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: cannot verify payload integrity: {e}')

    return NodeToNodeMessages.format_incoming_message(decrypted['message'])


def send_nodes(
        grpc_client: GrpcController,
        pending_requests: PendingRequests,
        researcher_id: str,
        nodes: List[str],
        messages: List[InnerMessage]) -> Optional[int]:
    """Send message to some other nodes using overlay communications.

        Args:
            grpc_client: object managing the communication with other components
            pending_requests: object for receiving overlay node to node reply messages

        Returns:
            A unique ID of type `int` for retrieving node to node reply messages for this request
            from the `pending_requests`, or `None` if no message sent to another node is of type request-reply 
    """
    request_ids = []

    for node, message in zip(nodes, messages):
        message_overlay = NodeMessages.format_outgoing_message(
            {
                'researcher_id': researcher_id,
                'node_id': environ['NODE_ID'],
                'dest_node_id': node,
                'overlay': format_outgoing_overlay(message),
                'command': 'overlay'
            })

        grpc_client.send(message_overlay)

        if isinstance(message, InnerRequestReply):
            request_ids += [message.get_param('request_id')]

    if request_ids:
        # at least one message is request-reply
        listener_id = pending_requests.add_listener(request_ids)
    else:
        listener_id = None

    return listener_id

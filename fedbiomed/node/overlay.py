# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from fedbiomed.common.message import Message, InnerMessage, NodeToNodeMessages
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.serializer import Serializer


def format_outgoing_overlay(message: Message) -> bytes:
    """Creates an overlay message payload from an inner message.

    Serialize, crypt, sign the inner message

    Args:
        message: Inner message to send as overlay payload

    Returns:
        Payload for overlay message
    """
    # robustify from developper error (try to encapsulate a bad message type)
    if not isinstance(message, InnerMessage):
        raise FedbiomedNodeToNodeError(f'{ErrorNumbers.FB324.value}: not an inner message')

    # consider encrypt-sign([message,node_id]) or other see https://theworld.com/~dtd/sign_encrypt/sign_encrypt7.html

    return Serializer.dumps(message.get_dict())


def format_incoming_overlay(payload: bytes) -> InnerMessage:
    """Retrieves inner message from overlay message payload.

    Check signature, decrypt, deserialize the inner message

    Args:
        payload: Payload of overlay message.

    Returns:
        Inner message retrieved from overlay payload
    """
    # decode and ensure only node2node (inner) messages are received
    return NodeToNodeMessages.format_incoming_message(Serializer.loads(payload))

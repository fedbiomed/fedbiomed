# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

from fedbiomed.common.message import Message, InnerMessage, InnerRequestReply, \
    NodeMessages, NodeToNodeMessages
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer

from fedbiomed.transport.controller import GrpcController

from fedbiomed.node.environ import environ
from ._pending_requests import PendingRequests


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


def send_overlay_message(
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
                'command': 'overlay-send'
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
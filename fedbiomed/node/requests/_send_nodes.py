# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Tuple

from fedbiomed.common.constants import TIMEOUT_NODE_TO_NODE_REQUEST, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.message import InnerMessage, OverlayMessage, InnerRequestReply
from fedbiomed.common.synchro import EventWaitExchange

from fedbiomed.transport.controller import GrpcController

from ._n2n_router import NodeToNodeRouter


def send_nodes(
        n2n_router: NodeToNodeRouter,
        grpc_client: GrpcController,
        pending_requests: EventWaitExchange,
        researcher_id: str,
        nodes: List[str],
        messages: List[InnerMessage],
        raise_if_not_all_received: bool = False,
) -> Tuple[bool, List[Any]]:
    """Send message to some other nodes using overlay communications and wait for their replies.

        Args:
            n2n_router: object managing node to node messages
            grpc_client: object managing the communication with other components
            pending_requests: object for receiving overlay node to node reply message
            researcher_id: unique ID of researcher connecting the nodes
            nodes: list of node IDs of the destination nodes
            messages: list of the inner messages for the destination nodes
            raise_if_not_all_received: if True, raise exception if not all answers from nodes were received.
                Default to False, return with `status` to False when not all answers from nodes were received.
        Returns:
            status: True if all messages are received
            replies: List of replies from each node.

        Raises:
            FedbiomedNodeToNodeError: not all answers received and raise_if_not_all_received is True
    """
    request_ids = []

    for node, message in zip(nodes, messages):
        overlay, salt, nonce = n2n_router.format_outgoing_overlay(message, researcher_id)
        message_overlay = OverlayMessage(
            researcher_id=researcher_id,
            node_id=n2n_router.node_id,
            dest_node_id=node,
            overlay=overlay,
            setup=False,
            salt=salt,
            nonce=nonce,
        )

        grpc_client.send(message_overlay)

        if isinstance(message, InnerRequestReply):
            request_ids += [message.get_param('request_id')]

    all_received, replies = pending_requests.wait(request_ids, TIMEOUT_NODE_TO_NODE_REQUEST)
    if not all_received and raise_if_not_all_received:
        nodes_no_answer = set(nodes) - set(m.node_id for m in replies)
        raise FedbiomedNodeToNodeError(
            f"{ErrorNumbers.FB318.value}: Some nodes did not answer request "
            f"{nodes_no_answer}"
        )

    return all_received, replies

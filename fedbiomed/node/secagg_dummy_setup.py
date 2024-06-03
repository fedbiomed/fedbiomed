# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union
import uuid

from fedbiomed.common.constants import REQUEST_PREFIX, \
    TIMEOUT_NODE_TO_NODE_REQUEST
from fedbiomed.common.message import NodeToNodeMessages, NodeMessages
from fedbiomed.common.logger import logger

from fedbiomed.transport.controller import GrpcController

from fedbiomed.node.environ import environ
from fedbiomed.node.overlay import format_outgoing_overlay
from fedbiomed.node.pending_requests import PendingRequests


# DUMMY TEST FOR OVERLAY MESSAGES
class SecaggDummySetup:
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            parties: List[str],
            experiment_id: Union[str, None],
            grpc_client: GrpcController,
            pending_requests: PendingRequests,
    ):
        self._researcher_id = researcher_id
        self._secagg_id = secagg_id
        self._experiment_id = experiment_id
        self._parties = parties
        self._grpc_client = grpc_client
        self._pending_requests = pending_requests

    def setup(self):
        other_nodes = [ e for e in self._parties[1:] if e != environ['NODE_ID'] ]
        request_ids = []

        for node in other_nodes:
            # For real use: catch FedbiomedNodeToNodeError when calling `format_outgoing_overlay`
            message_inner = NodeToNodeMessages.format_outgoing_message(
                {
                    'request_id': REQUEST_PREFIX + str(uuid.uuid4()),
                    'node_id': environ['NODE_ID'],
                    'dest_node_id': node,
                    'dummy': f"DUMMY INNER from {environ['NODE_ID']}",
                    'command': 'key-request'
                })

            message_overlay = NodeMessages.format_outgoing_message(
                {
                    'researcher_id': self._researcher_id,
                    'node_id': environ['NODE_ID'],
                    'dest_node_id': node,
                    'overlay': format_outgoing_overlay(message_inner),
                    'command': 'overlay-send'
                })

            logger.debug(f"SECAGG DUMMY: SENDING OVERLAY message to {node}: {message_overlay}")
            request_ids += [message_inner.get_param('request_id')]
            self._grpc_client.send(message_overlay)

        print(f"PENDING REQUESTS {self._pending_requests}")
        listener_id = self._pending_requests.add_listener(request_ids)
        all_received, messages = self._pending_requests.wait(listener_id, TIMEOUT_NODE_TO_NODE_REQUEST)
        logger.debug(f"SECAGG DUMMY: ALL RECEIVED ? {all_received}")
        logger.debug(f"SECAGG DUMMY: RECEIVED MESSAGES {messages}")


        return {
            'researcher_id': self._researcher_id,
            'secagg_id': self._secagg_id,
            'success': True,
            'msg': 'DUMMY DH SECAGG COMPLETED',
            'command': 'secagg'
        }

# END OF DUMMY TEST FOR OVERLAY MESSAGES

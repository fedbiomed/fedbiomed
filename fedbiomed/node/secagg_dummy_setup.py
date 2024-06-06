# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union
import uuid

from fedbiomed.common.constants import REQUEST_PREFIX, \
    TIMEOUT_NODE_TO_NODE_REQUEST
from fedbiomed.common.logger import logger

from fedbiomed.transport.controller import GrpcController

from fedbiomed.common.message import NodeToNodeMessages
from fedbiomed.node.environ import environ
from fedbiomed.node.requests._overlay import send_overlay_message
from fedbiomed.node.requests._pending_requests import PendingRequests


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

        # An example of a non request-reply message

        other_nodes_messages = []
        for node in other_nodes:
            # For real use: catch FedbiomedNodeToNodeError when calling `format_outgoing_overlay`
            other_nodes_messages += [
                NodeToNodeMessages.format_outgoing_message({
                    'node_id': environ['NODE_ID'],
                    'dest_node_id': node,
                    'dummy': f"TEST MESSAGE DUMMY FROM {environ['NODE_ID']}",
                    'command': 'dummy-inner'
                })
            ]
        listener_id = send_overlay_message(
            self._grpc_client,
            self._pending_requests,
            self._researcher_id,
            other_nodes,
            other_nodes_messages,
        )

        ## The real key request-reply
#
        #other_nodes_messages = []
        #for node in other_nodes:
        #    # For real use: catch FedbiomedNodeToNodeError when calling `format_outgoing_overlay`
        #    other_nodes_messages += [
        #        NodeToNodeMessages.format_outgoing_message({
        #            'request_id': REQUEST_PREFIX + str(uuid.uuid4()),
        #            'node_id': environ['NODE_ID'],
        #            'dest_node_id': node,
        #            'dummy': f"KEY REQUEST INNER from {environ['NODE_ID']}",
        #            'secagg_id': self._secagg_id,
        #            'command': 'key-request'
        #        })
        #    ]
        #listener_id = send_overlay_message(
        #    self._grpc_client,
        #    self._pending_requests,
        #    self._researcher_id,
        #    other_nodes,
        #    other_nodes_messages,
        #)
#
        #all_received, messages = self._pending_requests.wait(listener_id, TIMEOUT_NODE_TO_NODE_REQUEST)
        #logger.debug(f"SECAGG DUMMY: ALL RECEIVED ? {all_received}")
        #logger.debug(f"SECAGG DUMMY: RECEIVED MESSAGES {messages}")

        all_received = True

        return {
            'researcher_id': self._researcher_id,
            'secagg_id': self._secagg_id,
            'success': all_received,
            'msg': 'DUMMY DH SECAGG COMPLETED',
            'command': 'secagg'
        }

# END OF DUMMY TEST FOR OVERLAY MESSAGES

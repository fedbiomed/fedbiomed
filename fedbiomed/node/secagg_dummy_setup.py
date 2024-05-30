# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union
import uuid

from fedbiomed.common.constants import REQUEST_PREFIX
from fedbiomed.common.message import NodeToNodeMessages, NodeMessages

from fedbiomed.transport.controller import GrpcController

from fedbiomed.node.environ import environ
from fedbiomed.node.overlay import format_outgoing_overlay


# DUMMY TEST FOR OVERLAY MESSAGES
class SecaggDummySetup:
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            parties: List[str],
            experiment_id: Union[str, None],
            grpc_client: GrpcController
    ):
        self._researcher_id = researcher_id
        self._secagg_id = secagg_id
        self._experiment_id = experiment_id
        self._parties = parties
        self._grpc_client = grpc_client

    def setup(self):
        other_nodes = [ e for e in self._parties[1:] if e != environ['NODE_ID'] ]

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
            print(f"SENDING OVERLAY message to {node}: {message_overlay}")
            self._grpc_client.send(message_overlay)

        return {
            'researcher_id': self._researcher_id,
            'secagg_id': self._secagg_id,
            'success': True,
            'msg': 'DUMMY DH SECAGG COMPLETED',
            'command': 'secagg'
        }

# END OF DUMMY TEST FOR OVERLAY MESSAGES

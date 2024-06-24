# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union
import uuid

from fedbiomed.common.constants import REQUEST_PREFIX, \
    TIMEOUT_NODE_TO_NODE_REQUEST
from fedbiomed.common.logger import logger

from fedbiomed.transport.controller import GrpcController

from fedbiomed.common.message import NodeToNodeMessages
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.environ import environ
from fedbiomed.node.requests import send_overlay_message, PendingRequests


# BEGIN: TO BE REPLACED AFTER REFACTOR OF `BaseSecaggSetup`
class SecaggDHSetup:
    """Sets up a Diffie Hellman Secure Aggregation context element on the node side.
    """
    def __init__(
            self,
            researcher_id: str,
            secagg_id: str,
            parties: List[str],
            experiment_id: Union[str, None],
            grpc_client: GrpcController,
            pending_requests: PendingRequests,
    ):
        """Constructor of the class.

        Args:
            researcher_id: same as `BaseSecaggSetup`
            secagg_id: same as `BaseSecaggSetup`
            experiment_id: same as `BaseSecaggSetup`
            parties: same as `BaseSecaggSetup`
            grpc_client: object managing the communication with other components
            pending_requests: object for receiving overlay node to node messages
        """
        self._researcher_id = researcher_id
        self._secagg_id = secagg_id
        self._experiment_id = experiment_id
        self._parties = parties
        self._grpc_client = grpc_client
        self._pending_requests = pending_requests

    def _create_secagg_reply(self, message: str = '', success: bool = False) -> dict:
        """Same as `BaseSecaggSetup._create_secagg_reply()`

        TODO: use original function when refactoring `SecaggSetup`
        """

        # If round is not successful log error message
        if not success:
            logger.error(message)

        return {
            'researcher_id': self._researcher_id,
            'secagg_id': self._secagg_id,
            'success': success,
            'msg': message,
            'command': 'secagg'
        }

    def setup(self) -> dict:
        """Set up a secagg context element.

        Returns:
            message to return to the researcher after the setup
        """

        other_nodes = [ e for e in self._parties[1:] if e != environ['NODE_ID'] ]

        # Key exchange with other nodes

        other_nodes_messages = []
        for node in other_nodes:
            other_nodes_messages += [
                NodeToNodeMessages.format_outgoing_message({
                    'request_id': REQUEST_PREFIX + str(uuid.uuid4()),
                    'node_id': environ['NODE_ID'],
                    'dest_node_id': node,
                    'dummy': f"KEY REQUEST INNER from {environ['NODE_ID']}",
                    'secagg_id': self._secagg_id,
                    'command': 'key-request'
                })
            ]
        logger.debug(f'Sending Diffie-Hellman setup for {self._secagg_id} to nodes: {other_nodes}')
        try:
            listener_id = send_overlay_message(
                self._grpc_client,
                self._pending_requests,
                self._researcher_id,
                other_nodes,
                other_nodes_messages,
            )

            all_received, messages = self._pending_requests.wait(listener_id, TIMEOUT_NODE_TO_NODE_REQUEST)

        # TODO: for real use, this code will be in `BaseSecaggSetup._setup_specific()`
        # which catches FedbiomedError exceptions.
        except FedbiomedError as e:
            logger.debug(f"{e}")
            return self._create_secagg_reply(f'Can not setup secure aggregation it might be due to unregistered '
                                             f'certificate for the federated setup. Please see error: {e}', False)
        except Exception as e:
            logger.debug(f"{e}")
            return self._create_secagg_reply('Unexpected error occurred please '
                                             'report this to the node owner', False)

        logger.debug(f'Completed Diffie-Hellmann setup for {self._secagg_id}. Status: {all_received}')

        return self._create_secagg_reply(
            "Secagg element context setup for Diffie-Hellman completed", all_received
        )

# END: TO BE REPLACED AFTER REFACTOR OF `BaseSecaggSetup`

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, List

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedNodeStateAgentError


class NodeStateAgent:
    # TODO: check if it is working
    """
    Manages Node States collection, gathered from `Nodes` replies.
    """
    def __init__(self, node_ids: List[str]) -> None:
        """Constructor for NodeStateAgent.

        Initializes state ID of each node provided in `node_ids` to None (will maintain state,
        no previous node state yet)

        Args:
            node_ids: list of node IDs of the nodes for which to maintain state ID
        """
        self._collection_state_ids: Dict[str, str] = {  # Mapping <node_id, state_id>
            node_id: None for node_id in node_ids
        }

    def get_last_node_states(self) -> Dict[str, str]:
        """Returns a dictionary mapping <node_id, state_id> from latest `Nodes` replies.

        If used before the end of the first Round, each state_id is set to None

        Returns:
            Mapping of `<node_id, state_id>`
        """
        return self._collection_state_ids

    def update_node_states(self, all_node_ids: List[str], resp: Optional[Dict] = None):
        """Updates the state_id collection with respect to current nodes and latest Nodes replies.

        Adds node IDs contained in node_ids argument that was not part of the previous Round, and discards node_ids that
        do not belong to the current Round anymore.

        Args:
            all_node_ids: all possible nodes that can participate to the training.
            resp (optional): latest Nodes replies. Defaults to None.

        Raises:
            FedbiomedNodeStateAgentError: raised if Nodes replies have a missing entry that needs to be collected.
        """
        # first, we update _collection_state_id wrt new FederatedDataset (if it has been modified)
        self._update_collection_state_ids(all_node_ids)
        if resp is not None:
            for node_reply in resp.values():
                # adds Node replies
                try:
                    node_id, state_id = node_reply['node_id'], node_reply['state_id']
                except KeyError as ke:
                    raise FedbiomedNodeStateAgentError(f"{ErrorNumbers.FB419.value}: Missing entry in Response") from ke
                if node_id in self._collection_state_ids:
                    self._collection_state_ids[node_id] = state_id

    def _update_collection_state_ids(self, node_ids: List[str]):
        """Adds node_ids contained in self._data argument that was not part of the previous Round,
        and discards node_ids that do not belong to the current Round anymore.

        Args:
            node_ids: all possible nodes that can participate to the training.
        """
        for node_id in node_ids:
            if self._collection_state_ids.get(node_id, False) is False:
                self._collection_state_ids[node_id] = None
        for node_id in set(self._collection_state_ids.keys()):
            if node_id not in node_ids:
                # remove previous node_ids of collection_state_ids if _data has changed
                self._collection_state_ids.pop(node_id)

    def save_state_breakpoint(self) -> Dict:
        """NodeStateAgent's state, to be saved in a breakpoint.

        Returns:
            Node state for breakpoint
        """
        return {
            'collection_state_ids': self._collection_state_ids
        }

    def load_state_breakpoint(self, node_state: Dict):
        """Loads NodeStateAgent's state from saved state.

        Args:
            node_state: state of `NodeStateAgent` to be loaded.
        """
        self._collection_state_ids = node_state.get('collection_state_ids')

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Dict, Optional, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedNodeStateAgentError

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.responses import Responses


class NodeStateAgent:
    """
    Manages Node States collection, gathered from `Nodes` replies.
    """
    def __init__(self, fds: Optional[Union[FederatedDataSet, Dict[str, str]]] = None) -> None:
        """Constructors for NodeStateAgent. If `fds` argument has been provided, initializes all state_id of each Node
        provided in `fds` argument to None.

        Args:
            fds (optional): object that maps state_ids to node_id, and contains all possible node_ids. Defaults to None.
        """
        # NOTA: job's training_replies contains all previous Node states_id,
        # please consider using it to load previous Rounds
        self._data: Union[FederatedDataSet, Dict[str, str]] = None  # Mapping <node_id, state_id>
        self._collection_state_ids: Dict[str, str] = None  # Mapping <node_id, state_id>
        if fds is not None:
            self.set_federated_dataset(fds)

            self._initiate_collection_state_data()

    def get_last_node_states(self) -> Dict[str, str]:
        """Returns a dictionary mapping <node_id, state_id> from latest `Nodes` replies. If used before the end of 
        the first Round, each state_id is set to None

        Returns:
            Mapping of <node_id, state_id>
        """
        return self._collection_state_ids

    def update_node_states(self, fds: FederatedDataSet, resp: Optional[Responses] = None):
        """Updates the state_id collection with respect to current FederatedDataset and latest Nodes Responses.
        Adds node_ids contained in fds argument that was not part of the previous Round, and discards node_ids that 
        does not belong to the current Round anymore.

        Args:
            fds: current FederatedDataset, that contains all possible `Node` that can participate to the training.
            resp (optional): latest Nodes Responses. Defaults to None.

        Raises:
            FedbiomedNodeStateAgentError: raised if `Responses` has a missing entry that needs to be collected.
        """
        self.set_federated_dataset(fds)
        if self._collection_state_ids is None:
            self._initiate_collection_state_data()
        # first, we update _collection_state_id wrt new FedratedDataset (if it has been modified)
        self._update_collection_state_ids()
        if resp is not None:
            for node_reply in resp:
                # adds Node responses
                try:
                    node_id, state_id = node_reply['node_id'], node_reply['state_id']
                except KeyError as ke:
                    raise FedbiomedNodeStateAgentError(f"{ErrorNumbers.FB323.value}: Missing entry in Response") from ke
                if node_id in self._collection_state_ids:
                    self._collection_state_ids[node_id] = state_id

    def _update_collection_state_ids(self):
        """Adds node_ids contained in fds argument that was not part of the previous Round, and discards node_ids that 
        does not belong to the current Round anymore.
        """
        _previous_node_ids = copy.deepcopy(set(self._collection_state_ids.keys()))
        for node_id in self._data:
            if self._collection_state_ids.get(node_id, False) is False:
                self._collection_state_ids[node_id] = None
        for node_id in _previous_node_ids:
            if node_id not in self._data:
                # remove previous node_ids of collection_state_ids if _data has changed
                self._collection_state_ids.pop(node_id)

    def _initiate_collection_state_data(self):
        """Creates dcitionary that maps node_ids to node_state. When initializing, ie before starting Experiment,
        node_states are set by defaut to None.
        """
        self._collection_state_ids: Dict[str, str] = {
            node_id: None for node_id in self._data
        }

    def set_federated_dataset(self, fds: Union[FederatedDataSet, Dict[str, str]]) -> \
            Union[FederatedDataSet, Dict[str, str]]:
        """Sets a new fds that provides all possible node_id that could reply during the `Experiment`.

        Raises:
            FedbiomedNodeStateAgentError: raised if `fds` argument is neither a FederatedDataset nor a dictionary.
        """
        if isinstance(fds, FederatedDataSet):
            data: Dict[str, str] = fds.data()
        elif isinstance(fds, dict):
            data: Dict[str, str] = fds
        else:
            raise FedbiomedNodeStateAgentError(f"{ErrorNumbers.FB323.value}: fds argument should be either a "
                                               f"FederatedDataset or a dict, not a {type(fds)}")
        self._data = data

    def save_state_ids_in_bkpt(self) -> Dict[str, str]:
        """NodeStateAgent's state, to be saved in a breakpoint. 

        Returns:
            Mapping of node_ids and latest state_id
        """
        # FIXME: duplicate of get_last_node_state method
        return self._collection_state_ids

    def load_state_ids_from_bkpt(self, collection_state_ids: Optional[Dict[str, str]] = None):
        """Loads NodeStateAgent's state from collection_state_ids (which works as NodeStateAgent's state)

        Args:
            collection_state_ids (optional): state of `NodeStateAgent` to be loaded. Defaults to None. If set to None, 
                doesnot load anything.

        Raises:
            FedbiomedNodeStateAgentError: raised if `NodeStateAgent` doesnot contain any FederatedDataset.
            FedbiomedNodeStateAgentError: raised if `collection_state_ids contains `node_ids` that doesnot belong to
                                          the `NodeStateAgent` FederatedDataset.
        """
        if self._data is None:
            raise FedbiomedNodeStateAgentError(f"{ErrorNumbers.FB323.value}: can not load breakpoint, FederatedDataset"
                                               " missing. Please use `set_federated_dataset` beforehand")
        if collection_state_ids is not None:
            if set(collection_state_ids.keys()) <= set(self._data):
                self._collection_state_ids.update(collection_state_ids)
            else:
                raise FedbiomedNodeStateAgentError(f"{ErrorNumbers.FB323.value}: Error while loading breakpoints: some "
                                                   f"Node ids {set(collection_state_ids.keys()) - set(self._data)} in "
                                                   "the state agent are not present in the Federated Dataset!")

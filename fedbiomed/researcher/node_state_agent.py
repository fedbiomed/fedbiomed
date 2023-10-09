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
    Node states saving facility, 
    """
    def __init__(self, fds: Optional[Union[FederatedDataSet, Dict[str, str]]] = None) -> None:
        # NOTA: job's training_replies contains all previous Node states_id, 
        # please consider using it to load previous Rounds
        self._data: Union[FederatedDataSet, Dict[str, str]] = None  # Mapping <node_id, state_id>
        self._collection_state_ids: Dict[str, str] = None
        if fds is not None:
            self.set_federated_dataset(fds)

            self._initiate_collection_state_data()

    def get_last_node_states(self) -> Dict[str, str]:
        print("LAST STATE ID", self._collection_state_ids)
        return self._collection_state_ids

    def update_node_states(self, fds: FederatedDataSet, resp: Optional[Responses] = None):
        self.set_federated_dataset(fds)
        if self._collection_state_ids is None:
            self._initiate_collection_state_data()
        # first, we update _collection_state_id wrt new FedratedDataset (if it has been modified)
        self._update_collection_state_ids()
        if resp is not None:
            print("RESPONSES", self._data)
            for node_reply in resp:
                # adds Node responses 
                try:
                    node_id, state_id = node_reply['node_id'], node_reply['state_id']
                except KeyError as ke:
                    raise FedbiomedNodeStateAgentError(f"{ErrorNumbers.FB323.value}: Missing entry in Response") from ke
                print("STATE_ID", state_id)
                if node_id in self._collection_state_ids:
                    self._collection_state_ids[node_id] = state_id

    def _update_collection_state_ids(self):
        _previous_node_ids = copy.deepcopy(set(self._collection_state_ids.keys()))
        for node_id in self._data:
            if self._collection_state_ids.get(node_id, False) is False:
                self._collection_state_ids[node_id] = None
        for node_id in _previous_node_ids:
            if node_id not in self._data:
                # remove previous node_ids of collection_state_ids if _data has changed
                print("REMOVED", node_id, _previous_node_ids)
                self._collection_state_ids.pop(node_id)
                
    def _initiate_collection_state_data(self):

        self._collection_state_ids: Dict[str, str] = {
            node_id: None for node_id in self._data
        }

    def set_federated_dataset(self, fds: Union[FederatedDataSet, Dict[str, str]]) -> Union[FederatedDataSet, Dict[str, str]]:
        if isinstance(fds, FederatedDataSet):
            data: Dict[str, str] = fds.data()
        elif isinstance(fds, dict):
            data: Dict[str, str] = fds
        else:
            raise FedbiomedNodeStateAgentError(f"{ErrorNumbers.FB323.value}: fds argument should be either a "
                                               f"FederatedDataset or a dict, not a {type(fds)}")
        self._data = data

    def save_state_ids_in_bkpt(self) -> Dict[str, str]:
        # FIXME: duplicate of get_last_node_state method
        return self._collection_state_ids

    def load_state_ids_from_bkpt(self, collection_state_ids: Optional[Dict[str, str]] = None) -> 'NodeStateAgent':
        if collection_state_ids is not None:
            print("OADING COLLECTION STATE", set(collection_state_ids.keys()) , set(self._data))
            if set(collection_state_ids.keys()) <= set(self._data):
                self._collection_state_ids.update(collection_state_ids)
            else:
                raise FedbiomedNodeStateAgentError(f"{ErrorNumbers.FB323.value}: Error while loading breakpoints: some "
                                                   f"Node ids {set(collection_state_ids.keys()) - set(self._data)} in "
                                                   "the state agent are not present in the Federated Dataset!")

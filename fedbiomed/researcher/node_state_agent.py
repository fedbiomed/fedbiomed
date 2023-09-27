


from typing import Dict, Optional, Union
from fedbiomed.common.exceptions import FedBiomedNodeStateAgentError

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
        for node_id in self._data:
            if not self._collection_state_ids.get(node_id, False):
                self._collection_state_ids[node_id] = None
        if resp is not None:
            print("RESPONSES", self._data)
            for node_reply in resp:
                # adds Node responses 
                try:
                    node_id, state_id = node_reply['node_id'], node_reply['state_id']
                except KeyError as ke:
                    raise FedBiomedNodeStateAgentError("Missing entry in Response") from ke
                print("STATE_ID", state_id)
                self._collection_state_ids[node_id] = state_id

    def _initiate_collection_state_data(self):

        self._collection_state_ids: Dict[str, str] = {
            node_id: None for node_id in self._data
        }

    def set_federated_dataset(self, fds: Union[FederatedDataSet, Dict[str, str]]) -> Union[FederatedDataSet, Dict[str, str]]:
        if isinstance(fds, FederatedDataSet):
            self._data: Dict[str, str] = fds.data()
        elif isinstance(fds, dict):
            self._data: Dict[str, str] = fds
        else:
            raise FedBiomedNodeStateAgentError("fds argument should be either a FederatedDataset or a dict,"
                                              f" not a {type(fds)}")
            
    def save_state_ids_in_bkpt(self) -> Dict[str, str]:
        # FIXME: duplicate of get_last_node_state method
        return self._collection_state_ids

    def load_state_ids_from_bkpt(self, collection_state_ids: Optional[Dict[str, str]] = None) -> 'NodeStateAgent':
        if collection_state_ids is not None:
            print("OADING COLLECTION STATE", set(collection_state_ids.keys()) , set(self._data))
            if set(collection_state_ids.keys()) <= set(self._data):
                self._collection_state_ids.update(collection_state_ids)
            else:
                raise FedBiomedNodeStateAgentError("Error while loading breakpoints: some Node ids "
                                                   f"{set(self._data) - set(collection_state_ids.keys())} in the state"
                                                   " agent are not present in the Federated Dataset!")

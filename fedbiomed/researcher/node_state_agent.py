


from typing import Dict, Optional

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.responses import Responses


class NodeStateAgent:
    def __init__(self, fds: FederatedDataSet) -> None:
        self._data = fds.data()
        self._initiate_collection_state_data()
        
    
    def get_last_node_states(self) -> Dict[str, str]:
        return self._collection_state_ids

    def update_node_states(self, fds: FederatedDataSet, resp: Optional[Responses] = None):
        self._data = fds.data()
        self._initiate_collection_state_data()
        if resp is not None:
            for node_reply in self._data:
                node_id, state_id = node_reply['node_id'], node_reply.get('state_id')
                self._collection_state_ids[node_id] = state_id

    def _initiate_collection_state_data(self):
        self._collection_state_ids: Dict[str, str] = {
            node_id: None for node_id in self._data
        }

    def save_state_ids_in_bkpt(self):
        pass
    
    def load_state_ids_from_bkpt(self):
        pass

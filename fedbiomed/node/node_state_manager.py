from typing import Dict, Optional
from tinydb import TinyDB, Query
from tinydb.table import Table

from fedbiomed.common.utils import raise_for_version_compatibility, __default_version__
from fedbiomed.common.constants import ComponentType, ErrorNumbers, __node_state_version__

NODE_STATE_TABLE_NAME = "Node_states"



class NodeStateManager:
    #FIXME: should all Manager classes be inhereting from the same ManagerBase object?
    def __init__(self, db_path: str):
        if db_path is None:
            pass  # raise error here
        self._query: Query = Query()
        self._db: Table = TinyDB(db_path).table(NODE_STATE_TABLE_NAME) 
    
    def get(self, job_id: str, state_id: str):
        pass
    
    def add(self, job_id: str, state_id: str, state: Dict):
        pass
    
    def remove(self, job_id: Optional[str], state_id: Optional[str]):
        raise NotImplementedError
    
    def list_states(self, job_id: str):
        raise NotImplementedError

    def _load_state(self, job_id: str, state_id: str):
        pass
    
    def _save_state(self, state_entry: Dict):
        pass
    
    def _generate_new_state_id(self)-> str:
        pass
    
    def _check_version(self, version: str) -> bool:
        
        
        raise_for_version_compatibility(version, __node_state_version__,
                                        f"{ErrorNumbers.FB625.value}: Loaded a node state with version %s "
                                        f"which is incompatible with the current node state version %s")

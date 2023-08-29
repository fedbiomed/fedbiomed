from typing import Dict, Optional, Union
import uuid
from tinydb import TinyDB, Query
from tinydb.table import Table

from fedbiomed.common.utils import raise_for_version_compatibility, __default_version__
from fedbiomed.common.constants import ErrorNumbers, NODE_STATE_PREFIX, __node_state_version__
from fedbiomed.common.exceptions import FedbiomedNodeStateManager

NODE_STATE_TABLE_NAME = "Node_states"



class NodeStateManager:
    #FIXME: should all Manager classes be inhereting from the same ManagerBase object?
    def __init__(self, db_path: str):
        
            
        self._query: Query = Query()
        
        try:
            self._db: Table = TinyDB(db_path).table(name=NODE_STATE_TABLE_NAME, cache_size=0) 
        except Exception as e:
            raise FedbiomedNodeStateManager(f"{ErrorNumbers.FB323.value}: Error found when loading database") from e
    
    def get(self, job_id: str, state_id: str) -> Dict:
        state = self._load_state(job_id, state_id)
        
        if state is None:
            raise FedbiomedNodeStateManager(f"{ErrorNumbers.FB323.value}: no entries matching job_id and state_id found"
                                            " in the DataBase")
        # from this point, state should be a dictionary
        self._check_version(state.get("version_node_id"))
        return state

    def add(self, job_id: str, state: Dict) -> str:
        state_id = self._generate_new_state_id()
        header = {
            "version_node_id": str(__node_state_version__),
            "state_id": state_id,
            "job_id": job_id
        }
        
        state.update(header)
        self._save_state(state_id, state)
        return state_id
    
    def remove(self, job_id: Optional[str], state_id: Optional[str]):
        raise NotImplementedError
    
    def list_states(self, job_id: str):
        raise NotImplementedError

    def _load_state(self, job_id: str, state_id: str) -> Union[Dict, None]:
        try:
            res = self._db.get((self._query.job_id == job_id) & (self._query.state_id == state_id))
        except Exception as e:
            raise FedbiomedNodeStateManager(f"{ErrorNumbers.FB323.value}: Failing to load node state in DataBase") from e
        return res
    
    def _save_state(self, state_id: str, state_entry: Dict) -> True:
        try:
            self._db.upsert(state_entry, self._query.state_id == state_id)
        except Exception as e:
            raise FedbiomedNodeStateManager(f"{ErrorNumbers.FB323.value}: failing to"
                                            " save node state into DataBase") from e
        return True
    
    def _generate_new_state_id(self) -> str:
        state_id = NODE_STATE_PREFIX + str(uuid.uuid4())
        # TODO: would be better to check if state_id doesnot belong to the database
        return state_id
    
    def _check_first_entry_version(self):
        # assuming that this method has been run on each added entry in the database
        # TODO: ask if such method should be implemented
        pass

    def _check_version(self, version: str) -> bool:
        
        raise_for_version_compatibility(version, __node_state_version__,
                                        f"{ErrorNumbers.FB625.value}: Loaded a node state with version %s "
                                        f"which is incompatible with the current node state version %s")

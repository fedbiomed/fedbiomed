# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, Optional, Union
import uuid

from tinydb import TinyDB, Query
from tinydb.table import Table

from fedbiomed.common.utils import raise_for_version_compatibility
from fedbiomed.common.constants import (_BaseEnum, ErrorNumbers, NODE_STATE_PREFIX,
                                        __node_state_version__)
from fedbiomed.common.exceptions import FedbiomedNodeStateManagerError
from fedbiomed.common.logger import logger

from fedbiomed.node.environ import environ

NODE_STATE_TABLE_NAME = "Node_states"


class NodeStateFileName(_BaseEnum):
    """
    File names should contains 2 `%s`: one for round number, the second for state_id
    Example: VALUE = "some_values_%s_%s"
    """
    OPTIMIZER: str = "optim_state_%s_%s"


class NodeStateManager:
    # FIXME: should all Manager classes be inhereting from the same ManagerBase object?
    def __init__(self, db_path: str):

        # NOTA: constructor has been designed wrt other object handling DataBase
        self._query: Query = Query()
        self._node_state_base_dir: str = None  # node state base directory, where all node state related files are saved
        self.state_id: str = None
        self.previous_state_id: Optional[str] = None
        try:
            self._db: Table = TinyDB(db_path).table(name=NODE_STATE_TABLE_NAME, cache_size=0)
        except Exception as e:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: "
                                                 "Error found when loading database") from e

    def get(self, job_id: str, state_id: str) -> Dict:
        state = self._load_state(job_id, state_id)

        if state is None:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: no entries matching job_id "
                                                 "and state_id found in the DataBase")
        # from this point, state should be a dictionary
        self._check_version(state.get("version_node_id"))
        return state

    def add(self, job_id: str, state: Dict) -> str:
        if self.state_id is None:
            self._generate_new_state_id()
        header = {
            "version_node_id": str(__node_state_version__),
            "state_id": self.state_id,
            "job_id": job_id
        }

        state.update(header)
        self._save_state(self.state_id, state)
        return self.state_id

    def remove(self, job_id: Optional[str], state_id: Optional[str]):
        raise NotImplementedError

    def list_states(self, job_id: str):
        raise NotImplementedError

    def _load_state(self, job_id: str, state_id: str) -> Union[Dict, None]:
        try:

            res = self._db.get((self._query.job_id == job_id) & (self._query.state_id == state_id))
        except Exception as e:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: Failing to load node state "
                                                 "in DataBase") from e
        logger.debug("Successfully loaded previous state!")
        return res

    def _save_state(self, state_id: str, state_entry: Dict) -> True:
        try:
            print("REQ content", (self._query.state_id))
            self._db.upsert(state_entry, self._query.state_id == state_id)
        except Exception as e:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: failing to "
                                                 "save node state into DataBase") from e
        return True

    def initialize(self, previous_state_id: Optional[str] = None) -> True:
        self.previous_state_id = previous_state_id
        # self._generate_new_state_id()
        self._node_state_base_dir = os.path.join(environ["VAR_DIR"], "node_state_%s" % environ["NODE_ID"])
        # Should we ALWAYS create a folder when saving a state, even if the folder is empty?
        try:
            os.makedirs(self._node_state_base_dir, exist_ok=True)
        except Exception as e:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: Failing to create"
                                                 f" directories {self._node_state_base_dir}") from e
        return True

    def get_node_state_base_dir(self) -> str:
        return self._node_state_base_dir

    def generate_folder_and_create_file_name(self,
                                             job_id: str,
                                             round_nb: int,
                                             element: NodeStateFileName) -> str:

        node_state_base_dir = self.get_node_state_base_dir()
        if node_state_base_dir is None:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: working directory has not been "
                                                 "initialized, have you run `initialize` method beforehand ?")

        if self.state_id is None:
            self._generate_new_state_id()
        base_dir = os.path.join(node_state_base_dir, "job_id_%s" % job_id)
        try:
            os.makedirs(base_dir, exist_ok=True)

        except Exception as e:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: Failing to create directories "
                                                 f"{base_dir}") from e
        # TODO catch exception here
        file_name = element.value % (round_nb, self.state_id)
        return os.path.join(base_dir, file_name)

    def _generate_new_state_id(self) -> str:
        self.state_id = NODE_STATE_PREFIX + str(uuid.uuid4())
        # TODO: would be better to check if state_id doesnot belong to the database
        return self.state_id

    def _check_first_entry_version(self):
        # assuming that this method has been run on each added entry in the database
        # TODO: ask if such method should be implemented
        pass

    def _check_version(self, version: str) -> bool:

        raise_for_version_compatibility(version, __node_state_version__,
                                        f"{ErrorNumbers.FB625.value}: Loaded a node state with version %s "
                                        f"which is incompatible with the current node state version %s")

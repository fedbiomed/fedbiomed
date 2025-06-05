# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, Optional, Union, Any
import uuid

from tinydb import TinyDB, Query
from tinydb.table import Table
from fedbiomed.common.db import DBTable

from fedbiomed.common.utils import raise_for_version_compatibility
from fedbiomed.common.constants import (
    VAR_FOLDER_NAME,
    _BaseEnum,
    ErrorNumbers,
    NODE_STATE_PREFIX,
    __node_state_version__)
from fedbiomed.common.exceptions import FedbiomedNodeStateManagerError
from fedbiomed.common.logger import logger


NODE_STATE_TABLE_NAME = "Node_states"


class NodeStateFileName(_BaseEnum):
    """
    Collection of File names that composes a `Node` state.
    File names should contains 2 `%s`: one for round number, the second for state_id
    Example: VALUE = "some_values_%s_%s"
    """
    OPTIMIZER: str = "optim_state_%s_%s"


class NodeStateManager:
    """Node state saving facility: Handles saving and loading Node states from previous `Rounds`,
    given a `state_id`. `NodeStateManager` ensures that states are not reset from one `Round` to another.
    Currently a state is composed of the Optimizer state - only for
    [`DeclearnOptimizer`][fedbiomed.common.optimizers.DeclearnOptimizer],
    but it will be extended in the future with other components - such as model layers, validation dataset, ...

    Interfaces with database use to save and load Node State entries.

    Database Table that handles Node states is built with the following entries:
        - state_id
        - experiment_id
        - version_node_id: state id version, that checks if states can be used regarding FedBioMed versions.
        - state_entries: content of each entry that cmposes a Node State
    """

    def __init__(self, dir: str, node_id: str, db_path: str):
        """Constructor of the class.

        Args:
            dir: Root directory of the component
            node_id: Unique ID of the node
            db_path: path to the node state database

        Raises:
            FedbiomedNodeStateManagerError: failed to access the database

    """
        # NOTA: constructor has been designed wrt other object handling DataBase
        self._dir = dir
        self._node_id = node_id
        self._query: Query = Query()
        # node state base directory, where all node state related files are saved
        self._node_state_base_dir: Optional[str] = None
        self._state_id: Optional[str] = None
        self._previous_state_id: Optional[str] = None
        try:
            self._connection = TinyDB(db_path)
            self._connection.table_class = DBTable
            self._db: Table = self._connection.table(name=NODE_STATE_TABLE_NAME, cache_size=0)
        except Exception as e:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: "
                                                 "Error found when loading database") from e

    @property
    def state_id(self) -> Optional[str]:
        """Getter for state ID

        Returns:
            state ID, or None if not defined yet
        """
        return self._state_id

    @property
    def previous_state_id(self) -> Optional[str]:
        """Getter for previous state ID

        Returns:
            previous state ID, or None if does not exist
        """
        return self._previous_state_id

    def get(self, experiment_id: str, state_id: str) -> Dict:
        """Returns a state of an experiment on the `Node`.

        Args:
            experiment_id: the experiment for which a state is requested
            state_id: the unique identifier of the experiment

        Returns:
            dict containing the experiment state

        Raises:
            FedbiomedNodeStateManagerError: no matching state in the database
        """
        state = self._load_state(experiment_id, state_id)

        if state is None:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: no entries matching"
                                                 f" experiment_id {experiment_id} and "
                                                 f"state_id {state_id} found in the DataBase")

        # from this point, state should be a dictionary
        self._check_version(state.get("version_node_id"))
        return state

    def add(self, experiment_id: str, state: Dict[str, Dict[str, str]]) -> str:
        """Adds new `Node` State into Database.

        Args:
            experiment_id: experiment_id used to save entry in database
            state: state to be saved in database.

        Returns:
            state_id

        Raises:
            FedbiomedNodeStateManagerError: state manager is not initialized
        """

        if self._state_id is None:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: Node state manager is not initialized")
        header = {
            "version_node_id": str(__node_state_version__),
            "state_id": self._state_id,
            "experiment_id": experiment_id
        }

        state.update(header)
        self._save_state(self._state_id, state)
        return self._state_id

    def remove(self, experiment_id: Optional[str], state_id: Optional[str]):
        raise NotImplementedError

    def list_states(self, experiment_id: str):
        raise NotImplementedError

    def _load_state(self, experiment_id: str, state_id: str) -> Union[Dict, None]:
        """Loads Node state from DataBase. Directly interfaces with database request.

        Args:
            experiment_id: experiment_id from which to retrieve state
            state_id: state_id from which to retrieve state

        Raises:
            FedbiomedNodeStateManager: raised if request fails

        Returns:
            result of the request
        """
        try:

            res = self._db.get((self._query.experiment_id == experiment_id) & (self._query.state_id == state_id))
        except Exception as e:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: Failing to load node state "
                                                 "in DataBase") from e
        logger.debug("Successfully loaded previous state!")
        return res

    def _save_state(self, state_id: str, state_entry: Dict[str, Any]) -> None:
        """Saves Node state in Database. Interfaces with Database request. Issues a
        `upsert` request in Database.

        Args:
            state_id: state_id
            state_entry: dictionary that maps entries with contents in the database

        Raises:
            FedbiomedNodeStateManagerError: raised if request fails
        """
        # TODO: should we make sure `state_id` does not already exist in database ?`
        try:
            self._db.upsert(state_entry, self._query.state_id == state_id)
        except Exception as e:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: failing to "
                                                 "save node state into DataBase") from e

    def initialize(self, previous_state_id: Optional[str] = None, testing: Optional[bool] = False) -> None:
        """Initializes NodeStateManager, by creating folder that will contains Node state folders.

        Args:
            previous_state_id: state_id from previous Round, from whch to reload a Node state
            testing: only doing testing, not training
        """

        self._previous_state_id = previous_state_id
        if not testing:
            self._generate_new_state_id()

            self._node_state_base_dir = os.path.join(
                self._dir, VAR_FOLDER_NAME, "node_state_%s" % self._node_id
            )
            # Always create the base folder for saving states for this node
            try:
                os.makedirs(self._node_state_base_dir, exist_ok=True)
            except Exception as e:
                raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: Failing to create"
                                                     f" directories {self._node_state_base_dir}") from e

    def get_node_state_base_dir(self) -> Optional[str]:
        """Returns `Node` State base directory path, in which are saved Node state files and other contents

        Returns:
            path to `Node` state base directory, or None if not defined
        """
        return self._node_state_base_dir

    def generate_folder_and_create_file_name(self,
                                             experiment_id: str,
                                             round_nb: int,
                                             element: NodeStateFileName) -> str:
        """Creates folders and file name for each content (Optimizer, model, ...) that composes
        a Node State.

        Args:
            experiment_id: experiment_id
            round_nb: current Round number
            element: a NodeStateFileName object used to create specific file names. For instance,
                could be a NodeStateFileName.OPTIMIZER

        Raises:
            FedbiomedNodeStateManagerError: raised if node state manager is not initialized
            FedbiomedNodeStateManagerError: raised if folder can not be created.

        Returns:
            path to the file that corresponds to the object that needs to be saved.
        """

        node_state_base_dir = self.get_node_state_base_dir()
        if node_state_base_dir is None:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: working directory has not been "
                                                 "initialized, have you run `initialize` method beforehand ?")

        if self._state_id is None:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: Node state manager is not initialized")
        base_dir = os.path.join(node_state_base_dir, "experiment_id_%s" % experiment_id)
        try:
            os.makedirs(base_dir, exist_ok=True)

        except Exception as e:
            raise FedbiomedNodeStateManagerError(f"{ErrorNumbers.FB323.value}: Failing to create directories "
                                                 f"{base_dir}") from e
        # TODO catch exception here
        file_name = element.value % (round_nb, self._state_id)
        return os.path.join(base_dir, file_name)

    def _generate_new_state_id(self) -> str:
        """Generates randomly a state_id. Should be created at each Round, before saving Node State into Database.

        Returns:
            new state_id
        """
        self._state_id = NODE_STATE_PREFIX + str(uuid.uuid4())
        # TODO: would be better to check if state_id doesnot belong to the database
        return self._state_id

    def _check_version(self, version: str):
        """Checks that database entry version matches the current version of FedBioMed.

        Args:
            version: version found in the DataBase entries.
        """

        raise_for_version_compatibility(version, __node_state_version__,
                                        f"{ErrorNumbers.FB625.value}: Loaded a node state with version %s "
                                        f"which is incompatible with the current node state version %s")

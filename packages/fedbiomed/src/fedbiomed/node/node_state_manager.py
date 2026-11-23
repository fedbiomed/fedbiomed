# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from typing import Dict, Optional

from fedbiomed.common.constants import (
    NODE_STATE_PREFIX,
    VAR_FOLDER_NAME,
    ErrorNumbers,
    __node_state_version__,
    _BaseEnum,
)
from fedbiomed.common.db import get_database
from fedbiomed.common.exceptions import FedbiomedNodeStateManagerError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import raise_for_version_compatibility
from fedbiomed.node.db_models import NodeState


class NodeStateFileName(_BaseEnum):
    """
    Collection of File names that composes a `Node` state.
    File names should contains 2 `%s`: one for round number, the second for state_id
    Example: VALUE = "some_values_%s_%s"
    """

    OPTIMIZER: str = "optim_state_%s_%s"
    MODEL_WEIGHTS: str = "persistent_model_weights_%s_%s"
    PREPROC: str = "preproc_%s_%s"


class NodeStateManager:
    """Node state saving facility: Handles saving and loading Node states from previous `Rounds`,
    given a `state_id`. `NodeStateManager` ensures that states are not reset from one `Round` to another.

    Each public method using the database runs in one of its transactions.
    """

    # Tables this manager owns; the only ones it reads or writes
    MODELS = (NodeState,)

    def __init__(self, dir: str, node_id: str, db_path: str):
        """Constructor of the class.

        Args:
            dir: Root directory of the component
            node_id: Unique ID of the node
            db_path: Path of the node's database file, as in its config.
        """
        self._dir = dir
        self._node_id = node_id
        # node state base directory, where all node state related files are saved
        self._node_state_base_dir: Optional[str] = None
        self._state_id: Optional[str] = None
        self._previous_state_id: Optional[str] = None
        self._database = get_database(db_path)
        self._database.create_tables(self.MODELS)

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
        logger.debug(
            f"Loading node state: experiment_id={experiment_id} state_id={state_id}"
        )
        with self._database.transaction() as session:
            saved = session.get(NodeState, state_id)
            if saved is None or saved.experiment_id != experiment_id:
                raise FedbiomedNodeStateManagerError(
                    f"{ErrorNumbers.FB323.value}: no entries matching"
                    f" experiment_id {experiment_id} and "
                    f"state_id {state_id} found in the DataBase"
                )
            state = {
                **saved.state,
                "state_id": saved.state_id,
                "experiment_id": saved.experiment_id,
                "version_node_id": saved.version_node_id,
            }

        self._check_version(state["version_node_id"])
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
            raise FedbiomedNodeStateManagerError(
                f"{ErrorNumbers.FB323.value}: Node state manager is not initialized"
            )

        with self._database.transaction() as session:
            session.merge(
                NodeState(
                    state_id=self._state_id,
                    experiment_id=experiment_id,
                    version_node_id=str(__node_state_version__),
                    state=state,
                )
            )
        logger.debug(f"Node state saved to database: state_id={self._state_id}")
        return self._state_id

    def initialize(
        self, previous_state_id: Optional[str] = None, testing: Optional[bool] = False
    ) -> None:
        """Initializes NodeStateManager, by creating folder that will contains Node state folders.

        Args:
            previous_state_id: state_id from previous Round, from which to reload a Node state
            testing: only doing testing, not training

        Raises:
            FedbiomedNodeStateManagerError: raised if needed directories cannot be created
        """

        self._previous_state_id = previous_state_id
        logger.debug(
            f"Initializing node state manager: node_id={self._node_id} "
            f"previous_state_id={previous_state_id} testing={testing}"
        )
        if not testing:
            self._generate_new_state_id()

            self._node_state_base_dir = os.path.join(
                self._dir, VAR_FOLDER_NAME, "node_state_%s" % self._node_id
            )
            # Always create the base folder for saving states for this node
            try:
                os.makedirs(self._node_state_base_dir, exist_ok=True)
            except Exception as e:
                raise FedbiomedNodeStateManagerError(
                    f"{ErrorNumbers.FB323.value}: Failing to create"
                    f" directories {self._node_state_base_dir}"
                ) from e

    def get_node_state_base_dir(self) -> Optional[str]:
        """Returns `Node` State base directory path, in which are saved Node state files and other contents

        Returns:
            path to `Node` state base directory, or None if not defined
        """
        return self._node_state_base_dir

    def generate_folder_and_create_file_name(
        self, experiment_id: str, sequence_nb: int, element: NodeStateFileName
    ) -> str:
        """Creates folders and file name for each content (Optimizer, model, ...) that composes
        a Node State.

        Args:
            experiment_id: experiment_id
            sequence_nb: current Round number ou Step Number
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
            raise FedbiomedNodeStateManagerError(
                f"{ErrorNumbers.FB323.value}: working directory has not been "
                "initialized, have you run `initialize` method beforehand ?"
            )

        if self._state_id is None:
            raise FedbiomedNodeStateManagerError(
                f"{ErrorNumbers.FB323.value}: Node state manager is not initialized"
            )
        base_dir = os.path.join(node_state_base_dir, "experiment_id_%s" % experiment_id)
        try:
            os.makedirs(base_dir, exist_ok=True)

        except Exception as e:
            raise FedbiomedNodeStateManagerError(
                f"{ErrorNumbers.FB323.value}: Failing to create directories {base_dir}"
            ) from e
        # TODO catch exception here
        file_name = element.value % (sequence_nb, self._state_id)
        path = os.path.join(base_dir, file_name)
        logger.debug(
            f"Generated node state file path: experiment_id={experiment_id} "
            f"round={sequence_nb} element={getattr(element, 'name', repr(element))} path={path}"
        )
        return path

    def _generate_new_state_id(self) -> str:
        """Generates randomly a state_id. Should be created at each Round, before saving Node State into Database.

        Returns:
            new state_id
        """
        self._state_id = NODE_STATE_PREFIX + str(uuid.uuid4())
        logger.debug(f"Generated new node state id: state_id={self._state_id}")
        # TODO: would be better to check if state_id doesn't belong to the database
        return self._state_id

    def _check_version(self, version: str):
        """Checks that database entry version matches the current version of FedBioMed.

        Args:
            version: version found in the DataBase entries.
        """

        logger.debug(
            f"Checking node state version compatibility: found={version} "
            f"current={__node_state_version__}"
        )

        raise_for_version_compatibility(
            version,
            __node_state_version__,
            f"{ErrorNumbers.FB625.value}: Loaded a node state with version %s "
            f"which is incompatible with the current node state version %s",
        )

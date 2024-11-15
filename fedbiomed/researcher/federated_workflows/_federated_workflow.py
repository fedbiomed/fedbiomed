# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

""" This file defines the FederatedWorkflow class and some additional generic utility functions that can be used by
    all other workflows."""

import functools
import json
import os
import sys
import traceback
import uuid
from abc import ABC, abstractmethod
from re import findall
from typing import Any, Dict, List, TypeVar, Union, Optional, Tuple

import tabulate
from pathvalidate import sanitize_filename

from fedbiomed.common.constants import ErrorNumbers, EXPERIMENT_PREFIX, __breakpoints_version__, \
    SecureAggregationSchemes
from fedbiomed.common.exceptions import (
    FedbiomedExperimentError, FedbiomedError, FedbiomedSilentTerminationError, FedbiomedTypeError, FedbiomedValueError, \
    FedbiomedSecureAggregationError
)
from fedbiomed.common.ipython import is_ipython
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import raise_for_version_compatibility, __default_version__
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.config import config
from fedbiomed.researcher.filetools import create_exp_folder, find_breakpoint_path, choose_bkpt_file
from fedbiomed.researcher.node_state_agent import NodeStateAgent
from fedbiomed.researcher.requests import Requests
# need to import JoyeLibertSecureAggregation, LomSecureAggregation - used for load_breakpoint()
from fedbiomed.researcher.secagg import SecureAggregation, JoyeLibertSecureAggregation, LomSecureAggregation


TFederatedWorkflow = TypeVar("TFederatedWorkflow", bound='FederatedWorkflow')  # only for typing


# Exception handling at top level for researcher
def exp_exceptions(function):
    """
    Decorator for handling all exceptions in the Experiment class() :
    pretty print a message for the user, quit Experiment.
    """

    # wrap the original function catching the exceptions
    @functools.wraps(function)
    def payload(*args, **kwargs):
        code = 0

        if not os.environ.get('FEDBIOMED_DEBUG'):
            try:
                ret = function(*args, **kwargs)
            except FedbiomedSilentTerminationError:
                raise
            except SystemExit as e:
                sys.exit(e)
            except KeyboardInterrupt:
                code = 1
                print(
                    '\n--------------------',
                    'Fed-BioMed researcher stopped due to keyboard interrupt',
                    '--------------------',
                    sep=os.linesep)
                logger.critical('Fed-BioMed researcher stopped due to keyboard interrupt')
            except FedbiomedError as e:
                code = 1
                print(
                    '\n--------------------',
                    f'Fed-BioMed researcher stopped due to exception:\n{str(e)}',
                    '--------------------',
                    sep=os.linesep)
            except BaseException as e:
                code = 3
                print(
                    '\n--------------------',
                    f'Fed-BioMed researcher stopped due to unknown error:\n{str(e)}',
                    'More details in the backtrace extract below',
                    '--------------------',
                    sep=os.linesep)
                # at most 5 backtrace entries to avoid too long output
                traceback.print_exc(limit=5, file=sys.stdout)
                print('--------------------')
                logger.critical(f'Fed-BioMed stopped due to unknown error:\n{str(e)}')

            if code != 0:
                if is_ipython():
                    raise FedbiomedSilentTerminationError
                sys.exit(code)

            return ret

        return function(*args, **kwargs)

    return payload


class FederatedWorkflow(ABC):
    """
    A FederatedWorkflow is the abstract entry point for the researcher to orchestrate both local and remote operations.

    The FederatedWorkflow is an abstract base class from which the actual classes used by the researcher must inherit.
    It manages the life-cycle of:

    - the training arguments
    - secure aggregation
    - the node state agent

    Additionally, it provides the basis for the breakpoint functionality, and manages some backend functionalities such
    as the temporary directory, the experiment ID, etc...

    The attributes `training_data` and `tags` are co-dependent. Attempting to modify one of those may result
    in side effects modifying the other, according to the following rules:
    - modifying tags if training data is not None will reset the training data based on the new tags
    - modifying the training data using a FederatedDataset object or a dict will set tags to None
    """
    @exp_exceptions
    def __init__(
        self,
        tags: Optional[List[str] | str] = None,
        nodes: Optional[List[str]] = None,
        training_data: Union[FederatedDataSet, dict, None] = None,
        experimentation_folder: Union[str, None] = None,
        secagg: Union[bool, SecureAggregation] = False,
        save_breakpoints: bool = False,
        config_path: str | None = None
    ) -> None:
        """Constructor of the class.

        Args:
            tags: list of string with data tags or string with one data tag. Empty list of
                tags ([]) means any dataset is accepted, it is different from None
                (tags not set, cannot search for training_data yet).

            nodes: list of node_ids to filter the nodes to be involved in the experiment.
                Defaults to None (no filtering).

            training_data:
                * If it is a FederatedDataSet object, use this value as training_data.
                * else if it is a dict, create and use a FederatedDataSet object
                    from the dict and use this value as training_data. The dict should use
                    node ids as keys, values being list of dicts (each dict representing a
                    dataset on a node).
                * else if it is None (no training data provided)
                  - if `tags` is not None, set training_data by
                    searching for datasets with a query to the nodes using `tags` and `nodes`
                  - if `tags` is None, set training_data to None (no training_data set yet,
                    experiment is not fully initialized and cannot be launched)
                Defaults to None (query nodes for dataset if `tags` is not None, set training_data
                to None else)
            save_breakpoints: whether to save breakpoints or not after each training
                round. Breakpoints can be used for resuming a crashed experiment.

            experimentation_folder: choose a specific name for the folder
                where experimentation result files and breakpoints are stored. This
                should just contain the name for the folder not a path. The name is used
                as a subdirectory of `config.vars[EXPERIMENTS_DIR])`. Defaults to None
                (auto-choose a folder name)
                - Caveat : if using a specific name this experimentation will not be
                    automatically detected as the last experimentation by `load_breakpoint`
                - Caveat : do not use a `experimentation_folder` name finishing
                    with numbers ([0-9]+) as this would confuse the last experimentation
                    detection heuristic by `load_breakpoint`.
            secagg: whether to setup a secure aggregation context for this experiment, and
                use it to send encrypted updates from nodes to researcher.
                Defaults to `False`,
            config_name: Allows to use specific configuration for reseracher instead of default
                one. Confiuration file are kept in `{FEDBIOMED_DIR}/etc`, and a new configuration
                file will be generated if it is not existing.
        """

        if config_path:
            config.load(root=config_path)

        self.config = config
        # predefine all class variables, so no need to write try/except
        # block each time we use it
        self._fds: Optional[FederatedDataSet] = None  # dataset metadata from the full federation
        self._reqs: Requests = Requests(config=self.config)
        self._nodes_filter: Optional[List[str]] = None  # researcher-defined nodes filter
        self._tags: Optional[List[str]] = None
        self._experimentation_folder: Optional[str] = None
        self._secagg: Union[SecureAggregation, bool] = False
        self._save_breakpoints: Optional[bool] = None
        self._node_state_agent: Optional[NodeStateAgent] = None
        self._researcher_id: str = config.get('default', 'id')
        self._experiment_id: str = EXPERIMENT_PREFIX + str(uuid.uuid4())  # creating a unique experiment id

        # set internal members from constructor arguments
        self.set_secagg(secagg)

        # TODO: Manage tags within the FederatedDataset to avoid conflicts
        if training_data is not None and tags is not None:
            msg = f"{ErrorNumbers.FB410.value}: Can not set `training_data` and `tags` at the " \
                "same time. Please provide only `training_data`, or tags to search for " \
                "training data."
            logger.critical(msg)
            raise FedbiomedValueError(msg)

        # Set tags if it tags is not None
        if tags:
            self.set_tags(tags)

        if training_data:
            self.set_training_data(training_data)

        self.set_nodes(nodes)
        self.set_save_breakpoints(save_breakpoints)

        self.set_experimentation_folder(experimentation_folder)
        self._node_state_agent = NodeStateAgent(list(self._fds.data().keys())
                                                if self._fds and self._fds.data() else [])

    @property
    def requests(self) -> Requests:
        """Returns requests object"""
        return self._reqs

    @property
    def researcher_id(self) -> str:
        """Returns researcher id"""
        return self._researcher_id

    @property
    def secagg(self) -> SecureAggregation:
        """Gets secagg object `SecureAggregation`

        Returns:
            Secure aggregation object.
        """
        return self._secagg

    @exp_exceptions
    def tags(self) -> Union[List[str], None]:
        """Retrieves the tags from the experiment object.

        Please see [`set_tags`][fedbiomed.researcher.federated_workflows.FederatedWorkflow.set_tags] to set tags.

        Returns:
            List of tags that has been set. `None` if it isn't declare yet.
        """
        return self._tags

    @exp_exceptions
    def nodes(self) -> Union[List[str], None]:
        """Retrieves the nodes filter for the execution of the workflow.

        If nodes is None, then no filtering is applied, and all the nodes in the federation participate in the
        execution of the workflow.
        If nodes is not None, then the semantics of the nodes filter are as follows:

        | node_id in nodes filter | node_id in training data | outcome |
        | --- | --- | --- |
        | yes | yes | this node is part of the federation, and will take part in the execution the workflow |
        | yes | no | ignored |
        | no | yes | this node is part of the federation but will not be considered for executing the workflow |
        | no | no | ignored |

        Please see [`set_nodes`][fedbiomed.researcher.federated_workflows.FederatedWorkflow.set_nodes] to set `nodes`.

        Returns:
            The list of nodes to keep for workflow execution, or None if no filtering is applied
        """
        return self._nodes_filter

    @exp_exceptions
    def all_federation_nodes(self) -> List[str]:
        """Returns all the node ids in the federation"""
        return list(self._fds.data().keys()) if self._fds is not None else []

    @exp_exceptions
    def filtered_federation_nodes(self) -> List[str]:
        """Returns the node ids in the federation after filtering with the nodes filter"""
        if self._nodes_filter is not None:
            return [node for node in self.all_federation_nodes() if node in self._nodes_filter]
        else:
            return self.all_federation_nodes()

    @exp_exceptions
    def training_data(self) -> Union[FederatedDataSet, None]:
        """Retrieves the training data which is an instance of
        [`FederatedDataset`][fedbiomed.researcher.datasets.FederatedDataSet]

        This represents the dataset metadata available for the full federation.

        Please see [`set_training_data`][fedbiomed.researcher.federated_workflows.FederatedWorkflow.set_training_data]
        to set or update training data.

        Returns:
            Object that contains metadata for the datasets of each node. `None` if it isn't set yet.
        """
        return self._fds

    @exp_exceptions
    def experimentation_folder(self) -> str:
        """Retrieves the folder name where experiment data/result are saved.

        Please see also [`set_experimentation_folder`]
        [fedbiomed.researcher.federated_workflows.FederatedWorkflow.set_experimentation_folder]

        Returns:
            File name where experiment related files are saved
        """

        return self._experimentation_folder

    @exp_exceptions
    def experimentation_path(self) -> str:
        """Retrieves the file path where experimentation folder is located and experiment related files are saved.

        Returns:
            Experiment directory where all experiment related files are saved
        """

        return os.path.join(config.vars['EXPERIMENTS_DIR'], self._experimentation_folder)

    @property
    def id(self):
        """Retrieves the unique experiment identifier."""
        return self._experiment_id

    @exp_exceptions
    def save_breakpoints(self) -> bool:
        """Retrieves the status of saving breakpoint after each round of training.

        Returns:
            `True`, If saving breakpoint is active. `False`, vice versa.
        """

        return self._save_breakpoints

    @staticmethod
    def _create_default_info_structure() -> Dict[str, List]:
        """Initializes info variable

        Returns:
            dictionary containing all pieces of information, with 2 entries: `Arguments` and `Values`,
              both mapping an empty list.

        """

        return {
            'Arguments': [],
            'Values': []
        }

    @exp_exceptions
    def info(self,
             info: Dict[str, List[str]] = None,
             missing: str = '') -> Tuple[Dict[str, List[str]], str]:
        """Prints out the information about the current status of the experiment.

        Lists  all the parameters/arguments of the experiment and informs whether the experiment can be run.

        Args:
            info: Dictionary of sub-classes relevant attributes status that will be completed with some additional
                attributes status defined in this class. Defaults to None (no entries of sub-classes available or
                of importance).
            missing_object_to_check: dictionary mapping sub-classes attributes to attribute names, that may be
                needed to fully run the object. Defaults to None (no check will be performed).

        Returns:
            dictionary containing all pieces of information, with 2 entries: `Arguments` mapping a list
            of all argument, and `Values` mapping a list containing all the values.
        """
        if info is None:
            info = self._create_default_info_structure()
        info['Arguments'].extend([
            'Tags',
            'Nodes filter',
            'Training Data',
            'Experiment folder',
            'Experiment Path',
            'Secure Aggregation'
        ])

        info['Values'].extend(['\n'.join(findall('.{1,60}', str(e))) for e in [
            self._tags,
            self._nodes_filter,
            self._fds,
            self._experimentation_folder,
            self.experimentation_path(),
            f'- Using: {self._secagg}\n- Active: {self._secagg.active}'
        ]])

        # printing list of items set / not set yet
        print(tabulate.tabulate(info, headers='keys'))

        if missing:
            print("\nWarning: Object not fully defined, missing"
                  f": \n{missing}")
        else:
            print(f"{self.__class__.__name__} can be run now (fully defined)")
        return info, missing

    def _check_missing_objects(self, missing_objects: Optional[Dict[Any, str]] = None) -> str:
        """Checks if some objects required for running the `run` method are not set.

        Args:
            missing_objects: dictionary mapping a string of character
                naming the required object with the value of the corresponding object
        """
        # definitions found missing

        # definitions that may be missing for running the fedreated workflow
        # (value None == not defined yet for _fds,)

        _not_runable_if_missing = {
            'Training Data': self._fds,
        }

        if missing_objects:
            _not_runable_if_missing.update(missing_objects)
        missing: str = ''
        for key, value in _not_runable_if_missing.items():
            if value in (None, False):
                missing += f'- {key}\n'

        return missing

    # Setters
    @exp_exceptions
    def set_tags(
        self,
        tags: Union[List[str], str],
    ) -> List[str]:
        """Sets tags and verification on argument type

        Setting tags updates also training data by executing
        [`set_training_data`].[fedbiomed.researcher.federated_workflows.FederatedWorkflow.set_training_data]
        method.

        Args:
            tags: List of string with data tags or string with one data tag. Empty list
                of tags ([]) means any dataset is accepted, it is different from None
                (tags not set, cannot search for training_data yet).
        Returns:
            List of tags that are set.

        Raises:
            FedbiomedTypeError: Bad tags type
            FedbiomedValueError: Some issue prevented resetting the training
                data after an inconsistency was detected
        """
        # preprocess the tags argument to correct typing
        if not tags:
            msg = f"{ErrorNumbers.FB410.value}: Invalid value for tags argument {tags}, tags " \
                "should be non-empty list of str or non-empty str."
            logger.critical(msg)
            raise FedbiomedValueError(msg)

        if isinstance(tags, list):
            if not all(map(lambda tag: isinstance(tag, str), tags)):
                msg = f"{ErrorNumbers.FB410.value}: `tags` must be a non-empty str or " \
                    "a non-empty list of str."
                logger.critical(msg)
                raise FedbiomedTypeError(msg)

            # If it is empty list
            tags_to_set = tags

        elif isinstance(tags, str):
            tags_to_set = [tags]
        else:
            msg = f"{ErrorNumbers.FB410.value} `tags` must be a non-empty str, " \
                "a non-empty list of str"
            logger.critical(msg)
            raise FedbiomedTypeError(msg)

        self._tags = tags_to_set

        # Set training data
        logger.info(
            "Updating training data. This action will update FederatedDataset, "
            "and the nodes that will participate to the experiment.")

        self.set_training_data(None, from_tags=True)

        return self._tags

    @exp_exceptions
    def set_nodes(self, nodes: Union[List[str], None]) -> Union[List[str], None]:
        """Sets the nodes filter + verifications on argument type

        Args:
            nodes: List of node_ids to filter the nodes to be involved in the experiment.

        Returns:
            List of nodes that are set. None, if the argument `nodes` is None.

        Raises:
            FedbiomedTypeError : Bad nodes type
        """
        # immediately exit if setting nodes to None
        if nodes is None:
            self._nodes_filter = None
        # set nodes
        elif isinstance(nodes, list):
            if not all(map(lambda node: isinstance(node, str), nodes)):
                msg = ErrorNumbers.FB410.value + ' `nodes` argument must be a list of strings or None.'
                logger.critical(msg)
                raise FedbiomedTypeError(msg)
            self._nodes_filter = nodes
        else:
            msg = ErrorNumbers.FB410.value + ' `nodes` argument must be a list of strings or None.'
            logger.critical(msg)
            raise FedbiomedTypeError(msg)
        return self._nodes_filter

    @exp_exceptions
    def set_training_data(
            self,
            training_data: Union[FederatedDataSet, dict, None],
            from_tags: bool = False) -> \
            Union[FederatedDataSet, None]:
        """Sets training data for federated training + verification on arguments type


        The full expected behaviour when changing training data is given in the table below:

        | New value of `training_data` | `from_tags` | Outcome |
        | --- | --- | --- |
        | dict or FederatedDataset | True  | fail because user is attempting to set from tags but also providing a training_data argument|
        | dict or FederatedDataset | False | set fds attribute, set tags to None |
        | None | True | fail if tags are not set, else set fds attribute based tags |
        | None | False | set tags to None and keep same value and tags |

        !!! warning "Setting to None forfeits consistency checks"
            Setting training_data to None does not trigger consistency checks, and may therefore leave the class in an
            inconsistent state.

        Args:
            training_data:
                * If it is a FederatedDataSet object, use this value as training_data.
                * else if it is a dict, create and use a FederatedDataSet object from the dict
                  and use this value as training_data. The dict should use node ids as keys,
                  values being list of dicts (each dict representing a dataset on a node).
                * else if it is None (no training data provided)
                  - if `from_tags` is True and `tags` is not None, set training_data by
                    searching for datasets with a query to the nodes using `tags` and `nodes`
                  - if `from_tags` is False or `tags` is None, set training_data to None (no training_data set yet,
                    experiment is not fully initialized and cannot be launched)
            from_tags: If True, query nodes for datasets when no `training_data` is provided.
                Not used when `training_data` is provided.

        Returns:
            FederatedDataSet metadata

        Raises:
            FedbiomedTypeError: bad training_data or from_tags type.
            FedbiomedValueError: Invalid value for the arguments  `training_data` or `from_tags`.
        """

        if not isinstance(from_tags, bool):
            msg = ErrorNumbers.FB410.value + \
                f' `from_tags` : got {type(from_tags)} but expected a boolean'
            logger.critical(msg)
            raise FedbiomedTypeError(msg)
        if from_tags and training_data is not None:
            msg = ErrorNumbers.FB410.value + \
                ' set_training_data: cannot specify a training_data argument if ' \
                'from_tags is True'
            logger.critical(msg)
            raise FedbiomedValueError(msg)

        # case where no training data are passed
        if training_data is None:
            if from_tags is True:
                if not self._tags:
                    msg = f"{ErrorNumbers.FB410.value}: attempting to " \
                        "set training data from undefined tags. Please consider set tags before " \
                        "using set_tags method of the experiment."
                    logger.critical(msg)
                    raise FedbiomedValueError(msg)
                training_data = self._reqs.search(self._tags, self._nodes_filter)
            else:
                msg = f"{ErrorNumbers.FB410.value}: Can not set training data to `None`. " \
                    "Please set from_tags=True or provide a valid training data"
                logger.critical(msg)
                raise FedbiomedValueError(msg)

        if isinstance(training_data, FederatedDataSet):
            self._fds = training_data
        elif isinstance(training_data, dict):
            self._fds = FederatedDataSet(training_data)
        else:
            msg = ErrorNumbers.FB410.value + \
                f' `training_data` has incorrect type: {type(training_data)}'
            logger.critical(msg)
            raise FedbiomedTypeError(msg)

        # check and ensure consistency
        self._tags = self._tags if from_tags else None

        # return the new value
        return self._fds

    @exp_exceptions
    def set_experimentation_folder(self, experimentation_folder: Optional[str] = None) -> str:
        """Sets `experimentation_folder`, the folder name where experiment data/result are saved.

        Args:
            experimentation_folder: File name where experiment related files are saved

        Returns:
            The path to experimentation folder.

        Raises:
            FedbiomedExperimentError : bad `experimentation_folder` type
        """
        if experimentation_folder is None:
            self._experimentation_folder = create_exp_folder(
                self.config.vars["EXPERIMENTS_DIR"]
            )
        elif isinstance(experimentation_folder, str):
            sanitized_folder = sanitize_filename(experimentation_folder, platform='auto')
            self._experimentation_folder = create_exp_folder(
                self.config.vars["EXPERIMENTS_DIR"], sanitized_folder
            )
            if sanitized_folder != experimentation_folder:
                logger.warning(f'`experimentation_folder` was sanitized from '
                               f'{experimentation_folder} to {sanitized_folder}')
        else:
            msg = ErrorNumbers.FB410.value + \
                f' `experimentation_folder` : {type(experimentation_folder)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

            # at this point self._experimentation_folder is a str valid for a foldername

        return self._experimentation_folder

    @exp_exceptions
    def set_secagg(
            self,
            secagg: Union[bool, SecureAggregation],
            scheme: SecureAggregationSchemes = SecureAggregationSchemes.LOM):
        """Sets secure aggregation status and scheme

        Build secure aggregation controller/instance or sets given
        secure aggregation class

        Args:
            secagg: If True activates training request with secure aggregation by building
                [`SecureAggregation`][fedbiomed.researcher.secagg.SecureAggregation] class
                with default arguments. Or if argument is an instance of `SecureAggregation`
                it does only assignment. Secure aggregation activation and configuration
                depends on the instance provided.
            scheme: Secure aggregation scheme to use. Ig a `SecureAggregation` object is provided,
                the argument is not used, as the scheme comes from the object. Defaults is
                SecureAggregationSchemes.LOM.

        Returns:
            Secure aggregation controller instance.

        Raises:
            FedbiomedExperimentError: bad argument type or value
        """
        if not isinstance(scheme, SecureAggregationSchemes):
            raise FedbiomedExperimentError(
                f"{ErrorNumbers.FB410.value}: Expected `scheme` argument "
                "`SecureAggregationSchemes`, but got {type(scheme)}")

        if isinstance(secagg, bool):
            self._secagg = SecureAggregation(
                scheme=scheme, active=secagg
            )
        elif isinstance(secagg, SecureAggregation):
            self._secagg = secagg
        else:
            raise FedbiomedExperimentError(
                f"{ErrorNumbers.FB410.value}: Expected `secagg` argument bool or "
                f"`SecureAggregation` but got {type(secagg)}")

        return self._secagg

    @exp_exceptions
    def set_save_breakpoints(self, save_breakpoints: bool) -> bool:
        """ Setter for save_breakpoints + verification on arguments type

        Args:
            save_breakpoints (bool): whether to save breakpoints or
                not after each training round. Breakpoints can be used for resuming
                a crashed experiment.

        Returns:
            Status of saving breakpoints

        Raises:
            FedbiomedExperimentError: bad save_breakpoints type
        """
        if isinstance(save_breakpoints, bool):
            self._save_breakpoints = save_breakpoints
            # no warning if done during experiment, we may change breakpoint policy at any time
        else:
            msg = ErrorNumbers.FB410.value + f' `save_breakpoints` : {type(save_breakpoints)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        return self._save_breakpoints

    def secagg_setup(self, sampled_nodes: List[str]) -> Dict:
        """Retrieves the secagg arguments for setup."""
        secagg_arguments = {}
        if self._secagg.active:  # type: ignore
            if not self._secagg.setup(  # type: ignore
                parties=sampled_nodes,
                experiment_id=self._experiment_id,
                researcher_id=self._researcher_id,
                insecure_validation=self.config.getbool('security', 'secagg_insecure_validation')
            ):
                raise FedbiomedSecureAggregationError(
                    f"{ErrorNumbers.FB417.value}: Could not setup secure aggregation crypto "
                    "context."
                )
            secagg_arguments = self._secagg.train_arguments()  # type: ignore
        return secagg_arguments

    @exp_exceptions
    def breakpoint(self,
                   state: Dict,
                   bkpt_number: int) -> None:
        """
        Saves breakpoint with the state of the workflow.

        The following attributes will be saved:

          - tags
          - experimentation_folder
          - training_data
          - training_args
          - secagg
          - node_state

        Raises:
            FedbiomedExperimentError: experiment not fully defined, experiment did not run any round
                yet, or error when saving breakpoint
        """
        state.update({
            'id': self._experiment_id,
            'breakpoint_version': str(__breakpoints_version__),
            'training_data': self._fds.data(),
            'experimentation_folder': self._experimentation_folder,
            'tags': self._tags,
            'nodes': self._nodes_filter,
            'secagg': self._secagg.save_state_breakpoint(),
            'node_state': self._node_state_agent.save_state_breakpoint()
        })

        # save state into a json file
        breakpoint_path, breakpoint_file_name = choose_bkpt_file(
            self.config.vars["EXPERIMENTS_DIR"],
            self._experimentation_folder,
            bkpt_number - 1
        )
        breakpoint_file_path = os.path.join(breakpoint_path, breakpoint_file_name)
        try:
            with open(breakpoint_file_path, 'w', encoding="UTF-8") as bkpt:
                json.dump(state, bkpt)
            logger.info(f"breakpoint number {bkpt_number - 1} saved at " +
                        os.path.dirname(breakpoint_file_path))
        except (OSError, PermissionError, ValueError, TypeError, RecursionError) as e:
            # - OSError: heuristic for catching open() and write() errors
            # - see json.dump() documentation for documented errors for this call
            msg = ErrorNumbers.FB413.value + f' - save failed with message {str(e)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg) from e

    @classmethod
    @exp_exceptions
    def load_breakpoint(
        cls,
        breakpoint_folder_path: Optional[str] = None
    ) -> Tuple[TFederatedWorkflow, dict]:
        """
        Loads breakpoint (provided a breakpoint has been saved)
        so the workflow can be resumed.

        Args:
          breakpoint_folder_path: path of the breakpoint folder. Path can be absolute
            or relative eg: "var/experiments/Experiment_xxxx/breakpoints_xxxx".
            If None, loads the latest breakpoint of the latest workflow. Defaults to None.

        Returns:
            Tuple contaning reinitialized workflow object and the saved state as a dictionary

        Raises:
            FedbiomedExperimentError: bad argument type, error when reading breakpoint or
                bad loaded breakpoint content (corrupted)
        """
        # check parameters type
        if not isinstance(breakpoint_folder_path, str) and breakpoint_folder_path is not None:
            msg = (
                f"{ErrorNumbers.FB413.value}: load failed, `breakpoint_folder_path`"
                f" has bad type {type(breakpoint_folder_path)}"
            )
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # get breakpoint folder path (if it is None) and state file
        breakpoint_folder_path, state_file = find_breakpoint_path(
            config.vars['EXPERIMENTS_DIR'],
            breakpoint_folder_path
        )
        breakpoint_folder_path = os.path.abspath(breakpoint_folder_path)

        try:
            path = os.path.join(breakpoint_folder_path, state_file)
            with open(path, "r", encoding="utf-8") as file:
                saved_state = json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            # OSError: heuristic for catching file access issues
            msg = (
                f"{ErrorNumbers.FB413.value}: load failed,"
                f" reading breakpoint file failed with message {exc}"
            )
            logger.critical(msg)
            raise FedbiomedExperimentError(msg) from exc
        if not isinstance(saved_state, dict):
            msg = (
                f"{ErrorNumbers.FB413.value}: load failed, breakpoint file seems"
                f" corrupted. Type should be `dict` not {type(saved_state)}"
            )
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # First, check version of breakpoints
        bkpt_version = saved_state.get('breakpoint_version', __default_version__)
        raise_for_version_compatibility(bkpt_version, __breakpoints_version__,
                                        f"{ErrorNumbers.FB413.value}: Breakpoint "
                                        "file was generated with version %s "
                                        f"which is incompatible with the current version %s.")

        # retrieve breakpoint training data
        bkpt_fds = saved_state.get('training_data')
        bkpt_fds = FederatedDataSet(bkpt_fds)

        # initializing experiment
        loaded_exp = cls()
        loaded_exp._experiment_id = saved_state.get('id')
        loaded_exp.set_training_data(bkpt_fds)
        loaded_exp._tags = saved_state.get('tags')
        loaded_exp.set_nodes(saved_state.get('nodes'))
        loaded_exp.set_experimentation_folder(saved_state.get('experimentation_folder'))

        secagg = SecureAggregation.load_state_breakpoint(saved_state.get('secagg'))
        loaded_exp.set_secagg(secagg)
        loaded_exp._node_state_agent.load_state_breakpoint(saved_state.get('node_state'))
        loaded_exp.set_save_breakpoints(True)

        return loaded_exp, saved_state

    @abstractmethod
    def run(self) -> int:
        """Run the experiment"""

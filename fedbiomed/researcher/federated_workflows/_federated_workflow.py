# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import functools, os, sys, tabulate, traceback, uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from pathvalidate import sanitize_filename
from re import findall
from typing import Any, Dict, List, TypeVar, Union

from fedbiomed.common.constants import ErrorNumbers, JOB_PREFIX
from fedbiomed.common.exceptions import FedbiomedExperimentError, FedbiomedError, FedbiomedSilentTerminationError
from fedbiomed.common.logger import logger
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.utils import is_ipython

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.filetools import create_exp_folder
from fedbiomed.researcher.node_state_agent import NodeStateAgent
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.secagg import SecureAggregation

TFederatedWorkflow = TypeVar("TFederatedWorkflow", bound='FederatedWorkflow')  # only for typing


# Exception handling at top lever for researcher
def exp_exceptions(function):
    """
    Decorator for handling all exceptions in the Experiment class() :
    pretty print a message for the user, quit Experiment.
    """

    # wrap the original function catching the exceptions
    @functools.wraps(function)
    def payload(*args, **kwargs):
        code = 0
        try:
            ret = function(*args, **kwargs)
        except FedbiomedSilentTerminationError:
            # handle the case of nested calls will exception decorator
            raise
        except SystemExit as e:
            # handle the sys.exit() from other clauses
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
            # redundant, should be already logged when raising exception
            # logger.critical(f'Fed-BioMed researcher stopped due to exception:\n{str(e)}')
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
                # raise a silent specific exception, don't exit the interactive kernel
                raise FedbiomedSilentTerminationError
            else:
                # exit the process
                sys.exit(code)

        return ret

    return payload


class FederatedWorkflow(ABC):
    """
    This class represents the orchestrator managing the federated training
    """

    @exp_exceptions
    def __init__(
            self,
            tags: Union[List[str], str, None] = None,
            nodes: Union[List[str], None] = None,
            training_data: Union[FederatedDataSet, dict, None] = None,
            training_args: Union[TrainingArgs, dict, None] = None,
            experimentation_folder: Union[str, None] = None,
            secagg: Union[bool, SecureAggregation] = False,
    ) -> None:
        """Constructor of the class.

        Args:
            tags: list of string with data tags or string with one data tag. Empty list of tags ([]) means any dataset
                is accepted, it is different from None (tags not set, cannot search for training_data yet).
            nodes: list of node_ids to filter the nodes to be involved in the experiment. Defaults to None (no
                filtering).
            training_data:
                * If it is a FederatedDataSet object, use this value as training_data.
                * else if it is a dict, create and use a FederatedDataSet object from the dict and use this value as
                    training_data. The dict should use node ids as keys, values being list of dicts (each dict
                    representing a dataset on a node).
                * else if it is None (no training data provided)
                  - if `tags` is not None, set training_data by
                    searching for datasets with a query to the nodes using `tags` and `nodes`
                  - if `tags` is None, set training_data to None (no training_data set yet,
                    experiment is not fully initialized and cannot be launched)
                Defaults to None (query nodes for dataset if `tags` is not None, set training_data
                to None else)
            aggregator: object or class defining the method for aggregating local updates. Default to None (use
                [`FedAverage`][fedbiomed.researcher.aggregators.FedAverage] for aggregation)
            agg_optimizer: [`Optimizer`][fedbiomed.common.optimizers.Optimizer] instance, to refine aggregated
                model updates prior to their application. If None, merely apply the aggregated updates.
            node_selection_strategy:object or class defining how nodes are sampled at each round for training, and how
                non-responding nodes are managed.  Defaults to None:
                - use [`DefaultStrategy`][fedbiomed.researcher.strategies.DefaultStrategy] if training_data is
                    initialized
                - else strategy is None (cannot be initialized), experiment cannot be launched yet
            round_limit: the maximum number of training rounds (nodes <-> central server) that should be executed for
                the experiment. `None` means that no limit is defined. Defaults to None.
            training_plan_class: name of the training plan class [`str`][str] or training plan class
                (`Type_TrainingPlan`) to use for training.
                For experiment to be properly and fully defined `training_plan_class` needs to be:
                - a [`str`][str] when `training_plan_class_path` is not None (training plan class comes from a file).
                - a `Type_TrainingPlan` when `training_plan_class_path` is None (training plan class passed
                    as argument).
                Defaults to None (no training plan class defined yet)

            training_plan_path: path to a file containing training plan code [`str`][str] or None (no file containing
                training plan code, `training_plan` needs to be a class matching `Type_TrainingPlan`) Defaults to None.
            model_args: contains model arguments passed to the constructor of the training plan when instantiating it :
                output and input feature dimension, etc.
            training_args: contains training arguments passed to the `training_routine` of the training plan when
                launching it: lr, epochs, batch_size...
            save_breakpoints: whether to save breakpoints or not after each training round. Breakpoints can be used for
                resuming a crashed experiment.
            tensorboard: whether to save scalar values  for displaying in Tensorboard during training for each node.
                Currently, it is only used for loss values.
                - If it is true, monitor instantiates a `Monitor` object that write scalar logs into `./runs` directory.
                - If it is False, it stops monitoring if it was active.
            experimentation_folder: choose a specific name for the folder where experimentation result files and
                breakpoints are stored. This should just contain the name for the folder not a path. The name is used
                as a subdirectory of `environ[EXPERIMENTS_DIR])`. Defaults to None (auto-choose a folder name)
                - Caveat : if using a specific name this experimentation will not be automatically detected as the last
                experimentation by `load_breakpoint`
                - Caveat : do not use a `experimentation_folder` name finishing with numbers ([0-9]+) as this would
                confuse the last experimentation detection heuristic by `load_breakpoint`.
            secagg: whether to setup a secure aggregation context for this experiment, and use it
                to send encrypted updates from nodes to researcher. Defaults to `False`
        
        Raises:
            FedbiomedJobError: bad argument type or value
            FedbiomedJobError: cannot save training plan to file
        
        """
        # predefine all class variables, so no need to write try/except
        # block each time we use it
        self._fds = None
        self._reqs = None
        self._training_args = None
        self._tags = None
        self._experimentation_folder = None
        self._secagg = None
        self._node_state_agent: NodeStateAgent = None
        self._researcher_id = environ['RESEARCHER_ID']
        self._id = JOB_PREFIX + str(uuid.uuid4())  # creating a unique job id # TO BE RENAMED

        # set self._secagg
        self.set_secagg(secagg)

        # set self._tags and self._nodes
        self.set_tags(tags)
        self.set_nodes(nodes)

        # set self._model_args and self._training_args to dict
        self.set_training_args(training_args)

        # Useless to add a param and setter/getter for Requests() as it is a singleton ?
        self._reqs = Requests()

        # set self._fds: type Union[FederatedDataSet, None]
        self.set_training_data(training_data, True)

        # set self._experimentation_folder: type str
        self.set_experimentation_folder(experimentation_folder)

        # at this point self._fds is FederatedDataset or None (not a dict anymore)
        self._node_state_agent = NodeStateAgent(list(self._fds.data().keys())
                                                if self._fds and self._fds.data() else [])

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

        Please see [`set_tags`][fedbiomed.researcher.experiment.Experiment.set_tags] to set tags.

        Returns:
            List of tags that has been set. `None` if it isn't declare yet.
        """
        return self._tags

    @exp_exceptions
    def nodes(self) -> Union[List[str], None]:
        """Retrieves the `nodes` that are chosen for federated training.

        Please see [`set_nodes`][fedbiomed.researcher.experiment.Experiment.set_nodes] to set `nodes`.

        Returns:
            Object that contains meta-data for the datasets of each node. `None` if nodes are not set.
        """
        return self._nodes

    @exp_exceptions
    def training_data(self) -> Union[FederatedDataSet, None]:
        """Retrieves the training data which is an instance of [`FederatedDataset`]
        [fedbiomed.researcher.datasets.FederatedDataSet]

        Please see [`set_training_data`][fedbiomed.researcher.experiment.Experiment.set_training_data] to set or
        update training data.

        Returns:
            Object that contains meta-data for the datasets of each node. `None` if it isn't set yet.
        """
        return self._fds

    @exp_exceptions
    def experimentation_folder(self) -> str:
        """Retrieves the folder name where experiment data/result are saved.

        Please see also[`set_experimentation_folder`]
        [fedbiomed.researcher.experiment.Experiment.set_experimentation_folder]

        Returns:
            File name where experiment related files are saved
        """

        return self._experimentation_folder

    # derivative from experimentation_folder
    @exp_exceptions
    def experimentation_path(self) -> str:
        """Retrieves the file path where experimentation folder is located and experiment related files are saved.

        Returns:
            Experiment directory where all experiment related files are saved
        """

        return os.path.join(environ['EXPERIMENTS_DIR'], self._experimentation_folder)

    @exp_exceptions
    def training_args(self) -> dict:
        """Retrieves training arguments.

        Please see also [`set_training_args`][fedbiomed.researcher.experiment.Experiment.set_training_args]

        Returns:
            The arguments that are going to be passed to `training_routine` of [`training_plans`]
                [fedbiomed.common.training_plans] classes to perfom training on the node side.
                An example training routine: [`TorchTrainingPlan.training_routine`]
                [fedbiomed.common.training_plans.TorchTrainingPlan.training_routine]
        """

        return self._training_args.dict()

    @property
    def id(self):
        return self._id

    @exp_exceptions
    def info(self, info=None) -> Dict[str, Any]:
        """Prints out the information about the current status of the experiment.

        Lists  all the parameters/arguments of the experiment and informs whether the experiment can be run.

        Raises:
            FedbiomedExperimentError: Inconsistent experiment due to missing variables
        """
        if info is None:
            info = {
                'Arguments': [],
                'Values': []
            }
        info['Arguments'].extend([
                'Tags',
                'Nodes filter',
                'Training Data',
                'Training Arguments',
                'Experiment folder',
                'Experiment Path',
                'Secure Aggregation'
            ])
        info['Values'].extend(['\n'.join(findall('.{1,60}', str(e))) for e in [
                           self._tags,
                           self._nodes,
                           self._fds,
                           self._training_args,
                           self._experimentation_folder,
                           self.experimentation_path(),
                           f'- Using: {self._secagg}\n- Active: {self._secagg.active}'
                       ]])
        print(tabulate.tabulate(info, headers='keys'))
        return info

    # Setters
    @exp_exceptions
    def set_tags(self, tags: Union[List[str], str, None]) -> Union[List[str], None]:
        """Sets tags + verifications on argument type

        Args:
            tags: List of string with data tags or string with one data tag. Empty list
                of tags ([]) means any dataset is accepted, it is different from None (tags not set, cannot search
                for training_data yet).

        Returns:
            List of tags that are set. None, if the argument `tags` is None.

        Raises:
            FedbiomedExperimentError : Bad tags type
        """

        if isinstance(tags, list):
            for tag in tags:
                if not isinstance(tag, str):
                    msg = ErrorNumbers.FB410.value + f' `tags` : list of {type(tag)}'
                    logger.critical(msg)
                    raise FedbiomedExperimentError(msg)
            self._tags = tags
        elif isinstance(tags, str):
            self._tags = [tags]
        elif tags is None:
            self._tags = tags
        else:
            msg = ErrorNumbers.FB410.value + f' `tags` : {type(tags)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        # self._tags always exist at this point

        if self._fds is not None:
            logger.debug('Experimentation tags changed, you may need to update `training_data`')

        return self._tags

    @exp_exceptions
    def set_nodes(self, nodes: Union[List[str], None]) -> Union[List[str], None]:
        """Sets for nodes + verifications on argument type

        Args:
            nodes: List of node_ids to filter the nodes to be involved in the experiment.

        Returns:
            List of tags that are set. None, if the argument `nodes` is None.

        Raises:
            FedbiomedExperimentError : Bad nodes type
        """

        if isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, str):
                    msg = ErrorNumbers.FB410.value + f' `nodes` : list of {type(node)}'
                    logger.critical(msg)
                    raise FedbiomedExperimentError(msg)
            self._nodes = nodes
        elif nodes is None:
            self._nodes = nodes
        else:
            msg = ErrorNumbers.FB410.value + f' `nodes` : {type(nodes)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        # self._nodes always exist at this point

        if self._fds is not None:
            logger.debug('Experimentation nodes filter changed, you may need to update `training_data`')

        return self._nodes

    @exp_exceptions
    def set_training_data(
            self,
            training_data: Union[FederatedDataSet, dict, None],
            from_tags: bool = False) -> \
            Union[FederatedDataSet, None]:
        """Sets training data for federated training + verification on arguments type

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
            Nodes and dataset with meta-data

        Raises:
            FedbiomedExperimentError : bad training_data type
        """
        # we can trust _reqs _tags _nodes are existing and properly typed/formatted

        if not isinstance(from_tags, bool):
            msg = ErrorNumbers.FB410.value + f' `from_tags` : got {type(from_tags)} but expected a boolean'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # case where no training data are passed
        if training_data is None and from_tags is True:
            # cannot search for training_data if tags not initialized;
            # nodes can be None (no filtering on nodes by default)
            if self._tags is not None:
                training_data = self._reqs.search(self._tags, self._nodes)

        if isinstance(training_data, FederatedDataSet):
            self._fds = training_data
        elif isinstance(training_data, dict):
            self._fds = FederatedDataSet(training_data)
        elif training_data is not None:
            msg = ErrorNumbers.FB410.value + f' `training_data` has incorrect type: {type(training_data)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        else:
            self._fds = None
            logger.debug('Experiment not fully configured yet: no training data')
        # at this point, self._fds is either None or a FederatedDataSet object

        return self._fds

    @exp_exceptions
    def set_experimentation_folder(self, experimentation_folder: Union[str, None]) -> str:
        """Sets `experimentation_folder`, the folder name where experiment data/result are saved.

        Args:
            experimentation_folder: File name where experiment related files are saved

        Returns:
            experimentation_folder (str)

        Raises:
            FedbiomedExperimentError : bad `experimentation_folder` type
        """
        if experimentation_folder is None:
            self._experimentation_folder = create_exp_folder()
        elif isinstance(experimentation_folder, str):
            sanitized_folder = sanitize_filename(experimentation_folder, platform='auto')
            self._experimentation_folder = create_exp_folder(sanitized_folder)
            if sanitized_folder != experimentation_folder:
                logger.warning(f'`experimentation_folder` was sanitized from '
                               f'{experimentation_folder} to {sanitized_folder}')
        else:
            msg = ErrorNumbers.FB410.value + \
                  f' `experimentation_folder` : {type(experimentation_folder)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

            # at this point self._experimentation_folder is a str valid for a foldername

        # _job doesn't always exist at this point
        return self._experimentation_folder

    @exp_exceptions
    def set_training_args(self, training_args: dict, reset: bool = True) -> dict:
        """ Sets `training_args` + verification on arguments type

        Args:
            training_args (dict): contains training arguments passed to the `training_routine` of the
                [`fedbiomed.common.training_plans`][fedbiomed.common.training_plans] when launching it:
                lr, epochs, batch_size...
            reset (bool, optional): whether to reset the training_args (if previous training_args has already been
                set), or to update them with training_args. Defaults to True.

        Returns:
            Training arguments

        Raises:
            FedbiomedExperimentError : bad training_args type
        """

        if isinstance(training_args, TrainingArgs):
            self._training_args = deepcopy(training_args)
        else:
            self._training_args = TrainingArgs(training_args, only_required=False)
        # Propagate training arguments to job
        return self._training_args.dict()

    @exp_exceptions
    def set_secagg(self, secagg: Union[bool, SecureAggregation]):

        if isinstance(secagg, bool):
            self._secagg = SecureAggregation(active=secagg)
        elif isinstance(secagg, SecureAggregation):
            self._secagg = secagg
        else:
            msg = f"{ErrorNumbers.FB410.value}: Expected `secagg` argument bool or `SecureAggregation`, " \
                  f"but got {type(secagg)}"
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        return self._secagg

    def secagg_setup(self):
        secagg_arguments = {}
        if self._secagg.active:
            self._secagg.setup(parties=[environ["ID"]] + self._job.nodes,
                               job_id=self._id)
            secagg_arguments = self._secagg.train_arguments()
        return secagg_arguments

    def _raise_for_missing_job_prerequities(self) -> None:
        """Setter for job, it verifies pre-requisites are met for creating a job
        attached to this experiment. If yes, instantiate a job ; if no, return None.

        """
        if self._training_plan_is_defined is not True:
            # training plan not properly defined yet
            msg = f'Experiment not fully configured yet: no job. Missing proper training plan definition ' \
                  f'(training_plan={self._training_plan_class} training_plan_path={self._training_plan_path})'
            raise FedbiomedExperimentError(msg)
        elif self._fds is None:
            msg='Experiment not fully configured yet: no job. Missing training data'
            raise FedbiomedExperimentError(msg)

    def _update_nodes_states_agent(self, before_training: bool = True):
        """Updates [`NodeStateAgent`][fedbiomed.researcher.node_state_agent.NodeStateAgent], with the latest
        state_id coming from `Nodes` contained among all `Nodes` within
        [`FederatedDataset`][fedbiomed.researcher.datasets.FederatedDataset].

        Args:
            before_training: whether to update `NodeStateAgent` at the begining or at the end of a `Round`:
                - if before, only updates `NodeStateAgent` wrt `FederatedDataset`, otherwise
                - if after, updates `NodeStateAgent` wrt latest reply

        Raises:
            FedBiomedNodeStateAgenError: failing to update `NodeStateAgent`.

        """
        node_ids = list(self._fds.data().keys()) if self._fds and self._fds.data() else []
        self._node_state_agent.update_node_states(node_ids)

    
    def save_state_breakpoint(self) -> Dict:
        """"""
        state = {}
        state['node_state'] =  self._node_state_agent.save_state_breakpoint()
        return state

    def load_state_breakpoint(self, saved_state: Dict):
        """"""
        self._node_state_agent.load_state_breakpoint(saved_state.get('node_state'))

    @abstractmethod
    def run_once(self) -> int:
        """"""

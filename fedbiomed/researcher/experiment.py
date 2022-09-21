"""Code of the researcher. Implements the experiment orchestration"""


import functools
import os
import sys
import json
import inspect
import traceback
import torch

from copy import deepcopy
from re import findall
from tabulate import tabulate
from typing import Callable, Tuple, Union, Dict, Any, TypeVar, Type, List
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.researcher.aggregators.functional import initialize, calculate_param_updates_fedopt, \
    update_momentum_fedopt, update_second_moment_fedopt
from pathvalidate import sanitize_filename, sanitize_filepath

from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.utils import is_ipython
from fedbiomed.common.exceptions import FedbiomedExperimentError, FedbiomedError, \
    FedbiomedSilentTerminationError
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TorchTrainingPlan, SKLearnTrainingPlan
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.filetools import create_exp_folder, choose_bkpt_file, \
    create_unique_link, create_unique_file_link, find_breakpoint_path
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.monitor import Monitor
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.strategies.strategy import Strategy
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy


_E = TypeVar("Experiment")  # only for typing

# for checking class passed to experiment
# TODO : should we move this to common/constants.py ?
training_plans = (TorchTrainingPlan, SKLearnTrainingPlan)
# for typing only
TrainingPlan = TypeVar('TrainingPlan', TorchTrainingPlan, SKLearnTrainingPlan)
Type_TrainingPlan = TypeVar('Type_TrainingPlan', Type[TorchTrainingPlan], Type[SKLearnTrainingPlan])


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


# Experiment

class Experiment(object):
    """
    This class represents the orchestrator managing the federated training
    """

    @exp_exceptions
    def __init__(self,
                 tags: Union[List[str], str, None] = None,
                 nodes: Union[List[str], None] = None,
                 training_data: Union[FederatedDataSet, dict, None] = None,
                 aggregator: Union[Aggregator, Type[Aggregator], None] = None,
                 node_selection_strategy: Union[Strategy, Type[Strategy], None] = None,
                 round_limit: Union[int, None] = None,
                 model_class: Union[Type_TrainingPlan, str, None] = None,
                 model_path: Union[str, None] = None,
                 model_args: dict = {},
                 training_args: Union[TypeVar("TrainingArgs"), dict, None] = None,
                 save_breakpoints: bool = False,
                 tensorboard: bool = False,
                 experimentation_folder: Union[str, None] = None
                 ):

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
            node_selection_strategy:object or class defining how nodes are sampled at each round for training, and how
                non-responding nodes are managed.  Defaults to None:
                - use [`DefaultStrategy`][fedbiomed.researcher.strategies.DefaultStrategy] if training_data is initialized
                - else strategy is None (cannot be initialized), experiment cannot be launched yet
            round_limit: the maximum number of training rounds (nodes <-> central server) that should be executed for
                the experiment. `None` means that no limit is defined. Defaults to None.
            model_class: name of the model class [`str`][str] or model class (`Type_TrainingPlan`) to use for training.
                For experiment to be properly and fully defined `model_class` needs to be:
                - a [`str`][str] when `model_path` is not None (model class comes from a file).
                - a `Type_TrainingPlan` when `model_path` is None (model class passed as argument).
                Defaults to None (no model class defined yet)

            model_path: path to a file containing model code [`str`][str] or None (no file containing model code,
                `model_class` needs to be a class matching `Type_TrainingPlan`) Defaults to None.
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
        """

        # predefine all class variables, so no need to write try/except
        # block each time we use it
        self._fds = None
        self._node_selection_strategy = None
        self._job = None
        self._round_limit = None
        self._model_path = None
        self._reqs = None
        self._training_args = None
        self._node_selection_strategy = None
        self._tags = None
        self._monitor = None
        self._experimentation_folder = None

#        training_data: Union[FederatedDataSet, dict, None] = None,
#        aggregator: Union[Aggregator, Type[Aggregator], None] = None,
#        node_selection_strategy: Union[Strategy, Type[Strategy], None] = None,
#        model_class: Union[Type_TrainingPlan, str, None] = None,
#        model_args: dict = {},
#        training_args: dict = {},
#        save_breakpoints: bool = False,
#        tensorboard: bool = False,
#        experimentation_folder: Union[str, None] = None

        # set self._tags and self._nodes
        self.set_tags(tags)
        self.set_nodes(nodes)

        # set self._model_args and self._training_args to dict
        self.set_model_args(model_args)
        self.set_training_args(training_args)

        # Useless to add a param and setter/getter for Requests() as it is a singleton ?
        self._reqs = Requests()

        # set self._fds: type Union[FederatedDataSet, None]
        self.set_training_data(training_data, True)

        # set self._aggregator : type Aggregator
        self.set_aggregator(aggregator)

        # set self._node_selection_strategy: type Union[Strategy, None]
        self.set_strategy(node_selection_strategy)

        # "current" means number of rounds already trained
        self._set_round_current(0)
        self.set_round_limit(round_limit)

        # set self._experimentation_folder: type str
        self.set_experimentation_folder(experimentation_folder)
        # Note: currently keep this parameter as it cannot be updated in Job()
        # without refactoring Job() first

        # sets self._model_is_defined: bool == is the model properly defined ?
        # with current version of jobs, a correctly defined model requires:
        # - either model_path to None + model_class is the class a training plan
        # - or model_path not None + model_class is a name (str) of a training plan
        #
        # note: no need to set self._model_is_defined before calling `set_model_class`
        self.set_model_class(model_class)
        self.set_model_path(model_path)

        # set self._job to Union[Job, None]
        self.set_job()

        # TODO: rewrite after experiment results refactoring
        self._aggregated_params = {}

        self.set_save_breakpoints(save_breakpoints)

        # always create a monitoring process
        self._monitor = Monitor()
        self._reqs.add_monitor_callback(self._monitor.on_message_handler)
        self.set_tensorboard(tensorboard)

        self._strategy_info = {"strategy": aggregator.aggregator_name}
        if aggregator.aggregator_name in ["FedAdam", "FedAdagrad", "FedYogi"]:
            self._strategy_info['beta1'] = aggregator.beta1
            self._strategy_info['beta2'] = aggregator.beta2
            self._strategy_info['tau'] = aggregator.tau
            self._strategy_info['server_lr'] = aggregator.server_lr


    # destructor
    @exp_exceptions
    def __del__(self):
        # This part has been commented, self._reqs.remove_monitor_callback() removes monitor
        # callback when initializing an experiment for the second time with same name.
        # While recreating a class with same variable name python first calls __init__ and then __del__.

        # if self._reqs is not None:
        #     # TODO: confirm placement for finishing monitoring - should be at the end of the experiment
        #     self._reqs.remove_monitor_callback()

        if self._monitor is not None:
            self._monitor.close_writer()

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
    def aggregator(self) -> Aggregator:
        """ Retrieves aggregator class that will be used for aggregating model parameters.

        To set or update aggregator: [`set_aggregator`][fedbiomed.researcher.experiment.Experiment.set_aggregator].

        Returns:
            A class or an object that is an instance of [Aggregator][fedbiomed.researcher.aggregators.Aggregator]

        """
        return self._aggregator

    @exp_exceptions
    def strategy(self) -> Union[Strategy, None]:
        """Retrieves the class that represents the node selection strategy.

        Please see also [`set_strategy`][fedbiomed.researcher.experiment.Experiment.set_strategy] to set or update
        node selection strategy.

        Returns:
            A class or object as an instance of [`Strategy`][fedbiomed.researcher.strategies.Strategy]. `None` if
                it is not declared yet. It means that node selection strategy will be
                [`DefaultStrategy`][fedbiomed.researcher.strategies.DefaultStrategy].
        """
        return self._node_selection_strategy

    @exp_exceptions
    def round_limit(self) -> Union[int, None]:
        """Retrieves the round limit from the experiment object.

        Please see  also [`set_round_limit`][fedbiomed.researcher.experiment.Experiment.set_training_data] to change
        or set round limit.

        Returns:
            Round limit that shows maximum number of rounds that can be performed. `None` if it isn't declared yet.
        """
        return self._round_limit

    @exp_exceptions
    def round_current(self) -> int:
        """Retrieves the round where the experiment is at.

        Returns:
            Indicates the round number that the experiment will perform next.
        """
        return self._round_current

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
    def model_class(self) -> Union[Type_TrainingPlan, str, None]:
        """Retrieves the model (training plan class) that is created for training.

        Please see also [`set_model_class`][fedbiomed.researcher.experiment.Experiment.set_model_class].

        Returns:
            Training plan class as one of [`Type_TrainingPlan`][fedbiomed.researcher.experiment.Type_TrainingPlan]. None
                if it isn't declared yet. [`str`][str] if [`model_path`]
                [fedbiomed.researcher.experiment.Experiment.model_path]that represents model class created externally
                is provided.
        """

        return self._model_class

    @exp_exceptions
    def model_path(self) -> Union[str, None]:
        """Retrieves model path where model class is saved as python script externally.

        Please see also [`set_model_path`][fedbiomed.researcher.experiment.Experiment.set_model_path].

        Returns:
            Path to python script (`.py`) where model class (training plan) is created. None if it isn't declared yet.
        """

        return self._model_path

    @exp_exceptions
    def model_args(self) -> dict:
        """Retrieves model arguments.

        Please see also [`set_model_args`][fedbiomed.researcher.experiment.Experiment.set_model_args]

        Returns:
            The arguments that are going to be passed to [`training_plans`][fedbiomed.common.training_plans]
                classes in built time on the node side.
        """
        return self._model_args

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

        return self._training_args

    @exp_exceptions
    def test_ratio(self) -> float:
        """Retrieves the ratio for validation partition of entire dataset.

        Please see also [`set_test_ratio`][fedbiomed.researcher.experiment.Experiment.set_test_ratio] to change/set `test_ratio`

        Returns:
            The ratio for validation part, `1 - test_ratio` is ratio for training set.
        """

        return self._training_args['test_ratio']

    @exp_exceptions
    def test_metric(self) -> Union[MetricTypes, str, None]:
        """Retrieves the metric for validation routine.

        Please see also [`set_test_metric`][fedbiomed.researcher.experiment.Experiment.set_test_metric] to change/set `test_metric`

        Returns:
            A class as an instance of [`MetricTypes`][fedbiomed.common.metrics.MetricTypes]. [`str`][str] for referring
                one of  metric which provided as attributes in [`MetricTypes`][fedbiomed.common.metrics.MetricTypes].
                None, if it isn't declared yet.
        """

        return self._training_args['test_metric']

    @exp_exceptions
    def test_metric_args(self) -> Dict[str, Any]:
        """Retrieves the metric argument for the metric function that is going to be used.

        Please see also [`set_test_metric`][fedbiomed.researcher.experiment.Experiment.set_test_metric] to change/set
        `test_metric` and get more information on the arguments can be used.

        Returns:
            A dictionary that contains arguments for metric function. See [`set_test_metric`]
                [fedbiomed.researcher.experiment.Experiment.set_test_metric]
        """
        return self._training_args['test_metric_args']

    @exp_exceptions
    def test_on_local_updates(self) -> bool:
        """Retrieves the status of whether validation will be performed on locally updated parameters by
        the nodes at the end of each round.

        Please see also [`set_test_on_local_updates`][fedbiomed.researcher.experiment.Experiment.set_test_on_local_updates].

        Returns:
            True, if validation is active on locally updated parameters. False for vice versa.
        """

        return self._training_args['test_on_local_updates']

    @exp_exceptions
    def test_on_global_updates(self) -> bool:
        """ Retrieves the status of whether validation will be performed on globally updated (aggregated)
        parameters by the nodes at the beginning of each round.

        Please see also [`set_test_on_global_updates`]
        [fedbiomed.researcher.experiment.Experiment.set_test_on_global_updates].

        Returns:
            True, if validation is active on globally updated (aggregated) parameters. False for vice versa.
        """
        return self._training_args['test_on_global_updates']

    @exp_exceptions
    def job(self) -> Union[Job, None]:
        """Retrieves the [`Job`][fedbiomed.researcher.job] that manages training rounds.

        Returns:
            Initialized `Job` object. None, if it isn't declared yet or not information to set to job. Please see
                [`set_job`][fedbiomed.researcher.experiment.Experiment.set_job].
        """

        return self._job

    @exp_exceptions
    def save_breakpoints(self) -> bool:
        """Retrieves the status of saving breakpoint after each round of training.

        Returns:
            `True`, If saving breakpoint is active. `False`, vice versa.
        """

        return self._save_breakpoints

    @exp_exceptions
    def monitor(self) -> Monitor:
        """Retrieves the monitor object

        Monitor is responsible for receiving and parsing real-time training and validation feed-back from each node
        participate to federated training. See [`Monitor`][fedbiomed.researcher.monitor.Monitor]

        Returns:
            Monitor object that will always exist with experiment to retrieve feed-back from the nodes.
        """
        return self._monitor

    # TODO: update these getters after experiment results refactor / job refactor

    @exp_exceptions
    def aggregated_params(self) -> dict:
        """Retrieves all aggregated parameters of each round of training

        Returns:
            Dictionary of aggregated parameters keys stand for each round of training
        """

        return self._aggregated_params

    @exp_exceptions
    def training_replies(self) -> Union[dict, None]:
        """Retrieves training replies of each round of training.

        Training replies contains timing statistics and the files parth/URLs that has been received after each round.

        Returns:
            Dictionary of training replies keys stand for each round of training. None, if
                [Job][fedbiomed.researcher.job] isn't declared or empty dict if there is no training round has been run.
        """

        # at this point `job` is defined but may be None
        if self._job is None:
            logger.error('No `job` defined for experiment, cannot get `training_replies`')
            return None
        else:
            return self._job.training_replies

    # TODO: better checking of model object type in Job() to guarantee it is a TrainingPlan
    @exp_exceptions
    def model_instance(self) -> Union[TrainingPlan, None]:
        """ Retrieves model instance that has been built and send the nodes through HTTP restfull service
        for each round of training.


        !!! info "Loading aggregated parameters"

            After retrieving the model instance aggregated parameters should be loaded.

            Example:

            ```python
            model = epx.model_instance()
            model.load_state_dict(exp.aggregated_params()[rounds - 1]['params'])
            ```
        Returns:
            Model object which is an instance one of [training_plans][fedbiomed.common.training_plans].
        """
        # at this point `job` is defined but may be None
        if self._job is None:
            logger.error('No `job` defined for experiment, cannot get `model_instance`')
            return None
        else:
            return self._job.model

    # a specific getter-like
    @exp_exceptions
    def info(self) -> None:
        """Prints out the information about the current status of the experiment.

        Lists  all the parameters/arguments of the experiment and informs whether the experiment can be run.

        Raises:
            FedbiomedExperimentError: Inconsistent experiment due to missing variables
        """

        # at this point all attributes are initialized (in constructor)
        info = {
            'Arguments': [
                'Tags',
                'Nodes filter',
                'Training Data',
                'Aggregator',
                'Strategy',
                'Job',
                'Model Path',
                'Model Class',
                'Model Arguments',
                'Training Arguments',
                'Rounds already run',
                'Rounds total',
                'Experiment folder',
                'Experiment Path',
                'Breakpoint State'
            ],
            # max 60 characters per column for values - can we do that with tabulate() ?
            'Values': ['\n'.join(findall('.{1,60}',
                                         str(e))) for e in [
                                             self._tags,
                                             self._nodes,
                                             self._fds,
                                             self._aggregator.aggregator_name if self._aggregator is not None else None,
                                             self._node_selection_strategy,
                                             self._job,
                                             self._model_path,
                                             self._model_class,
                                             self._model_args,
                                             self._training_args,
                                             self._round_current,
                                             self._round_limit,
                                             self._experimentation_folder,
                                             self.experimentation_path(),
                                             self._save_breakpoints
                                         ]
                       ]
        }
        print(tabulate(info, headers='keys'))

        # definitions that may be missing for running the experiment
        # (value None == not defined yet for _fds et _job,
        # False == no valid model for _model_is_defined )
        may_be_missing = {
            '_fds': 'Training Data',
            '_node_selection_strategy': 'Strategy',
            '_model_is_defined': 'Model',
            '_job': 'Job'
        }
        # definitions found missing
        missing = ''

        for key, value in may_be_missing.items():
            try:
                if eval('self.' + key) is None or eval('self.' + key) is False:
                    missing += f'- {value}\n'
            except Exception:
                # should not happen, all eval variables should be defined
                msg = ErrorNumbers.FB400.value + \
                    f', in method `info` : self.{key} not defined for experiment'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        if missing:
            print(f'\nExperiment cannot be run (not fully defined), missing :\n{missing}')
        else:
            print('\nExperiment can be run now (fully defined)')

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
            self._nodes = nodes
            for node in nodes:
                if not isinstance(node, str):
                    msg = ErrorNumbers.FB410.value + f' `nodes` : list of {type(node)}'
                    logger.critical(msg)
                    raise FedbiomedExperimentError(msg)
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
            from_tags: Specificities; If True, query nodes for datasets when no `training_data` is provided.
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
            # TODO: FederatedDataSet constructor should verify typing and format
            self._fds = FederatedDataSet(training_data)
        elif training_data is not None:
            msg = ErrorNumbers.FB410.value + f' `training_data` has incorrect type: {type(training_data)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        else:
            self._fds = None
            logger.debug('Experiment not fully configured yet: no training data')
        # at this point, self._fds is either None or a FederatedDataSet object

        if self._node_selection_strategy is not None:
            logger.debug('Training data changed, '
                         'you may need to update `node_selection_strategy`')
        if self._job is not None:
            logger.debug('Training data changed, you may need to update `job`')

        return self._fds

    @exp_exceptions
    def set_aggregator(self, aggregator: Union[Aggregator, Type[Aggregator], None]) -> \
            Aggregator:
        """Sets aggregator + verification on arguments type

        Args:
            aggregator: Object or class defining the method for aggregating local updates. Default to None
                (use `FedAverage` for aggregation)

        Returns:
            aggregator (Aggregator)

        Raises:
            FedbiomedExperimentError : bad aggregator type
        """

        if aggregator is None:
            # default aggregator
            self._aggregator = FedAverage()
        elif inspect.isclass(aggregator):
            # a class is provided, need to instantiate an object
            if issubclass(aggregator, Aggregator):
                self._aggregator = aggregator()
            else:
                # bad argument
                msg = ErrorNumbers.FB410.value + f' `aggregator` : {aggregator} class'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        elif isinstance(aggregator, Aggregator):
            # an object of a proper class is provided, nothing to do
            self._aggregator = aggregator
        else:
            # other bad type or object
            msg = ErrorNumbers.FB410.value + f' `aggregator` : {type(aggregator)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # at this point self._aggregator is (non-None) aggregator object
        return self._aggregator

    @exp_exceptions
    def set_strategy(self, node_selection_strategy: Union[Strategy, Type[Strategy], None]) -> \
            Union[Strategy, None]:
        """Sets for `node_selection_strategy` + verification on arguments type

        Args:
            node_selection_strategy: object or class defining how nodes are sampled at each round for training, and
                how non-responding nodes are managed. Defaults to None:
                - use `DefaultStrategy` if training_data is initialized
                - else strategy is None (cannot be initialized), experiment cannot
                  be launched yet

        Returns:
            node selection strategy class

        Raises:
            FedbiomedExperimentError : bad strategy type
        """
        if self._fds is not None:
            if node_selection_strategy is None:
                # default node_selection_strategy
                self._node_selection_strategy = DefaultStrategy(self._fds)
            elif inspect.isclass(node_selection_strategy):
                # a class is provided, need to instantiate an object
                if issubclass(node_selection_strategy, Strategy):
                    self._node_selection_strategy = node_selection_strategy(self._fds)
                else:
                    # bad argument
                    msg = ErrorNumbers.FB410.value + \
                        f' `node_selection_strategy` : {node_selection_strategy} class'
                    logger.critical(msg)
                    raise FedbiomedExperimentError(msg)
            elif isinstance(node_selection_strategy, Strategy):
                # an object of a proper class is provided, nothing to do
                self._node_selection_strategy = node_selection_strategy
            else:
                # other bad type or object
                msg = ErrorNumbers.FB410.value + \
                    f' `node_selection_strategy` : {type(node_selection_strategy)}'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        else:
            # cannot initialize strategy if not FederatedDataSet yet
            self._node_selection_strategy = None
            logger.debug('Experiment not fully configured yet: no node selection strategy')

        # at this point self._node_selection_strategy is a Union[Strategy, None]
        return self._node_selection_strategy

    @exp_exceptions
    def set_round_limit(self, round_limit: Union[int, None]) -> Union[int, None]:
        """Sets `round_limit` + verification on arguments type

        Args:
            round_limit: the maximum number of training rounds (nodes <-> central server) that should be executed
                for the experiment. `None` means that no limit is defined.

        Returns:
            Round limit for experiment of federated learning

        Raises:
            FedbiomedExperimentError : bad rounds type or value
        """
        # at this point round_current exists and is an int >= 0

        if round_limit is None:
            # no limit for training rounds
            self._round_limit = None
        elif isinstance(round_limit, int):
            # at this point round_limit is an int
            if round_limit < 0:
                msg = ErrorNumbers.FB410.value + f' `round_limit` can not be negative: {round_limit}'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
            elif round_limit < self._round_current:
                # self._round_limit can't be less than current round
                logger.error(f'cannot set `round_limit` to less than the number of already run rounds '
                             f'({self._round_current})')
            else:
                self._round_limit = round_limit
        else:
            msg = ErrorNumbers.FB410.value + f' `round_limit` : {type(round_limit)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # at this point self._round_limit is a Union[int, None]
        return self._round_limit

    # no setter for self._round_current eg
    # def set_round_current(self, round_current: int) -> int:
    # ...
    #
    # - does not make sense to increase `self._round_current` == padding with "non existing" rounds,
    #   would need to invent some dummy data for strategy, experiment results, etc.
    # - erasing rounds is complicated: not only decreasing `self._round_current)`, need
    #   to clean some experiment results (aggregated_params, job.training_replies, ...),
    #   change state of aggregator, strategy, etc... == the proper way of doing it is to
    #   load a breakpoint

    # private 'setter' needed when loading experiment - should not be made public
    @exp_exceptions
    def _set_round_current(self, round_current: int) -> int:
        """Private setter for `round_current` + verification on arguments type

        Args:
            round_current: the number of already completed training rounds in the experiment.

        Returns:
            Current round that experiment will run as next round

        Raises:
            FedbiomedExperimentError : bad round_current type or value
        """
        if not isinstance(round_current, int):
            msg = ErrorNumbers.FB410.value + f' `round_current` : {type(round_current)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        if round_current < 0:
            # cannot set a round <0
            msg = ErrorNumbers.FB410.value + f' `round_current` : {round_current}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        #
        if self._round_limit is not None and round_current > self._round_limit:

            # cannot set a round over the round_limit (when it is not None)
            msg = ErrorNumbers.FB410.value + f' `round_current` : {round_current}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # everything is OK
        self._round_current = round_current

        # at this point self._round_current is an int
        return self._round_current

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
            if (sanitized_folder != experimentation_folder):
                logger.warning(f'`experimentation_folder` was sanitized from '
                               f'{experimentation_folder} to {sanitized_folder}')
        else:
            msg = ErrorNumbers.FB410.value + \
                f' `experimentation_folder` : {type(experimentation_folder)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

            # at this point self._experimentation_folder is a str valid for a foldername

        # _job doesn't always exist at this point
        if self._job is not None:
            logger.debug('Experimentation folder changed, you may need to update `job`')

        return self._experimentation_folder

    @exp_exceptions
    def set_model_class(self, model_class: Union[Type_TrainingPlan, str, None]) -> \
            Union[Type_TrainingPlan, str, None]:
        """Sets  `model_class` + verification on arguments type

        Args:
            model_class: name of the model class (`str`) or model class as one of [`TrainingPlans`]
                [fedbiomed.common.training_plans] to use for training. For experiment to be properly and fully defined
                `model_class` needs to be:
                    - a `str` when `model_path` is not None (model class comes from a file).
                    - a `Type_TrainingPlan` when `model_path` is None (model class passed
                    as argument).

        Returns:
            `model_class` that is set for experiment

        Raises:
            FedbiomedExperimentError : bad model_class type
        """
        if model_class is None:
            self._model_class = None
            self._model_is_defined = False
        elif isinstance(model_class, str):
            if str.isidentifier(model_class):
                # correct python identifier
                self._model_class = model_class
                # model_path may not be defined at this point

                self._model_is_defined = isinstance(self._model_path, str)

            else:
                # bad identifier
                msg = ErrorNumbers.FB410.value + f' `model_class` : {model_class} bad identifier'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        elif inspect.isclass(model_class):
            # model_class must be a subclass of a valid training plan
            if issubclass(model_class, training_plans):
                # valid class
                self._model_class = model_class
                # model_path may not be defined at this point

                self._model_is_defined = self._model_path is None
            else:
                # bad class
                msg = ErrorNumbers.FB410.value + f' `model_class` : {model_class} class'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        else:
            # bad type
            msg = ErrorNumbers.FB410.value + f' `model_class` of type: {type(model_class)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

            # self._model_is_defined and self._model_class always exist at this point
        if not self._model_is_defined:
            logger.debug(f'Experiment not fully configured yet: no valid model, '
                         f'model_class={self._model_class} model_path={self._model_path}')

        if self._job is not None:
            logger.debug('Experimentation model_class changed, you may need to update `job`')

        return self._model_class

    @exp_exceptions
    def set_model_path(self, model_path: Union[str, None]) -> Union[str, None]:
        """Sets `model_path` + verification on arguments type.

        Model path is the path where model class is saved as python script/module externally.

        Args:
            model_path (Union[str, None]) : path to a file containing  model code (`str`) or None (no file containing
                model code, `model_class` needs to be a class matching one of [`training_plans`]
                [fedbiomed.common.training_plans]

        Returns:
            The path that is set for retrieving module where model class is defined

        Raises:
            FedbiomedExperimentError : bad model_path type
        """
        # self._model_class and self._model_is_defined already exist when entering this function

        if model_path is None:
            self._model_path = None
            # .. so model is defined if it is a class (+ then, it has been tested as valid)
            self._model_is_defined = inspect.isclass(self._model_class)
        elif isinstance(model_path, str):
            if sanitize_filepath(model_path, platform='auto') == model_path \
                    and os.path.isfile(model_path):
                # provided model path is a sane path to an existing file
                self._model_path = model_path
                # if providing a model path, we expect a model class name (not a class)
                self._model_is_defined = isinstance(self._model_class, str)
            else:
                # bad filepath
                msg = ErrorNumbers.FB410.value + \
                    f' `model_path` : {model_path} is not a same path to an existing file'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        else:
            # bad type
            msg = ErrorNumbers.FB410.value + ' `model_path` : type(model_path)'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # self._model_path is also defined at this point
        if not self._model_is_defined:
            logger.debug(f'Experiment not fully configured yet: no valid model, '
                         f'model_class={self._model_class} model_path={self._model_path}')

        if self._job is not None:
            logger.debug('Experimentation model_path changed, you may need to update `job`')

        return self._model_path

    # TODO: model_args need checking of dict items, to be done by Job and node
    # (using a training plan method ?)
    @exp_exceptions
    def set_model_args(self, model_args: dict) -> dict:
        """Sets `model_args` + verification on arguments type

        Args:
            model_args (dict): contains model arguments passed to the constructor
                of the training plan when instantiating it : output and input feature
                dimension, etc.

        Returns:
            Model arguments that have been set.

        Raises:
            FedbiomedExperimentError : bad model_args type
        """
        if isinstance(model_args, dict):
            self._model_args = model_args
        else:
            # bad type
            msg = ErrorNumbers.FB410.value + f' `model_args` : {type(model_args)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        # self._model_args always exist at this point

        if self._job is not None:
            logger.debug('Experimentation model_args changed, you may need to update `job`')
        return self._model_args

    # TODO: training_args need checking of dict items, to be done by Job and node
    # (using a training plan method ? changing `training_routine` prototype ?)
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
            self._training_args = TrainingArgs(training_args, only_required = False)

        return self._training_args


    @exp_exceptions
    def set_test_ratio(self, ratio: float) -> float:
        """ Sets validation ratio for model validation.

        When setting test_ratio, nodes will allocate (1 - `test_ratio`) fraction of data for training and the
        remaining for validating model. This could be useful for validating the model, once every round, as well as
        controlling overfitting, doing early stopping, ....

        Args:
            ratio: validation ratio. Must be within interval [0,1].

        Returns:
            Validation ratio that is set

        Raises:
            FedbiomedExperimentError: bad data type
            FedbiomedExperimentError: ratio is not within interval [0, 1]
        """
        self._training_args['test_ratio'] = ratio

        if self._job is not None:
            # job setter function exists, use it
            self._job.training_args = self._training_args
            logger.debug('Experimentation training_args updated for `job`')

        return ratio

    @exp_exceptions
    def set_test_metric(self, metric: Union[MetricTypes, str, None], **metric_args: dict) -> \
            Tuple[Union[str, None], Dict[str, Any]]:
        """ Sets a metric for federated model validation

        Args:
            metric: A class as an instance of [`MetricTypes`][fedbiomed.common.metrics.MetricTypes]. [`str`][str] for
                referring one of  metric which provided as attributes in [`MetricTypes`]
                [fedbiomed.common.metrics.MetricTypes]. None, if it isn't declared yet.
            **metric_args: A dictionary that contains arguments for metric function. Arguments
                should be compatible with corresponding metrics in [`sklearn.metrics`][sklearn.metrics].

        Returns:
            Metric and  metric args as tuple

        Raises:
            FedbiomedExperimentError: Invalid type for `metric` argument
        """
        self._training_args['test_metric'] = metric

        # using **metric_args, we know `test_metric_args` is a Dict[str, Any]
        self._training_args['test_metric_args'] = metric_args

        if self._job is not None:
            # job setter function exists, use it
            self._job.training_args = self._training_args
            logger.debug('Experimentation training_args updated for `job`')

        return metric, metric_args

    @exp_exceptions
    def set_test_on_local_updates(self, flag: bool = True) -> bool:
        """
        Setter for `test_on_local_updates`, that indicates whether to perform a validation on the federated model on the
        node side where model parameters are updated locally after training in each node.

        Args:
            flag (bool, optional): whether to perform model validation on local updates. Defaults to True.

        Returns:
            value of the flag `test_on_local_updates`

        Raises:
            FedbiomedExperimentError: bad flag type
        """
        self._training_args['test_on_local_updates'] = flag

        if self._job is not None:
            # job setter function exists, use it
            self._job.training_args = self._training_args
            logger.debug('Experimentation training_args updated for `job`')

        return self._training_args['test_on_local_updates']

    @exp_exceptions
    def set_test_on_global_updates(self, flag: bool = True) -> bool:
        """
        Setter for test_on_global_updates, that indicates whether to  perform a validation on the federated model
        updates on the node side before training model locally where aggregated model parameters are received.

        Args:
            flag (bool, optional): whether to perform model validation on global updates. Defaults to True.

        Returns:
            Value of the flag `test_on_global_updates`.

        Raises:
            FedbiomedExperimentError : bad flag type
        """
        self._training_args['test_on_global_updates'] = flag

        if self._job is not None:
            # job setter function exists, use it
            self._job.training_args = self._training_args
            logger.debug('Experimentation training_args updated for `job`')

        return self._training_args['test_on_global_updates']

    # we could also handle `set_job(self, Union[Job, None])` but is it useful as
    # job is initialized with arguments that can be set ?
    @exp_exceptions
    def set_job(self) -> Union[Job, None]:
        """Setter for job, it verifies pre-requisites are met for creating a job
        attached to this experiment. If yes, instantiate a job ; if no, return None.

        Returns:
            The object that is initialized for creating round jobs.
        """
        # at this point all are defined among:
        # self.{_reqs,_fds,_model_is_defined,_model_class,_model_path,_model_args,_training_args}
        # self._experimentation_folder => self.experimentation_path()
        # self._round_current

        if self._job is not None:
            # a job is already defined, and it may also have run some rounds
            logger.debug('Experimentation `job` changed after running '
                         '{self._round_current} rounds, may give inconsistent results')

        if self._model_is_defined is not True:
            # model not properly defined yet
            self._job = None
            logger.debug('Experiment not fully configured yet: no job. Missing proper model '
                         f'definition (model_class={self._model_class} model_path={self._model_path})')
        elif self._fds is None:
            # not training data yet
            self._job = None
            logger.debug('Experiment not fully configured yet: no job. Missing training data')
        else:
            # meeting requisites for instantiating a job
            self._job = Job(reqs=self._reqs,
                            model=self._model_class,
                            model_path=self._model_path,
                            model_args=self._model_args,
                            training_args=self._training_args,
                            data=self._fds,
                            keep_files_dir=self.experimentation_path())

        return self._job

    # no setter implemented for experiment results, TODO after experiment results refactor
    # as decided during the refactor
    #
    # def set_aggregated_params(...)

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

    # Run experiment functions

    @exp_exceptions
    def set_tensorboard(self, tensorboard: bool) -> bool:
        """
        Sets the tensorboard flag

        Args:
            tensorboard: If `True` tensorboard log files will be writen after receiving training feedbacks

        Returns:
            Status of tensorboard
        """

        if isinstance(tensorboard, bool):
            self._tensorboard = tensorboard
            self._monitor.set_tensorboard(tensorboard)
        else:
            msg = ErrorNumbers.FB410.value + f' `tensorboard` : {type(tensorboard)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        return self._tensorboard

    def _init_empty_params(self, params: dict):
        """ Initialize empty tensors for given model parameters

        Args:
            params: dictionary containing prediction model parameters and corresponding weights

        Returns:
            a dictionary with same size as the input and empty tensors (full of zeros)
        """
        return {param:initialize(tensor)[1] for param, tensor in params.items()}

    def _calc_delta_aggregated_params(self, aggregated_params: dict, server_state: dict) -> dict:
        """ Calcultate aggregated weights delta, defined as the difference between the global state
        dict at the end of the round (after aggregation), and the global state dict at the beginning
        of the round (initial server state). In other words, it represents the weights updates made
        at the current round.

        Args:
            aggregated_params: dictionary containing prediction model parameters and corresponding aggregated weights
            server_state: dictionary containing weights of the global model at the beginning of the round

        Returns:
            a dictionary containing the difference between the aggregated params and the server state
        """
        delta_aggregated_params = {}
        for param, tensor in aggregated_params.items():
            delta_aggregated_params[param] = tensor - server_state[param]
        return delta_aggregated_params

    def _update_params(self, server_state: dict, updates: dict) -> dict:
        """ Defines new weights of a pytorch model, adding the computed updates to the server state.
        This is notably useful in the case of fedopt, where a specific treatment is applied to the aggregated
        weights delta, and the new update has to be added to the initial server state.

        Args:
            server_state: dictionary containing weights of the global model at the beginning of the round
            updates: dictionary containing weight updates for each parameter of a model

        Returns:
            a dictionary with updated model parameters, it will be the initial model of the next round
        """
        for param, tensor in updates.items():
            server_state[param] += tensor
        return server_state

    @exp_exceptions
    def run_once(self, increase: bool = False, test_after: bool = False) -> int:
        """Run at most one round of an experiment, continuing from the point the
        experiment had reached.

        If `round_limit` is `None` for the experiment (no round limit defined), run one round.
        If `round_limit` is not `None` and the `round_limit` of the experiment is already reached:
        * if `increase` is False, do nothing and issue a warning
        * if `increase` is True, increment total number of round `round_limit` and run one round

        Args:
            increase: automatically increase the `round_limit` of the experiment if needed. Does nothing if
                `round_limit` is `None`. Defaults to False
            test_after: if True, do a second request to the nodes after the round, only for validation on aggregated
                params. Intended to be used after the last training round of an experiment. Defaults to False.

        Returns:
            Number of rounds really run

        Raises:
            FedbiomedExperimentError: bad argument type or value
        """
        # check increase is a boolean
        if not isinstance(increase, bool):
            msg = ErrorNumbers.FB410.value + \
                f', in method `run_once` param `increase` : type {type(increase)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # nota:  we should never have self._round_current > self._round_limit, only ==
        if self._round_limit is not None and self._round_current >= self._round_limit:
            if increase is True:
                logger.debug(f'Auto increasing total rounds for experiment from {self._round_limit} '
                             f'to {self._round_current + 1}')
                self._round_limit = self._round_current + 1
            else:
                logger.warning(f'Round limit of {self._round_limit} was reached, do nothing')
                return 0

        # at this point, self._aggregator always exists and is not None
        # self.{_node_selection_strategy,_job} exist but may be None

        # check pre-requisites are met for running a round
        # for component in (self._node_selection_strategy, self._job):
        if self._node_selection_strategy is None:
            msg = ErrorNumbers.FB411.value + ', missing `node_selection_strategy`'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        elif self._job is None:
            msg = ErrorNumbers.FB411.value + ', missing `job`'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # Ready to execute a training round using the job, strategy and aggregator
        if self._strategy_info["strategy"] in ["FedAdam","FedAdagrad","FedYogi"]:
            self._server_state = self._job.get_server_model_params() # initial server state, before optimization/aggregation

        # Sample nodes using strategy (if given)
        self._job.nodes = self._node_selection_strategy.sample_nodes(self._round_current)
        logger.info('Sampled nodes in round ' + str(self._round_current) + ' ' + str(self._job.nodes))
        # Trigger training round on sampled nodes
        _ = self._job.start_nodes_training_round(round=self._round_current, do_training=True)

        # refining/normalizing model weights received from nodes
        model_params, weights = self._node_selection_strategy.refine(
            self._job.training_replies[self._round_current], self._round_current)

        # aggregate model from nodes to a global model
        aggregated_params = self._aggregator.aggregate(model_params,
                                                       weights)
        
        # In case of FedOpt, we apply a specific treatment on the aggregated & averaged parameters
        # depending the chosen strategy. Averaged parameters will be processed the same way as Adam,
        # Adagrad or Yogi algorithms do in a non-federated setting
        if self._strategy_info["strategy"] in ["FedAdam","FedAdagrad","FedYogi"]:
            _delta_aggregated_params = self._calc_delta_aggregated_params(aggregated_params, self._server_state)
            self._updates = self._init_empty_params(aggregated_params)
            if self._round_current == 0: # need to initialize momentum, second moment and adaptivity HP
                self._m = self._init_empty_params(aggregated_params) # momentum
                self._v = self._init_empty_params(aggregated_params) # second moment
                self._tauarray = {param:torch.ones_like(tensor) * self._tau for param, tensor in aggregated_params.items()}
            
            self._m = update_momentum_fedopt(strategy_info=self._strategy_info, momentum=self._m, delta_aggregated_params=_delta_aggregated_params)
            self._v = update_second_moment_fedopt(strategy_info=self._strategy_info, second_moment=self._v, delta_aggregated_params=_delta_aggregated_params)
            self._updates = calculate_param_updates_fedopt(strategy_info=self._strategy_info, updates=self._updates, momentum=self._m, second_moment=self._v, tau_array=self._tauarray)

            aggregated_params = self._update_params(self._server_state, self._updates)

        # write results of the aggregated model in a temp file
        aggregated_params_path = self._job.update_parameters(aggregated_params)
        logger.info(f'Saved aggregated params for round {self._round_current} '
                    f'in {aggregated_params_path}')

        self._aggregated_params[self._round_current] = {'params': aggregated_params,
                                                        'params_path': aggregated_params_path}

        self._round_current += 1

        # Update round in monitor for the next round
        self._monitor.set_round(round_=self._round_current + 1)

        if self._save_breakpoints:
            self.breakpoint()

        # do final validation after saving breakpoint :
        # not saved in breakpoint for current round, but more simple
        if test_after:
            # FIXME: should we sample nodes here too?
            self._job.start_nodes_training_round(round=self._round_current, do_training=False)

        return 1

    @exp_exceptions
    def run(self, rounds: Union[int, None] = None, increase: bool = False) -> int:
        """Run one or more rounds of an experiment, continuing from the point the
        experiment had reached.

        Args:
            rounds: Number of experiment rounds to run in this call.
                * `None` means "run all the rounds remaining in the experiment" computed as
                    maximum rounds (`round_limit` for this experiment) minus the number of
                    rounds already run rounds (`round_current` for this experiment).
                    It does nothing and issues a warning if `round_limit` is `None` (no
                    round limit defined for the experiment)
                * `int` >= 1 means "run at most `rounds` rounds".
                    If `round_limit` is `None` for the experiment, run exactly `rounds` rounds.
                    If a `round_limit` is set for the experiment and the number or rounds would
                increase beyond the `round_limit` of the experiment:
                - if `increase` is True, increase the `round_limit` to
                  (`round_current` + `rounds`) and run `rounds` rounds
                - if `increase` is False, run (`round_limit` - `round_current`)
                  rounds, don't modify the maximum `round_limit` of the experiment
                  and issue a warning.
            increase: automatically increase the `round_limit`
                of the experiment for executing the specified number of `rounds`.
                Does nothing if `round_limit` is `None` or `rounds` is None.
                Defaults to False

        Returns:
            Number of rounds have been run

        Raises:
            FedbiomedExperimentError: bad argument type or value
        """
        # check rounds is a >=1 integer or None
        if rounds is None:
            pass
        elif isinstance(rounds, int):
            if rounds < 1:
                msg = ErrorNumbers.FB410.value + \
                    f', in method `run` param `rounds` : value {rounds}'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        else:
            # bad type
            msg = ErrorNumbers.FB410.value + \
                f', in method `run` param `rounds` : type {type(rounds)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
            # check increase is a boolean
        if not isinstance(increase, bool):
            msg = ErrorNumbers.FB410.value + \
                f', in method `run` param `increase` : type {type(increase)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # compute number of rounds to run + updated rounds limit
        if rounds is None:
            if isinstance(self._round_limit, int):
                # run all remaining rounds in the experiment
                rounds = self._round_limit - self._round_current
                if rounds == 0:
                    # limit already reached
                    logger.warning(f'Round limit of {self._round_limit} already reached '
                                   'for this experiment, do nothing.')
                    return 0
            else:
                # cannot run if no number of rounds given and no round limit exists
                logger.warning('Cannot run, please specify a number of `rounds` to run or '
                               'set a `round_limit` to the experiment')
                return 0

        else:
            # at this point, rounds is an int >= 1
            if isinstance(self._round_limit, int):
                if (self._round_current + rounds) > self._round_limit:
                    if increase:
                        # dont change rounds, but extend self._round_limit as necessary
                        logger.debug(f'Auto increasing total rounds for experiment from {self._round_limit} '
                                     f'to {self._round_current + rounds}')
                        self._round_limit = self._round_current + rounds
                    else:
                        new_rounds = self._round_limit - self._round_current
                        if new_rounds == 0:
                            # limit already reached
                            logger.warning(f'Round limit of {self._round_limit} already reached '
                                           'for this experiment, do nothing.')
                            return 0
                        else:
                            # reduce the number of rounds to run in the experiment
                            logger.warning(f'Limit of {self._round_limit} rounds for the experiment '
                                           f'will be reached, reducing the number of rounds for this '
                                           f'run from {rounds} to {new_rounds}')
                            rounds = new_rounds

        # At this point `rounds` is an int > 0 (not None)

        # run the rounds
        for _ in range(rounds):
            if isinstance(self._round_limit, int) and self._round_current == (self._round_limit - 1) \
                    and self._training_args['test_on_global_updates'] is True:
                # Do "validation after a round" only if this a round limit is defined and we reached it
                # and validation is active on global params
                # When this condition is met, it also means we are running the last of
                # the `rounds` rounds in this function
                test_after = True
            else:
                test_after = False

            increment = self.run_once(increase=False, test_after=test_after)

            if increment == 0:
                # should not happen
                msg = ErrorNumbers.FB400.value + \
                    f', in method `run` method `run_once` returns {increment}'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)

        return rounds

    # Model checking functions

    @exp_exceptions
    def model_file(self, display: bool = True) -> str:
        """ This method displays saved final model for the experiment
            that will be sent to the nodes for training.

        Args:
            display: If `True`, prints content of the model file. Default is `True`

        Returns:
            Path to model file

        Raises:
            FedbiomedExperimentError: bad argument type, or cannot read model file content
        """
        if not isinstance(display, bool):
            # bad type
            msg = ErrorNumbers.FB410.value + \
                f', in method `model_file` param `display` : type {type(display)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # at this point, self._job exists (initialized in constructor)
        if self._job is None:
            # cannot check model file if job not defined
            msg = ErrorNumbers.FB412.value + \
                ', in method `model_file` : no `job` defined for experiment'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        model_file = self._job.model_file

        # Display content so researcher can copy
        try:
            if display:
                with open(model_file) as file:
                    content = file.read()
                    file.close()
                    print(content)
        except OSError as e:
            # cannot read model file content
            msg = ErrorNumbers.FB412.value + \
                f', in method `model_file` : error when reading model file - {e}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        return self._job.model_file

    # TODO: change format of returned data (during experiment results refactor ?)
    # a properly defined structure/class instead of the generic responses
    @exp_exceptions
    def check_model_status(self) -> Responses:
        """ Method for checking model status, ie whether it is approved or not by the nodes

        Returns:
            Model status for answering nodes

        Raises:
            FedbiomedExperimentError: bad argument type
        """
        # at this point, self._job exists (initialized in constructor)
        if self._job is None:
            # cannot check model status if job not defined
            msg = ErrorNumbers.FB412.value + \
                ', in method `check_model_status` : no `job` defined for experiment'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # always returns a `Responses()` object
        responses = self._job.check_model_is_approved_by_nodes()

        return responses

    # Breakpoint functions

    @exp_exceptions
    def breakpoint(self) -> None:
        """
        Saves breakpoint with the state of the training at a current round. The following Experiment attributes will
        be saved:
          - round_current
          - round_limit
          - tags
          - experimentation_folder
          - aggregator
          - node_selection_strategy
          - training_data
          - training_args
          - model_args
          - model_path
          - model_class
          - aggregated_params
          - job (attributes returned by the Job, aka job state)

        Raises:
            FedbiomedExperimentError: experiment not fully defined, experiment did not run any round yet, or error when
                saving breakpoint
        """
        # at this point, we run the constructor so all object variables are defined

        # check pre-requisistes for saving a breakpoint
        #
        # need to have run at least 1 round to save a breakpoint
        if self._round_current < 1:
            msg = ErrorNumbers.FB413.value + \
                ' - need to run at least 1 before saving a breakpoint'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        elif self._fds is None:
            msg = ErrorNumbers.FB413.value + \
                ' - need to define `training_data` for saving a breakpoint'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        elif self._node_selection_strategy is None:
            msg = ErrorNumbers.FB413.value + \
                ' - need to define `strategy` for saving a breakpoint'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        elif self._job is None:
            msg = ErrorNumbers.FB413.value + \
                ' - need to define `job` for saving a breakpoint'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

            # conditions are met, save breakpoint
        breakpoint_path, breakpoint_file_name = \
            choose_bkpt_file(self._experimentation_folder, self._round_current - 1)

        state = {
            'training_data': self._fds.data(),
            'training_args': self._training_args.dict(),
            'model_args': self._model_args,
            'model_path': self._job.model_file,  # only in Job we always model saved to a file
            # with current version
            'model_class': self._job.model_class,  # not always available properly
            # formatted in Experiment with current version
            'round_current': self._round_current,
            'round_limit': self._round_limit,
            'experimentation_folder': self._experimentation_folder,
            'aggregator': self._aggregator.save_state(),  # aggregator state
            'node_selection_strategy': self._node_selection_strategy.save_state(),
            # strategy state
            'tags': self._tags,
            'aggregated_params': self._save_aggregated_params(
                self._aggregated_params, breakpoint_path),
            'job': self._job.save_state(breakpoint_path)  # job state
        }

        # rewrite paths in breakpoint : use the links in breakpoint directory
        state['model_path'] = create_unique_link(
            breakpoint_path,
            # - Need a file with a restricted characters set in name to be able to import as module
            'model_' + str("{:04d}".format(self._round_current - 1)), '.py',
            # - Prefer relative path, eg for using experiment result after
            # experiment in a different tree
            os.path.join('..', os.path.basename(state["model_path"]))
        )

        # save state into a json file.
        breakpoint_file_path = os.path.join(breakpoint_path, breakpoint_file_name)
        try:
            with open(breakpoint_file_path, 'w') as bkpt:
                json.dump(state, bkpt)
            logger.info(f"breakpoint for round {self._round_current - 1} saved at " +
                        os.path.dirname(breakpoint_file_path))
        except (OSError, ValueError, TypeError, RecursionError) as e:
            # - OSError: heuristic for catching open() and write() errors
            # - see json.dump() documentation for documented errors for this call
            msg = ErrorNumbers.FB413.value + f' - save failed with message {str(e)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

    @classmethod
    @exp_exceptions
    def load_breakpoint(cls: Type[_E],
                        breakpoint_folder_path: Union[str, None] = None) -> _E:
        """
        Loads breakpoint (provided a breakpoint has been saved)
        so experience can be resumed. Useful if training has crashed
        researcher side or if user wants to resume experiment.

        Args:
          cls: Experiment class
          breakpoint_folder_path: path of the breakpoint folder. Path can be absolute or relative eg:
            "var/experiments/Experiment_xxxx/breakpoints_xxxx". If None, loads latest breakpoint of the latest
            experiment. Defaults to None.

        Returns:
            Reinitialized experiment object. With given object, user can then use `.run()` method to pursue model
                training.

        Raises:
            FedbiomedExperimentError: bad argument type, error when reading breakpoint or bad loaded breakpoint
                content (corrupted)
        """
        # check parameters type
        if not isinstance(breakpoint_folder_path, str) and breakpoint_folder_path is not None:
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                f'`breakpoint_folder_path` has bad type {type(breakpoint_folder_path)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # get breakpoint folder path (if it is None) and state file
        breakpoint_folder_path, state_file = find_breakpoint_path(breakpoint_folder_path)
        breakpoint_folder_path = os.path.abspath(breakpoint_folder_path)

        try:
            with open(os.path.join(breakpoint_folder_path, state_file), "r") as f:
                saved_state = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # OSError: heuristic for catching file access issues
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                f'reading breakpoint file failed with message {str(e)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        if not isinstance(saved_state, dict):
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                f'breakpoint file seems corrupted. Type should be `dict` not {type(saved_state)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # retrieve breakpoint training data
        bkpt_fds = saved_state.get('training_data')
        # keeping bkpt_fds a dict so that FederatedDataSet will be instantiated
        # in Experiment.__init__() applying some type checks.
        # More checks to verify the structure/content of saved_state.get('training_data')
        # should be added in FederatedDataSet.__init__() when refactoring it
        bkpt_fds = FederatedDataSet(bkpt_fds)
        # retrieve breakpoint sampling strategy
        bkpt_sampling_strategy_args = saved_state.get("node_selection_strategy")

        bkpt_sampling_strategy = cls._create_object(bkpt_sampling_strategy_args, data=bkpt_fds)

        # retrieve federator
        bkpt_aggregator_args = saved_state.get("aggregator")
        bkpt_aggregator = cls._create_object(bkpt_aggregator_args)

        # initializing experiment

        loaded_exp = cls(tags=saved_state.get('tags'),
                         nodes=None,  # list of previous nodes is contained in training_data
                         training_data=bkpt_fds,
                         aggregator=bkpt_aggregator,
                         node_selection_strategy=bkpt_sampling_strategy,
                         round_limit=saved_state.get("round_limit"),
                         model_class=saved_state.get("model_class"),
                         model_path=saved_state.get("model_path"),
                         model_args=saved_state.get("model_args"),
                         training_args=saved_state.get("training_args"),
                         save_breakpoints=True,
                         experimentation_folder=saved_state.get('experimentation_folder')
                         )

        # changing `Experiment` attributes
        loaded_exp._set_round_current(saved_state.get('round_current'))

        # TODO: checks when loading parameters
        model_instance = loaded_exp.model_instance()
        if model_instance is None:
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                'breakpoint file seems corrupted, `model_instance` is None'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        else:
            loaded_exp._aggregated_params = loaded_exp._load_aggregated_params(
                saved_state.get('aggregated_params'),
                model_instance.load
            )

        # changing `Job` attributes
        loaded_exp._job.load_state(saved_state.get('job'))
        # nota: exceptions should be handled in Job, when refactoring it

        logger.info(f"Experimentation reload from {breakpoint_folder_path} successful!")
        return loaded_exp

    @staticmethod
    @exp_exceptions
    def _save_aggregated_params(aggregated_params_init: dict, breakpoint_path: str) -> Dict[int, dict]:
        """Extract and format fields from aggregated_params that need to be saved in breakpoint.

        Creates link to the params file from the `breakpoint_path` and use them to reference the params files.

        Args:
            breakpoint_path: path to the directory where breakpoints files and links will be saved

        Returns:
            Extract from `aggregated_params`

        Raises:
            FedbiomedExperimentError: bad arguments type
        """
        # check arguments type, though is should have been done before
        if not isinstance(aggregated_params_init, dict):
            msg = ErrorNumbers.FB413.value + ' - save failed. ' + \
                f'Bad type for aggregated params, should be `dict` not {type(aggregated_params_init)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        if not isinstance(breakpoint_path, str):
            msg = ErrorNumbers.FB413.value + ' - save failed. ' + \
                f'Bad type for breakpoint path, should be `str` not {type(breakpoint_path)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        aggregated_params = {}
        for key, value in aggregated_params_init.items():
            if not isinstance(value, dict):
                msg = ErrorNumbers.FB413.value + ' - save failed. ' + \
                    f'Bad type for aggregated params item {str(key)}, ' + \
                    f'should be `dict` not {type(value)}'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
            params_path = create_unique_file_link(breakpoint_path,
                                                  value.get('params_path'))
            aggregated_params[key] = {'params_path': params_path}

        return aggregated_params

    @staticmethod
    @exp_exceptions
    def _load_aggregated_params(aggregated_params: Dict[str, dict], func_load_params: Callable
                                ) -> Dict[int, dict]:
        """Reconstruct experiment's aggregated params.

        Aggregated parameters structure from a breakpoint. It is identical to a classical `_aggregated_params`.

        Args
            aggregated_params: JSON formatted aggregated_params extract from a breakpoint
            func_load_params: function for loading parameters from file to aggregated params data structure

        Returns:
            Reconstructed aggregated params from breakpoint

        Raises:
            FedbiomedExperimentError: bad arguments type
        """
        # check arguments type
        if not isinstance(aggregated_params, dict):
            msg = ErrorNumbers.FB413.value + ' - load failed. ' + \
                f'Bad type for aggregated params, should be `dict` not {type(aggregated_params)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        if not callable(func_load_params):
            msg = ErrorNumbers.FB413.value + ' - load failed. ' + \
                'Bad type for aggregated params loader function, ' + \
                f'should be `Callable` not {type(func_load_params)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

            # needed for iteration on dict for renaming keys
        keys = [key for key in aggregated_params.keys()]
        # JSON converted all keys from int to string, need to revert
        try:
            for key in keys:
                aggregated_params[int(key)] = aggregated_params.pop(key)
        except (TypeError, ValueError):
            msg = ErrorNumbers.FB413.value + ' - load failed. ' + \
                f'Bad key {str(key)} in aggregated params, should be convertible to int'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        for aggreg in aggregated_params.values():
            aggreg['params'] = func_load_params(aggreg['params_path'], to_params=True)
            # errors should be handled in training plan loader function

        return aggregated_params

    # TODO: factorize code with Job and node
    @staticmethod
    @exp_exceptions
    def _create_object(args: Dict[str, Any], **object_kwargs: dict) -> Any:
        """
        Instantiate a class object from breakpoint arguments.

        Args:
            args: breakpoint definition of a class with `class` (classname),
                `module` (module path) and optional additional parameters containing object state
            **object_kwargs: optional named arguments for object constructor

        Returns:
            Instance of the class defined by `args` with state restored from breakpoint

        Raises:
            FedbiomedExperimentError: bad object definition
        """
        # check `args` type
        if not isinstance(args, dict):
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                f'breakpoint file seems corrupted. Bad type {type(args)} for object, ' + \
                'should be a `dict`'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        module_class = args.get("class")
        module_path = args.get("module")

        # import module class
        try:
            import_str = 'from ' + module_path + ' import ' + module_class
            exec(import_str)
        # could do a `except Exception as e` as exceptions may be diverse
        # reasonable heuristic:
        except (ModuleNotFoundError, ImportError, SyntaxError, TypeError) as e:
            # ModuleNotFoundError : bad module name
            # ImportError : bad class name
            # SyntaxError : expression cannot be exec()'ed
            # TypeError : module_path or module_class are not strings
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                f'breakpoint file seems corrupted. Module import for class {str(module_class)} ' + \
                f'fails with message {str(e)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # create a class variable containing the class
        try:
            class_code = eval(module_class)
        except Exception as e:
            # can we restrict the type of exception ? difficult as
            # it may be SyntaxError, TypeError, NameError, ValueError, ArithmeticError, etc.
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                f'breakpoint file seems corrupted. Evaluating class {str(module_class)} ' + \
                f'fails with message {str(e)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # instantiate object from module
        try:
            if not object_kwargs:
                object_instance = class_code()
            else:
                object_instance = class_code(**object_kwargs)
        except Exception as e:
            # can we restrict the type of exception ? difficult as
            # it may be SyntaxError, TypeError, NameError, ValueError,
            # ArithmeticError, AttributeError, etc.
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                'breakpoint file seems corrupted. Instantiating object of class ' + \
                f'{str(module_class)} fails with message {str(e)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # load breakpoint state for object
        object_instance.load_state(args)
        # note: exceptions for `load_state` should be handled in training plan

        return object_instance



    @exp_exceptions
    def model_approve(self,
                      model,
                      description: str = "no description provided",
                      nodes: list = [],
                      timeout: int = 5) -> dict:
        """Send a model and a ApprovalRequest message to node(s).

        This is a simple redirect to the Requests.model_approve() method.

        If a list of node id(s) is provided, the message will be individually sent
        to all nodes of the list.
        If the node id(s) list is None (default), the message is broadcast to all nodes.

        Args:
            model: the model to upload and send to the nodes for approval.
                   It can be:
                   - a path_name (str)
                   - a model (class)
                   - an instance of a model (TrainingPlan instance)
            nodes: list of nodes (specified by their UUID)
            timeout: maximum waiting time for the answers

        Returns:
            a dictionary of pairs (node_id: status), where status indicates to the researcher
            that the model has been correctly downloaded on the node side.
            Warning: status does not mean that the model is approved, only that it has been added
            to the "approval queue" on the node side.
        """
        return self._reqs.model_approve(model,
                                        description,
                                        nodes,
                                        timeout)

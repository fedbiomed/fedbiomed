import os
import sys
import json
import inspect
from typing import Callable, Union, Dict, Any, TypeVar, Type, List

from tabulate import tabulate
from pathvalidate import sanitize_filename, sanitize_filepath
from re import findall
import traceback

from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedExperimentError, FedbiomedError, \
    FedbiomedSilentTerminationError
from fedbiomed.researcher.environ import environ
from fedbiomed.common.fedbiosklearn import SGDSkLearnModel
from fedbiomed.common.torchnn import TorchTrainingPlan
from fedbiomed.researcher.filetools import create_exp_folder, choose_bkpt_file, \
    create_unique_link, create_unique_file_link, find_breakpoint_path
from fedbiomed.researcher.aggregators.fedavg import FedAverage
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.strategies.strategy import Strategy
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.monitor import Monitor


_E = TypeVar("Experiment")  # only for typing

# for checking class passed to experiment
# TODO : should we move this to common/constants.py ?
training_plans = (TorchTrainingPlan, SGDSkLearnModel)
# for typing only
TrainingPlan = TypeVar('TrainingPlan', TorchTrainingPlan, SGDSkLearnModel)
Type_TrainingPlan = TypeVar('Type_TrainingPlan', Type[TorchTrainingPlan], Type[SGDSkLearnModel])



# Exception handling at top lever for researcher ---------------------------------------------

def exp_exceptions(function):
    """Decorator for handling all exceptions in the Experiment class() :
    pretty print a message for the user, quit Experiment.
    """
    # try to guess if running in a notebook 
    def in_notebook():
        try:
            # not imported, just for checking
            pyshell = get_ipython().__class__.__name__
            if pyshell == 'ZMQInteractiveShell':
                # in a notebook
                return True
            else:
                return False
        except NameError:
            # not defined : we are not running in ipython, thus not in notebook
            return False

    # wrap the original function catching the exceptions
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
            logger.critical(f'Fed-BioMed researcher stopped due to exception:\n{str(e)}')
        except BaseException as e:
            code = 3
            print(
                '\n--------------------',
                f'Fed-BioMed researcher stopped due to unknown error:\n{str(e)}',
                '\nThis is either an error not yet caught by Fed-BioMed or a bug',
                'More details in the backtrace extract below',
                '--------------------',
                sep=os.linesep)
            # at most 5 backtrace entries to avoid too long output 
            traceback.print_exc(limit=5, file=sys.stdout)
            print('--------------------')
            logger.critical(f'Fed-BioMed stopped due to unknown error:\n{str(e)}')

        if code != 0:
            if in_notebook():
                # raise a silent specific exception, don't exit the interactive kernel
                raise FedbiomedSilentTerminationError
            else:
                # exit the process
                sys.exit(code)

        return ret
    return payload


# Experiment ---------------------------------------------------------------------------------

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
                round_limit: int = 1,
                model_class: Union[Type_TrainingPlan, str, None] = None,
                model_path: Union[str, None] = None,
                model_args: dict = {},
                training_args: dict = {},
                save_breakpoints: bool = False,
                tensorboard: bool = False,
                experimentation_folder: Union[str, None] = None
                ):

        """ Constructor of the class.

        Args:
            - tags (Union[List[str], str, None], optional): list of string with data tags
                or string with one data tag.
                Empty list of tags ([]) means any dataset is accepted, it is different from
                None (tags not set, cannot search for training_data yet).
                Default to None.
            - nodes (Union[List[str], None], optional): list of node_ids to filter the nodes
                to be involved in the experiment.
                Defaults to None (no filtering).
            - training_data (Union[FederatedDataSet, dict, None], optional):
                * If it is a FederatedDataSet object, use this value as training_data.
                * else if it is a dict, create and use a FederatedDataSet object from the dict
                  and use this value as training_data. The dict should use node ids as keys,
                  values being list of dicts (each dict representing a dataset on a node).
                * else if it is None (no training data provided)
                  - if `tags` is not None, set training_data by
                    searching for datasets with a query to the nodes using `tags` and `nodes`
                  - if `tags` is None, set training_data to None (no training_data set yet,
                    experiment is not fully initialized and cannot be launched)
                Defaults to None (query nodes for dataset if `tags` is not None, set training_data
                to None else)
            - aggregator (Union[Aggregator, Type[Aggregator], None], optional):
                object or class defining the method for aggregating local updates.
                Default to None (use `FedAverage` for aggregation)
            - node_selection_strategy (Union[Strategy, Type[Strategy], None], optional):
                object or class defining how nodes are sampled at each round
                for training, and how non-responding nodes are managed.
                Defaults to None:
                - use `DefaultStrategy` if training_data is initialized
                - else strategy is None (cannot be initialized), experiment cannot
                  be launched yet
            - round_limit (int, optional): the total number of training rounds
                (nodes <-> central server) of the experiment.
                Defaults to 1.
            - model_class (Union[Type_TrainingPlan, str, None], optional): name of the model class
                (`str`) or model class (`Type_TrainingPlan`) to use for training.
                For experiment to be properly and fully defined `model_class` needs to be:
                - a `str` when `model_path` is not None (model class comes from a file).
                - a `Type_TrainingPlan` when `model_path` is None (model class passed
                as argument).
                Defaults to None (no model class defined yet)
            - model_path (Union[str, None], optional) : path to a file containing
                model code (`str`) or None (no file containing model code, `model_class`
                needs to be a class matching `Type_TrainingPlan`)
                Defaults to None. 
            - model_args (dict, optional): contains model arguments passed to the constructor
                of the training plan when instantiating it : output and input feature
                dimension, etc.
                Defaults to {}.
            - training_args (dict, optional): contains training arguments passed to the 
                `training_routine` of the training plan when launching it:
                lr, epochs, batch_size...
                Defaults to {}.
            - save_breakpoints (bool, optional): whether to save breakpoints or
                not after each training round. Breakpoints can be used for resuming
                a crashed experiment.
                Defaults to False.
            - tensorboard (bool, optional): whether to save scalar values 
                for displaying in Tensorboard during training for each node.
                Currently it is only used for loss values.
                * If it is true, monitor instantiates a `Monitor` object that write
                  scalar logs into `./runs` directory.
                * If it is False, it stops monitoring if it was active.
                Defaults to False.
            - experimentation_folder (Union[str, None], optional):
                choose a specific name for the
                folder where experimentation result files and breakpoints are stored.
                This should just contain the name for the folder not a path.
                The name is used as a subdirectory of `environ[EXPERIMENTS_DIR])`.
                Defaults to None (auto-choose a folder name)
                - Caveat : if using a specific name this experimentation will not be
                automatically detected as the last experimentation by `load_breakpoint`
                - Caveat : do not use a `experimentation_folder` name finishing
                with numbers ([0-9]+) as this would confuse the last experimentation
                detection heuristic by `load_breakpoint`.
        """

        # set self._tags and self._nodes
        self.set_tags(tags)
        self.set_nodes(nodes)

        # Useless to add a param and setter/getter for Requests() as it is a singleton ?
        self._reqs = Requests()

        # set self._fds: type Union[FederatedDataSet, None]
        self.set_training_data(training_data)

        # set self._aggregator : type Aggregator
        self.set_aggregator(aggregator)

        # set self._node_selection_strategy: type Union[Strategy, None]
        self.set_strategy(node_selection_strategy)

        # "current" means number of rounds already trained
        self._round_current = 0
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

        # set self._model_args and self._training_args to dict
        self.set_model_args(model_args)
        self.set_training_args(training_args)
        
        # set self._job to Union[Job, None]
        self.set_job()

        # TODO: rewrite after experiment results refactoring
        self._aggregated_params = {}

        self.set_save_breakpoints(save_breakpoints)
        self.set_monitor(tensorboard)


    # destructor
    @exp_exceptions
    def __del__(self):
        # TODO: confirm placement for finishing monitoring - should be at the end of the experiment

        # _monitor may not exist (early del == constructor could not complete - will this happen ?)
        try:
            if self._monitor is not None:
                # stop writing in SummaryWriters
                self._reqs.remove_monitor_callback()
                # Close SummaryWriters for tensorboard
                self._monitor.close_writer()
        except AttributeError:
            # no monitor to finish, if not yet defined
            pass


    # Getters ---------------------------------------------------------------------------------------------------------

    @exp_exceptions
    def tags(self) -> Union[List[str], None]:
        return self._tags

    @exp_exceptions
    def nodes(self) -> Union[List[str], None]:
        return self._nodes

    @exp_exceptions
    def training_data(self) -> Union[FederatedDataSet, None]:
        return self._fds

    @exp_exceptions
    def aggregator(self) -> Aggregator:
        return self._aggregator

    @exp_exceptions
    def strategy(self) -> Union[Strategy, None]:
        return self._node_selection_strategy

    @exp_exceptions
    def round_limit(self) -> int:
        return self._round_limit

    @exp_exceptions
    def round_current(self):
        return self._round_current

    @exp_exceptions
    def experimentation_folder(self) -> str:
        return self._experimentation_folder

    # derivative from experimentation_folder
    @exp_exceptions
    def experimentation_path(self) -> str:
        return os.path.join(environ['EXPERIMENTS_DIR'], self._experimentation_folder)

    @exp_exceptions
    def model_class(self) -> Union[Type_TrainingPlan, str, None]:
        return self._model_class

    @exp_exceptions
    def model_path(self) -> Union[str, None]:
        return self._model_path

    @exp_exceptions
    def model_args(self) -> dict:
        return self._model_args

    @exp_exceptions
    def training_args(self) -> dict:
        return self._training_args

    @exp_exceptions
    def job(self) -> Union[Job, None]:
        return self._job

    @exp_exceptions
    def save_breakpoints(self) -> bool:
        return self._save_breakpoints

    @exp_exceptions
    def monitor(self) -> Union[Monitor, None]:
        return self._monitor


    # TODO: update these getters after experiment results refactor / job refactor 

    @exp_exceptions
    def aggregated_params(self) -> dict:
        return self._aggregated_params

    @exp_exceptions
    def training_replies(self) -> Union[dict, None]:
        # at this point `job` is defined but may be None
        if self._job is None:
            logger.error('No `job` defined for experiment, cannot get `training_replies`')
            return None
        else:
            return self._job.training_replies

    # TODO: better checking of model object type in Job() to guarantee it is a TrainingPlan
    @exp_exceptions
    def model_instance(self) -> Union[TrainingPlan, None]:
        # at this point `job` is defined but may be None
        if self._job is None:
            logger.error('No `job` defined for experiment, cannot get `model_instance`')
            return None
        else:
            return self._job.model


    # a specific getter-like
    @exp_exceptions
    def info(self) -> None:
        """Pretty print information about status of the current experiment.
        
        Method lists on the standard output all the parameters/arguments of the
        experiment and inform user whether the the experiment can be run now.

        Raises:
            - FedbiomedExperimentError: unconsistant experiment (missing variables)

        """
        # at this point all attributes are initialized (in constructor)
        info = {
            'Arguments': [
                'Tags', 'Nodes filter', 'Training Data',
                'Aggregator', 'Strategy', 'Job',
                'Model Path', 'Model Class',
                'Model Arguments', 'Training Arguments', 
                'Rounds already run', 'Rounds total',
                'Experiment folder', 'Experiment Path',
                'Breakpoint State', 'Monitoring'
                ],
            # max 40 characters per column for values - can we do that with tabulate() ?
            'Values': [ '\n'.join(findall('.{1,40}', str(e))) for e in
                    [
                    self._tags, self._nodes, self._fds,
                    self._aggregator, self._node_selection_strategy, self._job,
                    self._model_path, self._model_class,
                    self._model_args, self._training_args,
                    self._round_current, self._round_limit,
                    self._experimentation_folder,
                    self.experimentation_path(),
                    self._save_breakpoints, self._monitor
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
            except:
                # should not happen, all eval variables should be defined
                msg = ErrorNumbers.FB400.value + \
                    f', in method `info` : self.{key} not defined for experiment'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        if missing:
            print(f'\nExperiment cannot be run (not fully defined), missing :\n{missing}')
        else:
            print('\nExperiment can be run now (fully defined)')



    # Setters ---------------------------------------------------------------------------------------------------------

    @exp_exceptions
    def set_tags(self, tags: Union[List[str], str, None]) -> Union[List[str], None]:
        """ Setter for tags + verifications on argument type

        Args:
            - tags (Union[List[str], str, None]): list of string with data tags
                or string with one data tag.
                Empty list of tags ([]) means any dataset is accepted, it is different from
                None (tags not set, cannot search for training_data yet).

        Raises:
            - FedbiomedExperimentError : bad tags type

        Returns :
            - tags (Union[List[str], None])
        """
        if isinstance(tags, list):
            self._tags = tags
            for tag in tags:
                if not isinstance(tag, str):
                    msg = ErrorNumbers.FB410.value + f' `tags` : list of {type(tag)}'
                    logger.critical(msg)
                    raise FedbiomedExperimentError(msg)
        elif isinstance(tags, str):
            self._tags = [tags]
        elif tags is None:
            self._tags = tags
        else:
            msg = ErrorNumbers.FB410.value + f' `tags` : {type(tags)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        # self._tags always exist at this point

        # self._fds doesn't always exist at this point
        try:
            if self._fds is not None:
                logger.debug('Experimentation tags changed, you may need to update `training_data`')
        except AttributeError:
            # nothing to do if not defined yet
            pass

        return self._tags


    @exp_exceptions
    def set_nodes(self, nodes: Union[List[str], None]) -> Union[List[str], None]:
        """ Setter for nodes + verifications on argument type

        Args:
            - nodes (Union[List[str], None]): list of node_ids to filter the nodes
                to be involved in the experiment.

        Raises:
            - FedbiomedExperimentError : bad nodes type

        Returns:
            - nodes (Union[List[str], None])
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
            self._nodes = None
            msg = ErrorNumbers.FB410.value + f' `nodes` : {type(nodes)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        # self._nodes always exist at this point

        # self._fds doesn't always exist at this point
        try:
            if self._fds is not None:
                logger.debug('Experimentation nodes filter changed, you may need to update `training_data`')
        except AttributeError:
            # nothing to do if not defined yet
            pass
        
        return self._nodes


    @exp_exceptions
    def set_training_data(self, training_data: Union[FederatedDataSet, dict, None]) -> \
            Union[FederatedDataSet, None]:
        """ Setter for training data for federated training + verification on arguments type

        Args:
            - training_data (Union[FederatedDataSet, dict, None]):
                * If it is a FederatedDataSet object, use this value as training_data.
                * else if it is a dict, create and use a FederatedDataSet object from the dict
                  and use this value as training_data. The dict should use node ids as keys,
                  values being list of dicts (each dict representing a dataset on a node).
                * else if it is None (no training data provided)
                  - if `tags` is not None, set training_data by
                    searching for datasets with a query to the nodes using `tags` and `nodes`
                  - if `tags` is None, set training_data to None (no training_data set yet,
                    experiment is not fully initialized and cannot be launched)

        Raises:
            - FedbiomedExperimentError : bad training_data type

        Returns:
            - nodes (Union[FederatedDataSet, None])
        """
        # we can trust _reqs _tags _nodes are existing and properly typed/formatted

        # case where no training data are passed
        if training_data is None:
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
            self._fds = None
            msg = ErrorNumbers.FB410.value + f' `training_data` : {type(training_data)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        else:
            self._fds = None
            logger.debug('Experiment not fully configured yet: no training data')
        # at this point, self._fds is either None or a FederatedDataSet object
        
        # self._strategy and self._job don't always exist at this point
        try:
            if self._node_selection_strategy is not None:
                logger.debug('Training data changed, '
                    'you may need to update `node_selection_strategy`')
        except AttributeError:
            # nothing to do if not defined yet
            pass
        try:
            if self._job is not None:
                logger.debug('Training data changed, you may need to update `job`')
        except AttributeError:
            # nothing to do if not defined yet
            pass

        return self._fds


    @exp_exceptions
    def set_aggregator(self, aggregator: Union[Aggregator, Type[Aggregator], None]) -> \
            Aggregator:
        """ Setter for aggregator + verification on arguments type

        Args:
            - aggregator (Union[Aggregator, Type[Aggregator], None]):
                object or class defining the method for aggregating local updates.
                Default to None (use `FedAverage` for aggregation)
        
        Raises:
            - FedbiomedExperimentError : bad aggregator type

        Returns:
            - aggregator (Aggregator)
        """

        if aggregator is None:
            # default aggregator
            self._aggregator = FedAverage()
        elif inspect.isclass(aggregator):
            # a class is provided, need to instantiate an object
            if issubclass(aggregator, Aggregator):
                self._aggregator = aggregator()
            else:
                # bad argument, need to provide an Aggregator class
                self._aggregator = FedAverage() # be robust if we continue execution
                msg = ErrorNumbers.FB410.value + f' `aggregator` : {aggregator} class'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg) 
        elif isinstance(aggregator, Aggregator):
            # an object of a proper class is provided, nothing to do
            self._aggregator = aggregator
        else:
            # other bad type or object
            self._aggregator = FedAverage() # be robust if we continue execution
            msg = ErrorNumbers.FB410.value + f' `aggregator` : {type(aggregator)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        
        # at this point self._aggregator is (non-None) aggregator object
        return self._aggregator


    @exp_exceptions
    def set_strategy(self, node_selection_strategy: Union[Strategy, Type[Strategy], None]) -> \
            Union[Strategy, None]:
        """ Setter for `node_selection_strategy` + verification on arguments type

        Args:
            - node_selection_strategy (Union[Strategy, Type[Strategy], None]):
                object or class defining how nodes are sampled at each round
                for training, and how non-responding nodes are managed.
                Defaults to None:
                - use `DefaultStrategy` if training_data is initialized
                - else strategy is None (cannot be initialized), experiment cannot
                  be launched yet

        Raises:
            - FedbiomedExperimentError : bad strategy type

        Returns:
            - node_selection_strategy (Union[Strategy, None])
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
                    # bad argument, need to provide a Strategy class
                    self._node_selection_strategy = DefaultStrategy(self._fds) # be robust
                    msg = ErrorNumbers.FB410.value + \
                        f' `node_selection_strategy` : {node_selection_strategy} class'
                    logger.critical(msg)
                    raise FedbiomedExperimentError(msg)
            elif isinstance(node_selection_strategy, Strategy):
                # an object of a proper class is provided, nothing to do
                self._node_selection_strategy = node_selection_strategy
            else:
                # other bad type or object
                self._node_selection_strategy = DefaultStrategy(self._fds) # be robust
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
    def set_round_limit(self, round_limit: int) -> int:
        """Setter for `round_limit` + verification on arguments type

        Args:
            - round_limit (int): the total maximum number of training rounds
                (nodes <-> central server) of the experiment.

        Raise:
            - FedbiomedExperimentError : bad rounds type

        Returns:
            - round_limit (int)
        """
        # at this point round_current exists and is an int >= 0
        if not isinstance(round_limit, int):
            msg = ErrorNumbers.FB410.value + f' `round_limit` : {type(round_limit)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)            
        else:
            # at this point round_limit is an int
            if round_limit < self._round_current:
                # self._round_limit can't be less than current round
                logger.error(f'cannot set `round_limit` to less than number of already run rounds '
                    f'({self._round_current})')
            else:
                self._round_limit = round_limit

        # at this point self._round_limit is an int
        return self._round_limit


    # no setter for self._round_current eg
    #def set_round_current(self, round_current: int) -> int:
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
            - round_current (int): the number of already completed training rounds
                in the experiment.

        Raise:
            - FedbiomedExperimentError : bad round_current type or value

        Returns:
            - round_current (int)
        """
        # at this point self._round_current exists and is an int >= 0
        if not isinstance(round_current, int):
            msg = ErrorNumbers.FB410.value + f' `round_current` : {type(round_current)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)            
        else:
            # at this point self._round_limit is an int
            if round_current < 0 or round_current > self._round_limit:
                msg = ErrorNumbers.FB410.value + f' `round_current` : {round_current}'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg) 
            else:
                # correct value
                self._round_current = round_current

        # at this point self._round_limit is an int
        return self._round_current


    @exp_exceptions
    def set_experimentation_folder(self, experimentation_folder: Union[str, None]) -> str:
        """Setter for `experimentation_folder` + verification on arguments type

        Args:
            - experimentation_folder (Union[str, None]): 

        Raise:
            - FedbiomedExperimentError : bad experimentation_folder type

        Returns:
            - experimentation_folder (str)
        """
        if experimentation_folder is None:
            self._experimentation_folder = create_exp_folder()
        elif isinstance(experimentation_folder, str):
            sanitized_folder = sanitize_filename(experimentation_folder, platform='auto')
            self._experimentation_folder = create_exp_folder(sanitized_folder)

            if(sanitized_folder != experimentation_folder):
                logger.warning(f'`experimentation_folder` was sanitized from '
                    f'{experimentation_folder} to {sanitized_folder}')
        else:
            msg = ErrorNumbers.FB410.value + \
                f' `experimentation_folder` : {type(experimentation_folder)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)           
        
        # at this point self._experimentation_folder is a str valid for a foldername

        # _job doesn't always exist at this point
        try:
            if self._job is not None:
                logger.debug('Experimentation folder changed, you may need to update `job`')
        except AttributeError:
            # nothing to do if not defined yet
            pass

        return self._experimentation_folder


    @exp_exceptions
    def set_model_class(self, model_class: Union[Type_TrainingPlan, str, None]) -> \
            Union[Type_TrainingPlan, str, None]:
        """Setter for `model_class` + verification on arguments type

        Args:
            - model_class (Union[Type_TrainingPlan, str, None]): name of the model class
                (`str`) or model class (`Type_TrainingPlan`) to use for training.
                For experiment to be properly and fully defined `model_class` needs to be:
                - a `str` when `model_path` is not None (model class comes from a file).
                - a `Type_TrainingPlan` when `model_path` is None (model class passed
                as argument).
                Defaults to None (no model class defined yet)

        Raise:
            - FedbiomedExperimentError : bad model_class type

        Returns:
            - model_class (Union[Type_TrainingPlan, str, None])
        """
        if model_class is None:
            self._model_class = None
            self._model_is_defined = False
        elif isinstance(model_class, str):
            if str.isidentifier(model_class):
                # correct python identifier
                self._model_class = model_class
                # model_path may not be defined at this point
                try:
                    # valid model definition if we use model_path
                    self._model_is_defined = isinstance(self._model_path, str)
                except AttributeError:
                    # we don't set model_is_defined to True because
                    # model_path is not defined (!= defined to None ...)
                    self._model_is_defined = False
            else:
                # bad identifier
                self._model_class = None # be robust if we continue execution
                self._model_is_defined = False
                msg = ErrorNumbers.FB410.value + f' `model_class` : {model_class} bad identifier'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)                
        elif inspect.isclass(model_class):
            # model_class must be a subclass of a valid training plan
            if issubclass(model_class, training_plans):
                # valid class
                self._model_class = model_class
                # model_path may not be defined at this point
                try:
                    # valid model definition if we don't use model_path
                    self._model_is_defined = self._model_path is None
                except AttributeError:
                    # we don't set model_is_defined to True because
                    # model_path is not defined (!= defined to None ...)
                    self._model_is_defined = False
            else:
                # bad class
                self._model_class = None # be robust if we continue execution
                self._model_is_defined = False
                msg = ErrorNumbers.FB410.value + f' `model_class` : {model_class} class'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        else:
            # bad type
            self._model_class = None # be robust if we continue execution
            self._model_is_defined = False
            msg = ErrorNumbers.FB410.value + f' `model_class` : type(model_class)'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)            

        # self._model_is_defined and self._model_class always exist at this point
        try:
            self._model_path # raise exception if not defined
            if not self._model_is_defined:
                logger.debug(f'Experiment not fully configured yet: no valid model, '
                    f'model_class={self._model_class} model_path={self._model_path}')
        except AttributeError:
            # we don't want to issue a warning is model_path not initialized yet
            # (== still initializing the class)
            pass

        # _job doesn't always exist at this point
        try:
            if self._job is not None:
                logger.debug('Experimentation model_class changed, you may need to update `job`')
        except AttributeError:
            # nothing to do if not defined yet
            pass

        return self._model_class
        

    @exp_exceptions
    def set_model_path(self, model_path: Union[str, None]) -> Union[str, None]:
        """Setter for `model_path` + verification on arguments type

        Args:
            - model_path (Union[str, None]) : path to a file containing
                model code (`str`) or None (no file containing model code, `model_class`
                needs to be a class matching `Type_TrainingPlan`) 

        Raise:
            - FedbiomedExperimentError : bad model_path type

        Returns:
            - model_path (Union[str, None])
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
                self._model_path = None # be robust if we continue execution
                self._model_is_defined = inspect.isclass(self._model_class)
                msg = ErrorNumbers.FB410.value + \
                    f' `model_path` : {model_path} is not a sane path to an existing file'
                logger.critical(msg)
                raise FedbiomedExperimentError(msg)
        else:
            # bad type
            self._model_path = None # be robust if we continue execution
            self._model_is_defined = inspect.isclass(self._model_class)
            msg = ErrorNumbers.FB410.value + f' `model_path` : type(model_path)'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # self._model_path is also defined at this point
        if not self._model_is_defined:
            logger.debug(f'Experiment not fully configured yet: no valid model, '
                f'model_class={self._model_class} model_path={self._model_path}')

        # _job doesn't always exist at this point
        try:
            if self._job is not None:
                logger.debug('Experimentation model_path changed, you may need to update `job`')
        except AttributeError:
            # nothing to do if not defined yet
            pass

        return self._model_path
        

    # TODO: model_args need checking of dict items, to be done by Job and node
    # (using a training plan method ?)
    @exp_exceptions
    def set_model_args(self, model_args: dict):
        """Setter for `model_args` + verification on arguments type

        Args:
            - model_args (dict): contains model arguments passed to the constructor
                of the training plan when instantiating it : output and input feature
                dimension, etc.

        Raise:
            - FedbiomedExperimentError : bad model_args type

        Returns:
            - model_args (dict)
        """
        if isinstance(model_args, dict):
            self._model_args = model_args
        else:
            # bad type
            self._model_args = {}
            msg = ErrorNumbers.FB410.value + f' `model_args` : {type(model_args)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        # self._model_args always exist at this point

        # _job doesn't always exist at this point
        try:
            if self._job is not None:
                logger.debug('Experimentation model_args changed, you may need to update `job`')
        except AttributeError:
            # nothing to do if not defined yet
            pass

        return self._model_args


    # TODO: training_args need checking of dict items, to be done by Job and node
    # (using a training plan method ? changing `training_routine` prototype ?)
    @exp_exceptions
    def set_training_args(self, training_args: dict):
        """Setter for `training_args` + verification on arguments type

        Args:
            - training_args (dict): contains training arguments passed to the 
                `training_routine` of the training plan when launching it:
                lr, epochs, batch_size...

        Raise:
            - FedbiomedExperimentError : bad training_args type

        Returns:
            - training_args (dict)
        """
        if isinstance(training_args, dict):
            self._training_args = training_args
        else:
            # bad type
            self._training_args = {}
            msg = ErrorNumbers.FB410.value + f' `training_args` : {type(training_args)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)            
        # self._training_args always exist at this point

        # _job doesn't always exist at this point
        try:
            if self._job is not None:
                # job setter function exists, use it
                self._job.training_args = self._training_args
                logger.debug('Experimentation training_args updated for `job`')
        except AttributeError:
            # nothing to do if not defined yet
            pass

        return self._training_args


    # we could also handle `set_job(self, Union[Job, None])` but is it useful as
    # job is initialized with arguments that can be set ?
    @exp_exceptions
    def set_job(self) -> Union[Job, None]:
        """Setter for job, it verifies pre-requisites are met for creating a job
        attached to this experiment. If yes, instantiate a job ; if no, return None.

        Returns:
            - job (Union[Job, None])
        """
        # at this point all are defined among:
        # self.{_reqs,_fds,_model_is_defined,_model_class,_model_path,_model_args,_training_args}
        # self._experimentation_folder => self.experimentation_path()
        # self._round_current

        # _job may not be defined at this point
        try:
            if self._job is not None:
                # a job is already defined, and it may also have run some rounds
                logger.debug(f'Experimentation `job` changed after running '
                    '{self._round_current} rounds, may give inconsistent results')
        except:
            # nothing to do if not defined yet
            pass

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
            - save_breakpoints (bool): whether to save breakpoints or
                not after each training round. Breakpoints can be used for resuming
                a crashed experiment.
        
        Raises:
            - FedbiomedExperimentError : bad save_breakpoints type

        Returns:
            - save_breakpoints (bool)
        """
        if isinstance(save_breakpoints, bool):
            self._save_breakpoints = save_breakpoints
            # no warning if done during experiment, we may change breakpoint policy at any time
        else:
            msg = ErrorNumbers.FB410.value + f' `save_breakpoints` : {type(save_breakpoints)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        return self._save_breakpoints


    # TODO: accept an optional Monitor param (`monitor: Monitor = None`)
    @exp_exceptions
    def set_monitor(self, tensorboard: bool) -> Union[Monitor, None]:
        """ Setter for monitoring in tensorboard + verification on arguments type

        Args:
            - tensorboard (bool): whether to save scalar values 
                for displaying in Tensorboard during training for each node.
                Currently it is only used for loss values.
                * If it is true, monitor instantiates a `Monitor` object that write
                  scalar logs into `./runs` directory.
                * If it is False, it stops monitoring if it was active.
        
        Raises:
            - FedbiomedExperimentError : bad tensorboard type

        Returns:
            - monitor (Union[Monitor, None])
        """
        if isinstance(tensorboard, bool):
            # do nothing if setting is the same as active configuration
            action = True 
            try:
                if self._monitor is not None and tensorboard:
                    action = False
                    logger.info('Experimentation monitoring is already active, nothing to do')
                if self._monitor is None and not tensorboard:
                    action = False
                    logger.info('Experimentation monitoring is already inactive, nothing to do')
            except AttributeError:
                pass

            # Q: should we issue a warning if activating monitoring during an experiment ?

            if action:
                #  set monitoring loss values with tensorboard
                if tensorboard:
                    self._monitor = Monitor()
                    self._reqs.add_monitor_callback(self._monitor.on_message_handler)
                else:
                    self._monitor = None
                    # Remove callback. Since request class is singleton callback
                    # function might be already added into request before.
                    self._reqs.remove_monitor_callback()
        else:
            # bad type
            self._reqs.remove_monitor_callback()
            msg = ErrorNumbers.FB410.value + f' `tensorboard` : {type(tensorboard)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # self._monitor exists at this point
        return self._monitor



    # Run experiment functions -------------------------------------------------------------------

    @exp_exceptions
    def run_once(self, increase: bool = False) -> int:
        """Run at most one round of an experiment, continuing from the point the
        experiment had reached.
        If the maximum number of rounds `round_limit` of the experiment is already reached:
        * if `increase` is False, do nothing
        * if `increase` is True, increment total number of round `round_limit` and run one round

        Args:
            - increase (bool, optional) : automatically increase the `round_limit` of the 
              experiment if needed
              Defaults to False
        
        Raises:
            - FedbiomedExperimentError : bad argument type or value
        
        Returns:
            - real rounds (int) : number of rounds really run

        """
        # check increase is a boolean
        if not isinstance(increase, bool):
            msg = ErrorNumbers.FB410.value + \
                f', in method `run_once` param `increase` : type {type(increase)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # nota: robustify test, but we should never have self._round_current > self._round_limit
        if self._round_current >= self._round_limit:
            if increase is True:
                logger.info(f'Auto increasing total rounds for experiment from {self._round_limit} '
                        f'to {self._round_current + 1}')
                self._round_limit = self._round_current + 1
            else:
                logger.info(f'Round limit of {self._round_limit} was reached, stopping execution')
                return 0

        # at this point, self._aggregator always exists and is not None
        # self.{_node_selection_strategy,_job} exist but may be None

        # check pre-requisites are met for running a round
        #for component in (self._node_selection_strategy, self._job):
        if self._node_selection_strategy is None:
            msg = ErrorNumbers.FB411.value + f', missing `node_selection_strategy`'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        elif self._job is None:
            msg = ErrorNumbers.FB411.value + f', missing `job`'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # Ready to execute a training round using the job, strategy and aggregator

        # Sample nodes using strategy (if given)
        self._job.nodes = self._node_selection_strategy.sample_nodes(self._round_current)
        logger.info('Sampled nodes in round ' + str(self._round_current) + ' ' + str(self._job.nodes))
        # Trigger training round on sampled nodes
        self._job.start_nodes_training_round(round=self._round_current)

        # refining/normalizing model weights received from nodes
        model_params, weights = self._node_selection_strategy.refine(
            self._job.training_replies[self._round_current], self._round_current)

        # aggregate model from nodes to a global model
        aggregated_params = self._aggregator.aggregate(model_params,
                                                       weights)
        # write results of the aggregated model in a temp file
        aggregated_params_path = self._job.update_parameters(aggregated_params)
        logger.info(f'Saved aggregated params for round {self._round_current} '
            f'in {aggregated_params_path}')

        self._aggregated_params[self._round_current] = {'params': aggregated_params,
                                                        'params_path': aggregated_params_path}

        self._round_current += 1
        if self._save_breakpoints:
            self.breakpoint()
        return 1


    @exp_exceptions
    def run(self, run_rounds: int = 0, increase: bool = False) -> int:
        """Run one or more rounds of an experiment, continuing from the point the
        experiment had reached.

        Args:
            - run_rounds (int, optional): Number of experiment rounds to run in this call.
              * 0 means "run all the rounds remaining in the experiment" computed as
                maximum rounds (`round_limit`) minus the number of rounds already run
                rounds (`round_current`)
              * >= 1 means "run `run_rounds` rounds" at most.
                If it goes beyond the maximum rounds `round_limit` of the experiment:
                - if `increase` is True,
                  increase the `round_limit` to (`round_current` + `run_rounds`)
                  and run `run_rounds` rounds
                - if `increase` is False, run (`round_limit` - `round_current`)
                  rounds and don't modify the maximum `round_limit` of the experiment
              Defaults to 0
            - increase (bool, optional) : automatically increase the maximum
              number of rounds `round_limit` of the experiment if needed
              Defaults to False
        
        Raises:
            - FedbiomedExperimentError : bad argument type or value
        
        Returns:
            - real rounds (int) : number of rounds really run

        """
        # check run_rounds is a >=0 integer
        if not isinstance(run_rounds, int):
            msg = ErrorNumbers.FB410.value + \
                f', in method `run` param `run_rounds` : type {type(run_rounds)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)      
        elif run_rounds < 0:
            msg = ErrorNumbers.FB410.value + \
                f', in method `run` param `run_rounds` : value {run_rounds}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        # check increase is a boolean
        if not isinstance(increase, bool):
            msg = ErrorNumbers.FB410.value + \
                f', in method `run` param `increase` : type {type(increase)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)               

        # compute maximum number of rounds to run
        if run_rounds == 0:
            # all remaining rounds in the experiment
            run_rounds = self._round_limit - self._round_current
        else:
            # dont change run_rounds, but extend self._round_limit if necessary
            if increase is True:
                if (self._round_current + run_rounds) > self._round_limit:
                    logger.info(f'Auto increasing total rounds for experiment from {self._round_limit} '
                        f'to {self._round_current + run_rounds}')
                self._round_limit = max (self._round_limit, self._round_current + run_rounds)

        # run the rounds
        real_rounds = 0
        for _ in range(run_rounds):
            increment = self.run_once(increase)
            if increment == 0:
                logger.info(f'Only {real_rounds} were run out of requested {run_rounds} rounds')
                break
            real_rounds += increment

        return real_rounds



    # Model checking functions -------------------------------------------------------------------

    @exp_exceptions
    def model_file(self, display: bool = True) -> str:
        """ This method displays saved final model for the experiment
            that will be sent to the nodes for training.

        Args:
            - display (bool): If `True`, prints content of the model file.
            Default is `True`

        Raises:
            - FedbiomedExperimentError: bad argument type, or cannot read model file content

        Returns:
            - model_file (str) : path to model file
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
                f', in method `model_file` : no `job` defined for experiment'
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
        """ Method for checking model status, ie whether it is approved or
            not by the nodes

        Raises:
            - FedbiomedExperimentError: bad argument type

        Returns:
            - responses (str) : model status for answering nodes
        """
        # at this point, self._job exists (initialized in constructor)         
        if self._job is None:
            # cannot check model status if job not defined
            msg = ErrorNumbers.FB412.value + \
                f', in method `check_model_status` : no `job` defined for experiment'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # always returns a `Responses()` object
        responses = self._job.check_model_is_approved_by_nodes()

        return responses


    # Breakpoint functions ----------------------------------------------------------------

    @exp_exceptions
    def breakpoint(self) -> None:
        """
        Saves breakpoint with the state of the training at a current round.
        The following Experiment attributes will be saved:
          - round_number
          - round_number_due
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
          - FedbiomedExperimentError: experiment not fully defined ; experiment did not run any
            round yet ; error when saving breakpoint
        """
        # at this point, we run the constructor so all object variables are defined

        # check pre-requisistes for saving a breakpoint
        #
        # need to have run at least 1 round to save a breakpoint
        if self._round_current < 1:
            msg = ErrorNumbers.FB413.value + \
                f' - need to run at least 1 before saving a breakpoint'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        elif self._fds is None:
            msg = ErrorNumbers.FB413.value + \
                f' - need to define `training_data` for saving a breakpoint'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        elif self._node_selection_strategy is None:
            msg = ErrorNumbers.FB413.value + \
                f' - need to define `strategy` for saving a breakpoint'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)  
        elif self._job is None:
            msg = ErrorNumbers.FB413.value + \
                f' - need to define `job` for saving a breakpoint'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)        


        # conditions are met, save breakpoint
        breakpoint_path, breakpoint_file_name = \
            choose_bkpt_file(self._experimentation_folder, self._round_current - 1)

        state = {
            'training_data': self._fds.data(),
            'training_args': self._training_args,
            'model_args': self._model_args,
            'model_path': self._job.model_file,  # only in Job we always model saved to a file
            # with current version
            'model_class': self._job.model_class,  # not always available properly
            # formatted in Experiment with current version
            'round_number': self._round_current,
            'round_number_due': self._round_limit,
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
            logger.info(f"breakpoint for round {self._round_current - 1} saved at " + \
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
          - cls (Type[_E]): Experiment class
          - breakpoint_folder_path (Unione[str, None], optional): path of the breakpoint folder.
            Path can be absolute or relative eg: "var/experiments/Experiment_xxxx/breakpoints_xxxx".
            If None, loads latest breakpoint of the latest experiment.
            Defaults to None.

        Raises: 
          - FedbiomedExperimentError: bad argument type ; error when reading breakpoint ; 
            bad loaded breakpoint content (corrupted)

        Returns:
          - _E: Reinitialized experiment. With given object,
            user can then use `.run()` method to pursue model training.
        """
        # check parameters type
        if not isinstance(breakpoint_folder_path, str) and breakpoint_folder_path is not None:
            msg = ErrorNumbers.FB413.value + f' - load failed, ' + \
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
            msg = ErrorNumbers.FB413.value + f' - load failed, ' + \
                f'reading breakpoint file failed with message {str(e)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        if not isinstance(saved_state, dict):
            msg = ErrorNumbers.FB413.value + f' - load failed, ' + \
                f'breakpoint file seems corrupted. Type should be `dict` not {type(saved_state)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # -----  retrieve breakpoint training data ---
        bkpt_fds = saved_state.get('training_data')
        # keeping bkpt_fds a dict so that FederatedDataSet will be instantiated
        # in Experiment.__init__() applying some type checks.
        # More checks to verify the structure/content of saved_state.get('training_data')
        # should be added in FederatedDataSet.__init__() when refactoring it

        # -----  retrieve breakpoint sampling strategy ----
        bkpt_sampling_strategy_args = saved_state.get("node_selection_strategy")
        bkpt_sampling_strategy = cls._create_object(bkpt_sampling_strategy_args, data=bkpt_fds)

        # ----- retrieve federator -----
        bkpt_aggregator_args = saved_state.get("aggregator")
        bkpt_aggregator = cls._create_object(bkpt_aggregator_args)

        # ------ initializing experiment -------

        loaded_exp = cls(tags=saved_state.get('tags'),
                         nodes=None,  # list of previous nodes is contained in training_data
                         training_data=bkpt_fds,
                         aggregator=bkpt_aggregator,
                         node_selection_strategy=bkpt_sampling_strategy,
                         round_limit=saved_state.get("round_number_due"),
                         model_class=saved_state.get("model_class"),
                         model_path=saved_state.get("model_path"),
                         model_args=saved_state.get("model_args"),
                         training_args=saved_state.get("training_args"),
                         save_breakpoints=True,
                         experimentation_folder=saved_state.get('experimentation_folder')
                         )

        # ------- changing `Experiment` attributes -------
        loaded_exp._set_round_current(saved_state.get('round_number'))

        #TODO: checks when loading parameters
        model_instance = loaded_exp.model_instance()
        if model_instance is None:
            msg = ErrorNumbers.FB413.value + f' - load failed, ' + \
                f'breakpoint file seems corrupted, `model_instance` is None'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        else:
            loaded_exp._aggregated_params = loaded_exp._load_aggregated_params(
                saved_state.get('aggregated_params'),
                model_instance.load
            )

        # ------- changing `Job` attributes -------
        loaded_exp._job.load_state(saved_state.get('job'))
        # nota: exceptions should be handled in Job, when refactoring it

        logger.info(f"Experimentation reload from {breakpoint_folder_path} successful!")
        return loaded_exp


    @staticmethod
    @exp_exceptions
    def _save_aggregated_params(aggregated_params_init: dict, breakpoint_path: str) -> Dict[int, dict]:
        """Extracts and format fields from aggregated_params that need
        to be saved in breakpoint. Creates link to the params file from the `breakpoint_path`
        and use them to reference the params files.

        Args:
            - breakpoint_path (str): path to the directory where breakpoints files
                and links will be saved

        Raises:
            - FedbiomedExperimentError: bad arguments type

        Returns:
            - Dict[int, dict] : extract from `aggregated_params`
        """
        # check arguments type, though is should have been done before
        if not isinstance(aggregated_params_init, dict):
            msg = ErrorNumbers.FB413.value + f' - save failed. ' + \
                f'Bad type for aggregated params, should be `dict` not {type(aggregated_params_init)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        if not isinstance(breakpoint_path, str):
            msg = ErrorNumbers.FB413.value + f' - save failed. ' + \
                f'Bad type for breakpoint path, should be `str` not {type(breakpoint_path)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)           

        aggregated_params = {}
        for key, value in aggregated_params_init.items():
            if not isinstance(value, dict):
                msg = ErrorNumbers.FB413.value + f' - save failed. ' + \
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
        """Reconstruct experiment results aggregated params structure
        from a breakpoint so that it is identical to a classical `_aggregated_params`

        Args:
            - aggregated_params (Dict[str, dict]) : JSON formatted aggregated_params
              extract from a breakpoint
            - func_load_params (Callable) : function for loading parameters
              from file to aggregated params data structure

        Returns:
            - Dict[int, dict] : reconstructed aggregated params from breakpoint
        """
        # check arguments type
        if not isinstance(aggregated_params, dict):
            msg = ErrorNumbers.FB413.value + f' - load failed. ' + \
                f'Bad type for aggregated params, should be `dict` not {type(aggregated_params)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        if not callable(func_load_params):
            msg = ErrorNumbers.FB413.value + f' - load failed. ' + \
                f'Bad type for aggregated params loader function, ' + \
                f'should be `Callable` not {type(func_load_params)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)        

        # needed for iteration on dict for renaming keys
        keys = [key for key in aggregated_params.keys()]
        # JSON converted all keys from int to string, need to revert
        try:
            for key in keys:
                aggregated_params[int(key)] = aggregated_params.pop(key)
        except (TypeError, ValueError) as e:
            msg = ErrorNumbers.FB413.value + f' - load failed. ' + \
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
    def _create_object(args: Dict[str, Any], **object_kwargs) -> Any:
        """
        Instantiate a class object from breakpoint arguments.

        Args:
            - args (Dict[str, Any]) : breakpoint definition of a class with `class` (classname),
              `module` (module path) and optional additional parameters containing object state
            - **object_kwargs : optional named arguments for object constructor

        Raises:
            - FedbiomedExperimentError: bad object definition

        Returns:
            - Any: instance of the class defined by `args` with state restored from breakpoint
        """
        # check `args` type
        if not isinstance(args, dict):
            msg = ErrorNumbers.FB413.value + f' - load failed, ' + \
                f'breakpoint file seems corrupted. Bad type {type(args)} for object, ' + \
                f'should be a `dict`'
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
            msg = ErrorNumbers.FB413.value + f' - load failed, ' + \
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
            msg = ErrorNumbers.FB413.value + f' - load failed, ' + \
                f'breakpoint file seems corrupted. Evaluating class {str(module_class)} ' + \
                f'fails with message {str(e)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # instantiate object from module
        try:
            if object_kwargs is None:
                object_instance = class_code()
            else:
                object_instance = class_code(**object_kwargs)
        except Exception as e:
            # can we restrict the type of exception ? difficult as
            # it may be SyntaxError, TypeError, NameError, ValueError,
            # ArithmeticError, AttributeError, etc.
            msg = ErrorNumbers.FB413.value + f' - load failed, ' + \
                f'breakpoint file seems corrupted. Instantiating object of class ' + \
                f'{str(module_class)} fails with message {str(e)}'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # load breakpoint state for object
        object_instance.load_state(args)
        # note: exceptions for `load_state` should be handled in training plan

        return object_instance

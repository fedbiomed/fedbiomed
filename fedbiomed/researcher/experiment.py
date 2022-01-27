import logging
import os
import json
import inspect
from typing import Callable, Union, Dict, Any, TypeVar, Type, List

from tabulate import tabulate
from pathvalidate import sanitize_filename

from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers
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
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.monitor import Monitor

_E = TypeVar("Experiment")  # only for typing


class Experiment(object):
    """
    This class represents the orchestrator managing the federated training
    """

    def __init__(self,
                tags: Union[List[str], str, None] = None,
                nodes: Union[List[str], None] = None,
                training_data: Union[FederatedDataSet, dict, None] = None,
                aggregator: Union[Aggregator, Type[Aggregator], None] = None,
                node_selection_strategy: Union[Strategy, Type[Strategy], None] = None,
                rounds: int = 1,
                model_class: Union[Type[Callable], Callable] = None,
                model_path: str = None,
                model_args: dict = {},
                training_args: dict = None,
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
                  - set training_data by
                    searching for datasets with a query to the nodes using `tags` and `nodes`
                    (if `tags` is set)
                  - or set training_data to None (no training_data set yet,
                    experiment is not fully initialized and cannot be launched)
                Defaults to None (query nodes for dataset if `tags` is set, set training_data
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
            - rounds (int, optional): the total number of training rounds
                (nodes <-> central server) of the experiment.
                Defaults to 1.
            - model_class (Union[Type[Callable], Callable], optional): name or
                instance (object) of the model class to use
                for training.
                Should be a str type when using jupyter notebook
                or a Callable when using a simple python
                script.
            - model_path (string, optional) : path to file containing model code
            - model_args (dict, optional): contains output and input feature
                dimension. Defaults to None.
            - training_args (dict, optional): contains training parameters:
                lr, epochs, batch_size...
                Defaults to None.
            - save_breakpoints (bool, optional): whether to save breakpoints or
                not. Breakpoints can be used
                for resuming a crashed
                experiment. Defaults to False.
            - tensorboard (bool): Tensorboard flag for displaying scalar values
                during training in every node. If it is true,
                monitor will write scalar logs into
                `./runs` directory.
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
        self.set_rounds(rounds)

        # set self._experimentation_folder: type str
        self.set_experimentation_folder(experimentation_folder)
        # Note: currently keep this parameter as it cannot be updated in Job()
        # without refactoring Job() first

        self._model_class = model_class
        self._model_path = model_path
        self._model_args = model_args
        self._training_args = training_args

        status, _ = self._before_job_init()
        if status:
            self._job = Job(reqs=self._reqs,
                            model=self._model_class,
                            model_path=self._model_path,
                            model_args=self._model_args,
                            training_args=self._training_args,
                            data=self._fds,
                            keep_files_dir=self.experimentation_path())
        else:
            self._job = None

        self._aggregated_params = {}
        self._save_breakpoints = save_breakpoints

        #  Monitoring loss values with tensorboard
        if tensorboard:
            self._monitor = Monitor()
            self._reqs.add_monitor_callback(self._monitor.on_message_handler)
        else:
            self._monitor = None
            # Remove callback. Since reqeust class is singleton callback
            # function might be already added into request before.
            self._reqs.remove_monitor_callback()

    # Getters ---------------------------------------------------------------------------------------------------------

    def tags(self):
        return self._tags

    def nodes(self):
        return self._nodes

    def training_data(self):
        return self._fds

    def aggregator(self):
        return self._aggregator

    def strategy(self):
        return self._node_selection_strategy

    def rounds(self):
        return self._rounds

    def round_current(self):
        return self._round_current

    def experimentation_folder(self):
        return self._experimentation_folder

    def experimentation_path(self):
        return os.path.join(environ['EXPERIMENTS_DIR'], self._experimentation_folder)




    def training_replies(self):
        return self._job.training_replies

    def aggregated_params(self):
        return self._aggregated_params

    def job(self):
        return self._job

    def model_instance(self):
        return self._job.model

    def model_args(self):
        return self._model_args

    def training_args(self):
        return self._training_args

    def model_path(self):
        return self._model_path

    def model_class(self):
        return self._model_class

    def monitor(self):
        return self._monitor

    def breakpoint(self):
        return self._save_breakpoints


    # Setters ---------------------------------------------------------------------------------------------------------

    def set_tags(self, tags: Union[List[str], str, None]):
        """ Setter for tags + verifications on argument type

        Args:
            - tags (Union[List[str], str, None]): list of string with data tags
                or string with one data tag.
                Empty list of tags ([]) means any dataset is accepted, it is different from
                None (tags not set, cannot search for training_data yet).

        Raises:
            - TypeError : bad tags type

        Returns :
            - tags (Union[List[str], str, None])
        """
        if isinstance(tags, list):
            self._tags = tags
            for tag in tags:
                if not isinstance(tag, str):
                    self._tags = [] # robust default, in case we try to continue execution
                    raise TypeError(ErrorNumbers.FB421.value % f'list of {type(tag)}')
        elif isinstance(tags, str):
            self._tags = [tags]
        elif tags is None:
            self._tags = tags
        else:
            self._tags = None # robust default, in case we try to continue execution
            logger.error(ErrorNumbers.FB421.value % type(tags))
            raise TypeError(ErrorNumbers.FB421.value % type(tags))

        return self._tags


    def set_nodes(self, nodes: Union[List[str], None]):
        """ Setter for nodes + verifications on argument type

        Args:
            - nodes (Union[List[str], None]): list of node_ids to filter the nodes
                to be involved in the experiment.

        Raises:
            - TypeError : bad nodes type

        Returns:
            - nodes (Union[List[str], None])
        """
        if isinstance(nodes, list):
            self._nodes = nodes
            for node in nodes:
                if not isinstance(node, str):
                    self._nodes = None # robust default
                    raise TypeError(ErrorNumbers.FB422.value % f'list of {type(node)}')
        elif nodes is None:
            self._nodes = nodes
        else:
            self._nodes = None
            logger.error(ErrorNumbers.FB422.value % type(nodes))
            raise TypeError(ErrorNumbers.FB422.value % type(nodes))

        return self._nodes


    def set_training_data(self, training_data: Union[FederatedDataSet, dict, None]):
        """ Setter for training data for federated training + verification on arguments type

        Args:
            - training_data (Union[FederatedDataSet, dict, None]):
                * If it is a FederatedDataSet object, use this value as training_data.
                * else if it is a dict, create and use a FederatedDataSet object from the dict
                  and use this value as training_data. The dict should use node ids as keys,
                  values being list of dicts (each dict representing a dataset on a node).
                * else if it is None (no training data provided),
                  search for datasets by a query to the nodes using `tags` and `nodes`

        Raises:
            - TypeError : bad training_data type

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
            logger.error((ErrorNumbers.FB420.value % type(training_data)))
            raise TypeError(ErrorNumbers.FB420.value % type(training_data))
        else:
            self._fds = None
            logger.warning('Experiment not fully configured yet: no training data')
        # at this point, self._fds is either None or a FederatedDataSet object
        
        # strategy and job don't always exist at this point
        try:
            if self._node_selection_strategy is not None:
                logger.warning('Training data changed, '
                    'you may need to update `node_selection_strategy`')
        except AttributeError:
            # nothing to do if not defined yet
            pass
        try:
            if self._job is not None:
                logger.warning('Training data changed, you may need to update `job`')
        except AttributeError:
            # nothing to do if not defined yet
            pass

        return self._fds


    def set_aggregator(self, aggregator: Union[Aggregator, Type[Aggregator], None]):
        """ Setter for aggregator + verification on arguments type

        Args:
            - aggregator (Union[Aggregator, Type[Aggregator], None], optional):
                object or class defining the method for aggregating local updates.
                Default to None (use `FedAverage` for aggregation)
        
        Raises:
            - TypeError : bad aggregator type

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
                logger.error(ErrorNumbers.FB419.value % f'{aggregator} class')
                raise TypeError(ErrorNumbers.FB419.value % f'{aggregator} class')
        elif isinstance(aggregator, Aggregator):
            # an object of a proper class is provided, nothing to do
            self._aggregator = aggregator
        else:
            # other bad type or object
            self._aggregator = FedAverage() # be robust if we continue execution
            logger.error(ErrorNumbers.FB419.value % type(aggregator))
            raise TypeError(ErrorNumbers.FB419.value % type(aggregator))
        
        # at this point self._aggregator is (non-None) aggregator object
        return self._aggregator


    def set_strategy(self, node_selection_strategy: Union[Strategy, Type[Strategy], None]):
        """ Setter for `node_selection_strategy` + verification on arguments type

        Args:
            - node_selection_strategy (Union[Strategy, Type[Strategy], None], optional):
                object or class defining how nodes are sampled at each round
                for training, and how non-responding nodes are managed.
                Defaults to None:
                - use `DefaultStrategy` if training_data is initialized
                - else strategy is None (cannot be initialized), experiment cannot
                  be launched yet

        Raises:
            - TypeError : bad strategy type

        Returns:
            - node_selection_strategy (Union[Strategy, None]
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
                    logger.error(ErrorNumbers.FB418.value % f'{node_selection_strategy} class')
                    raise TypeError(ErrorNumbers.FB418.value % f'{node_selection_strategy} class')
            elif isinstance(node_selection_strategy, Strategy):
                # an object of a proper class is provided, nothing to do
                self._node_selection_strategy = node_selection_strategy
            else:
                # other bad type or object
                self._node_selection_strategy = DefaultStrategy(self._fds) # be robust
                logger.error(ErrorNumbers.FB418.value % type(node_selection_strategy))
                raise TypeError(ErrorNumbers.FB418.value % type(node_selection_strategy))
        else:
            # cannot initialize strategy if not FederatedDataSet yet
            self._node_selection_strategy = None
            logger.warning('Experiment not fully configured yet: no node selection strategy')

        # at this point self._node_selection_strategy is a Union[Strategy, None]
        return self._node_selection_strategy


    def set_rounds(self, rounds: int):
        """Setter for `rounds` + verification on arguments type

        Args:
            - rounds (int, optional): the total number of training rounds
                (nodes <-> central server) of the experiment.

        Raise:
            - TypeError : bad rounds type

        Returns:
            - rounds (int)
        """
        # at this point round_current exists and is an int >= 0
        if not isinstance(rounds, int):
            self._rounds = max(1, self._round_current) # robust default
            raise TypeError(ErrorNumbers.FB417.value % type(rounds))
        else:
            # at this point rounds is an int
            self._rounds = max(rounds, self._round_current)
            if rounds < self._round_current:
                # self._rounds can't be less than current round
                # need to change round_current if really wanted to set this value
                logger.warning(f'`rounds` cannot be less than `round_current`: '
                    f'setting `rounds`={self._round_current}')

        # at this point self._rounds is an int
        return self._rounds


    # no setter for self._round_current eg
    #def set_round_current(self, round_current: int):
    # ...
    #
    # - does not make sense to increase `self._round_current` == padding with "non existing" rounds,
    #   would need to invent some dummy data for strategy, experiment results, etc.
    # - erasing rounds is complicated: not only decreasing `self._round_current)`, need
    #   to clean some experiment results (aggregated_params, job.training_replies, ...),
    #   change state of aggregator, strategy, etc... == the proper way of doing it is to
    #   load a breakpoint


    def set_experimentation_folder(self, experimentation_folder: Union[str, None]):
        """Setter for `experimentation_folder` + verification on arguments type

        Args:
            - experimentation_folder (Union[str, None], optional): 

        Raise:
            - TypeError : bad experimentation_folder type

        Returns:
            - experimentation_folder (str)
        """
        if experimentation_folder is None:
            self._experimentation_folder = create_exp_folder()
        elif isinstance(experimentation_folder, str):
            sanitized_folder = sanitize_filename(experimentation_folder)
            self._experimentation_folder = create_exp_folder(sanitized_folder)

            if(sanitized_folder != experimentation_folder):
                logger.warning(f'`experimentation_folder` was sanitized from '
                    f'{experimentation_folder} to {sanitized_folder}')
        else:
            self._experimentation_folder = create_exp_folder() # robust default
            raise TypeError(ErrorNumbers.FB416.value % type(experimentation_folder))            
        
        # at this point self._experimentation_folder is a str valid for a foldername

        # _job doesn't always exist at this point
        try:
            if self._job is not None:
                logger.warning('Experimentation folder changed, you may need to update `job`')
        except AttributeError:
            # nothing to do if not defined yet
            pass

        return self._experimentation_folder



    def set_model_args(self, model_args: Dict):
        """ Setter for Model Arguments. This method should also update/set model arguments in
        Job object.

        Args:
            model_args (dict): Model arguments
        """

        # TODO: Job uses model arguments in init method for building TrainingPlan (Model Class).
        # After Job has initialized setting new model arguments will require to reinitialize the job.
        # Job needs to be refactored to avoid rebuild after the arguments have changed.
        self._model_args = model_args
        if self._job:
            logger.info('Model arguments has been changed, please update the Job by running `.set_job()`')

        return

    def set_training_args(self, training_args):

        """ Setter for training arguments. Updates the Job object with new
            training arguments.

        Args:
            training_args (dict): Training arguments
        """
        self._training_args = training_args

        # Update training arguments if job is already initialized
        if self._job:
            self._job._training_args = training_args

        return

    def set_model_path(self, model_path: str):
        """ Setter for model path. Since model path is directly connected to Job, it is required to
        run `.set_job()` after updating it. If the Job is already initialize the method will inform
        about `set_job()` should be called.

        Args:
            model_path (str): Path of python file that contains Model class
        """
        self._model_path = model_path

        # FIXME: Changing model path requires to rebuild Job (Should this method do that or User)
        if self._job:
            logger.info('Model path has been modified. You might need to update Job by running `.set_job()`')

    def set_model_class(self, model_class: Union[type[Callable], Callable, str]):
        """ Setter for model class. Since model path is used in Job in build time, if Job is already initialize
        it is required to run `.set_job()` after updating it. If the Job is already initialize the method will inform
        about `set_job()` should be called.

        Args:
            model_class (str): Path of python file that contains Model class
        """
        self._model_class = model_class

        # FIXME: Changing model class requires to rebuild Job (Should this method do this action or User)
        if self._job:
            logger.info('Model class has been modified. You might need to update Job by running `.set_job()`')


    def set_breakpoints(self, save_breakpoints: bool = True):

        """
            TODO: decide which option is better?
            breakpoints option 1: keep it as now
              def breakpoints(self) -> bool:
              def set_breakpoints(self, save_breakpoints: bool = False) -> bool:

            breakpoints option 2: implement more detail choice of breakpoints save
               - bkpt_enable True/False => save all/no breakpoints
               - bkpt_rounds List[int] => round numbers where we save breakpoints
               - bkpt_every int => save breakpoints every xxx rounds
               eg: if enable is True, or round number in bkpt_rounds, or bkpt_every
                       is not None and round is N * bkpt_every then save breakpoint
               def set_breakpoints(self,
                     bkpt_enable: bool = False,
                     bkpt_rounds: List[int] = None,
                     bkpt_every: int = None)

        """
        self._save_breakpoints = save_breakpoints

        pass

    def set_job(self):
        """ Setter for Job class. To be able to set Job, the arguments: model_path, model_class, training_data
        should be set. Otherwise, set_job() will raise critical error.

        Returns:
            None
        """
        status, messages = self._before_job_init()
        if status:
            self._job = Job(reqs=self._reqs,
                            model=self._model_class,
                            model_path=self._model_path,
                            model_args=self._model_args,
                            training_args=self._training_args,
                            data=self._fds,
                            keep_files_dir=self.experimentation_path())
            return True
        else:
            logger.critical('Error while setting Job: \n\n- %s' % '\n- '.join(messages))


    def set_monitor(self, tensorboard: bool = True, monitor: Monitor = None):
        """ Setter for monitoring loss values on Tensorboard. Currently, Monitor
        is used for only displaying loss values on Tensorboard.

        Args:
            tensorboard (bool): If it is true will build Monitor class and register a callback
            function in Request. Otherwise, it will remove callback from reqeust. Default is True
            monitor (Monitor): An instance of Monitor class. Default is None
        """
        if tensorboard:
            if monitor:
                self._monitor = monitor if isinstance(monitor, Callable) else monitor()
                self._reqs.add_monitor_callback(self._monitor.on_message_handler)
            else:
                self._monitor = Monitor()
        else:
            self._monitor = None
            # Remove callback. Since reqeust class is singleton callback
            # function might be already added into request before.
            self._reqs.remove_monitor_callback()



    # Run experiment functions -------------------------------------------------------------------

    def run_once(self):
        """ Runs the experiment only once. It will increase global round each time
        this method is called

        """

        if self._round_current >= self._rounds:
            logger.info(f'Round limit has been reached. Number of rounds: {self._rounds} and completed rounds: '
                        f'{self._round_current}. Please set higher value with `.set_rounds` or '
                        f'use `.run(rounds=<number of rounds>)`.')
            return False

        # FIXME: While running run_one with exp.run(rounds=2) second control will be useless
        # Check; are all the necessary arguments has been set for running a run
        status, messages = self._before_experiment_run()

        if status:
            # Sample nodes using strategy (if given)
            self._job.nodes = self._node_selection_strategy.sample_nodes(self._round_current)
            logger.info('Sampled nodes in round ' + str(self._round_current) + ' ' + str(self._job.nodes))
            # Trigger training round on sampled nodes
            answering_nodes = self._job.start_nodes_training_round(round=self._round_current)

            # refining/normalizing model weights received from nodes
            model_params, weights = self._node_selection_strategy.refine(
                self._job.training_replies[self._round_current], self._round_current)

            # aggregate model from nodes to a global model
            aggregated_params = self._aggregator.aggregate(model_params,
                                                           weights)
            # write results of the aggregated model in a temp file
            aggregated_params_path = self._job.update_parameters(aggregated_params)
            logger.info(f'Saved aggregated params for round {self._round_current} in {aggregated_params_path}')

            self._aggregated_params[self._round_current] = {'params': aggregated_params,
                                                            'params_path': aggregated_params_path}
            if self._save_breakpoints:
                self._save_breakpoint(self._round_current)

            if self._monitor is not None:
                # Close SummaryWriters for tensorboard
                self._monitor.close_writer()

            self._round_current += 1
        else:
            raise ValueError('Error while running the experiment: \n\n- %s' % '\n- '.join(messages))

    def run(self, rounds: int = None):
        """Runs an experiment, ie trains a model on nodes for a
        given number of rounds.
        It involves the following steps:

        Args:
            rounds (int, optional): Number of round that the experiment will run. If it is not
            provided method will use built-in rounds
        Raises:
            NotImplementedError: [description]
        Returns:
            None

        """

        # Extend round if rounds is not None and if it is needed
        if rounds:
            rounds_left = self._rounds - self._round_current
            if rounds_left < rounds:
                self._rounds = self._rounds + (rounds - rounds_left)

        for _ in range(self._rounds):
            status = self.run_once()
            if status is False:
                break

    def model_file(self, display: bool = True):

        """ This method displays saved final model for the experiment
            that will be sent to the nodes for training.

        Args:
            display (bool): If `True`, prints content of the model file. Default is `True`
        """
        model_file = self._job.model_file

        # Display content so researcher can copy
        if display:
            with open(model_file) as file:
                content = file.read()
                file.close()
                print(content)
        return self._job.model_file

    def check_model_status(self):

        """ Method for checking model status whether it is approved or
            not by the nodes
        """
        responses = self._job.check_model_is_approved_by_nodes()
        return responses

    def info(self):
        """ Information about status of the current experiment. Method  lists all the
        parameters/arguments of the experiment and inform user about the
        can the experiment be run.
        """

        info = {
            'Arguments': [
                'Tags', 'Nodes filter', 'Training Data',
                'Aggregator', 'Strategy',
                'Already run rounds', 'Total rounds',
                'Model Path', 'Model Class', 'Model Arguments',
                'Training Arguments', 
                'Job', 'Breakpoint State', 'Exp  folder',
                'Exp folder', 'Exp Path'
                ],
            'Values': [
                self._tags, self._nodes, self._fds,
                self._aggregator,  self._node_selection_strategy,
                self._round_current, self._rounds,
                self._model_path, self._model_class, self._model_args,
                self._training_args,
                self._job, self._save_breakpoint, self._experimentation_folder,
                os.path.join(environ['EXPERIMENTS_DIR'], self._experimentation_folder)
                ]
        }
        print(tabulate(info, headers='keys'))

    def _before_job_init(self):
        """ This method checks are all the necessary arguments has been set to
        initialize `Job` class.
`
        Returns:
            status, missing_attributes (bool, List)
        """
        no_none_args_msg = {"_training_args": ErrorNumbers.FB410.value,
                            "_fds": ErrorNumbers.FB411.value,
                            '_model_class': ErrorNumbers.FB412.value,
                            }

        status, messages = self._argument_controller(no_none_args_msg)

        # Model_path is not required if the model_class is a class
        # if it is string Job requires knowing here model is saved
        if self._model_path is None and isinstance(self._model_class, str):
            messages.append(ErrorNumbers.FB413.value)
            status = False

        return status, messages

    def _before_experiment_run(self):

        no_none_args_msg = {"_job": ErrorNumbers.FB414.value,
                            "_node_selection_strategy": ErrorNumbers.FB415.value,
                            }

        return self._argument_controller(no_none_args_msg)

    def _argument_controller(self, arguments: dict):

        messages = []
        for arg, message in arguments.items():
            if arg in self.__dict__ and self.__dict__[arg] is not None:
                continue
            else:
                messages.append(message)
        status = True if len(messages) == 0 else False

        return status, messages


    # Breakpoint functions ----------------------------------------------------------------

    def _save_breakpoint(self, round: int = 0):
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

        Args:
            - round (int, optional): number of rounds already executed.
              Starts from 0. Defaults to 0.
        """

        breakpoint_path, breakpoint_file_name = \
            choose_bkpt_file(self._experimentation_folder, round)

        state = {
            'training_data': self._fds.data(),
            'training_args': self._training_args,
            'model_args': self._model_args,
            'model_path': self._job.model_file,  # only in Job we always model saved to a file
            # with current version
            'model_class': self._job.model_class,  # not always available properly
            # formatted in Experiment with current version
            'round_number': round + 1,
            'round_number_due': self._rounds,
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
            'model_' + str("{:04d}".format(round)), '.py',
            # - Prefer relative path, eg for using experiment result after
            # experiment in a different tree
            os.path.join('..', os.path.basename(state["model_path"]))
        )

        # save state into a json file.
        breakpoint_file_path = os.path.join(breakpoint_path, breakpoint_file_name)
        with open(breakpoint_file_path, 'w') as bkpt:
            json.dump(state, bkpt)
        logger.info(f"breakpoint for round {round} saved at " + \
                    os.path.dirname(breakpoint_file_path))

    @classmethod
    def load_breakpoint(cls: Type[_E],
                        breakpoint_folder_path: str = None) -> _E:
        """
        Loads breakpoint (provided a breakpoint has been saved)
        so experience can be resumed. Useful if training has crashed
        researcher side or if user wants to resume experiment.

        Args:
            - cls (Type[_E]): Experiment class
            - breakpoint_folder_path (str, optional): path of the breakpoint folder.
              Path can be absolute or relative eg: "var/experiments/Experiment_xx/breakpoints_xx".
              If None, loads latest breakpoint of the latest experiment.
              Defaults to None.

        Returns:
            - _E: Reinitialized experiment. With given object,
              user can then use `.run()` method to pursue model training.
        """

        # get breakpoint folder path (if it is None) and
        # state file
        breakpoint_folder_path, state_file = find_breakpoint_path(breakpoint_folder_path)
        breakpoint_folder_path = os.path.abspath(breakpoint_folder_path)

        # TODO: check if all elements needed for breakpoint are present
        with open(os.path.join(breakpoint_folder_path, state_file), "r") as f:
            saved_state = json.load(f)

        # -----  retrieve breakpoint training data ---
        bkpt_fds = FederatedDataSet(saved_state.get('training_data'))

        # -----  retrieve breakpoint sampling strategy ----
        bkpt_sampling_strategy_args = saved_state.get("node_selection_strategy")
        bkpt_sampling_strategy = cls._create_object(bkpt_sampling_strategy_args, data=bkpt_fds)

        # ----- retrieve federator -----
        bkpt_aggregator_args = saved_state.get("aggregator")
        bkpt_aggregator = cls._create_object(bkpt_aggregator_args)

        # ------ initializing experiment -------

        loaded_exp = cls(tags=saved_state.get('tags'),
                         nodes=None,  # list of previous nodes is contained in training_data
                         model_class=saved_state.get("model_class"),
                         model_path=saved_state.get("model_path"),
                         model_args=saved_state.get("model_args"),
                         training_args=saved_state.get("training_args"),
                         rounds=saved_state.get("round_number_due"),
                         aggregator=bkpt_aggregator,
                         node_selection_strategy=bkpt_sampling_strategy,
                         save_breakpoints=True,
                         training_data=bkpt_fds,
                         experimentation_folder=saved_state.get('experimentation_folder')
                         )

        # ------- changing `Experiment` attributes -------
        loaded_exp._round_current = saved_state.get('round_number')
        loaded_exp._aggregated_params = loaded_exp._load_aggregated_params(
            saved_state.get('aggregated_params'),
            loaded_exp.model_instance.load
        )

        # ------- changing `Job` attributes -------
        loaded_exp._job.load_state(saved_state.get('job'))

        logging.info(f"experimentation reload from {breakpoint_folder_path} successful!")
        return loaded_exp

    @staticmethod
    def _save_aggregated_params(aggregated_params_init: dict, breakpoint_path: str) -> Dict[int, dict]:
        """Extracts and format fields from aggregated_params that need
        to be saved in breakpoint. Creates link to the params file from the `breakpoint_path`
        and use them to reference the params files.

        Args:
            - breakpoint_path (str): path to the directory where breakpoints files
                and links will be saved

        Returns:
            - Dict[int, dict] : extract from `aggregated_params`
        """
        aggregated_params = {}
        for key, value in aggregated_params_init.items():
            params_path = create_unique_file_link(breakpoint_path,
                                                  value.get('params_path'))
            aggregated_params[key] = {'params_path': params_path}

        return aggregated_params

    @staticmethod
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
        # needed for iteration on dict for renaming keys
        keys = [key for key in aggregated_params.keys()]
        # JSON converted all keys from int to string, need to revert
        for key in keys:
            aggregated_params[int(key)] = aggregated_params.pop(key)

        for aggreg in aggregated_params.values():
            aggreg['params'] = func_load_params(aggreg['params_path'], to_params=True)

        return aggregated_params

    # TODO: factorize code with Job and node
    # TODO: add signal handling for error cases
    @staticmethod
    def _create_object(args: Dict[str, Any], **object_kwargs) -> Callable:
        """
        Instantiate a class object from breakpoint arguments.

        Args:
            - args (Dict[str, Any]) : breakpoint definition of a class with `class` (classname),
              `module` (module path) and optional additional parameters containing object state
            - **object_kwargs : optional named arguments for object constructor

        Returns:
            - Callable: object of the class defined by `args` with state restored from breakpoint
        """
        module_class = args.get("class")
        module_path = args.get("module")
        import_str = 'from ' + module_path + ' import ' + module_class

        # import module
        exec(import_str)
        # create a class variable containing the class
        class_code = eval(module_class)
        # instantiate object from module
        if object_kwargs is None:
            object_instance = class_code()
        else:
            object_instance = class_code(**object_kwargs)

        # load breakpoint state for object
        object_instance.load_state(args)

        return object_instance

import uuid
from re import findall
from typing import Any, Dict, Optional, List, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from declearn.model.api import Vector

from fedbiomed.common.constants import ErrorNumbers, __breakpoints_version__
from fedbiomed.common.exceptions import (
    FedbiomedExperimentError
)
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.optimizers import Optimizer
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan, TorchTrainingPlan, SKLearnTrainingPlan, FederatedDataPlan
from fedbiomed.common.utils import is_ipython, raise_for_version_compatibility, __default_version__

from fedbiomed.researcher.aggregators import Aggregator, FedAverage
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.filetools import (
    choose_bkpt_file, create_unique_link, create_unique_file_link, find_breakpoint_path
)
from fedbiomed.researcher.monitor import Monitor
from fedbiomed.researcher.secagg import SecureAggregation
from fedbiomed.researcher.strategies.strategy import Strategy
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from fedbiomed.researcher.federated_workflows._federated_workflow import exp_exceptions, \
    FederatedWorkflow, Type_TrainingPlan
from fedbiomed.researcher.federated_workflows._training_job import TrainingJob

TExperiment = TypeVar("TExperiment", bound='Experiment')  # only for typing


class Experiment(FederatedWorkflow):
    """
    This class represents the orchestrator managing the federated training
    """

    @exp_exceptions
    def __init__(
        self,
        tags: Union[List[str], str, None] = None,
        nodes: Union[List[str], None] = None,
        training_data: Union[FederatedDataSet, dict, None] = None,
        aggregator: Union[Aggregator, Type[Aggregator], None] = None,
        agg_optimizer: Optional[Optimizer] = None,
        node_selection_strategy: Union[Strategy, Type[Strategy], None] = None,
        round_limit: Union[int, None] = None,
        training_plan_class: Union[Type_TrainingPlan, str, None] = None,
        training_plan_path: Union[str, None] = None,
        model_args: dict = {},
        training_args: Union[TrainingArgs, dict, None] = None,
        tensorboard: bool = False,
        experimentation_folder: Union[str, None] = None,
        secagg: Union[bool, SecureAggregation] = False,
        save_breakpoints: bool = False,
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
        """

        # predefine all class variables, so no need to write try/except
        # block each time we use it
        super().__init__(
            tags,
            nodes,
            training_data,
            training_plan_class,
            training_plan_path,
            training_args,
            experimentation_folder,
            secagg
        )
        self._node_selection_strategy = None
        self._round_limit = None
        self._monitor = None
        self._aggregator = None
        self._global_model = None
        self._agg_optimizer = None  # type: Optional[Optimizer]
        self.aggregator_args = {}
        self._aggregated_params = {}
        self._client_correction_states_dict = {}
        self._client_states_dict = {}
        self._server_state = None
        self._save_breakpoints = None
        self._training_replies: Dict = {}

        # set self._model_args and self._training_args to dict
        self.set_model_args(model_args)

        # set self._aggregator : type Aggregator
        self.set_aggregator(aggregator)

        # set self._agg_optimizer: type Optional[Optimizer]
        self.set_agg_optimizer(agg_optimizer)

        # set self._node_selection_strategy: type Union[Strategy, None]
        self.set_strategy(node_selection_strategy)

        # "current" means number of rounds already trained
        self._set_round_current(0)
        self.set_round_limit(round_limit)

        self.set_save_breakpoints(save_breakpoints)

        # always create a monitoring process
        self._monitor = Monitor()
        self._reqs.add_monitor_callback(self._monitor.on_message_handler)
        self.set_tensorboard(tensorboard)
        self.reset_model_parameters()

    def reset_model_parameters(self):
        self._raise_for_missing_job_prerequities()
        job = TrainingJob(reqs=self._reqs,
                          keep_files_dir=self.experimentation_path())
        self._training_plan = job.get_initialized_workflow_instance(self._training_plan_path,
                                                                    self._training_plan_class,
                                                                    self._training_args,
                                                                    self._model_args)
        self._global_model = self._training_plan.after_training_params()

    # destructor
    @exp_exceptions
    def __del__(self):
        # This part has been commented, self._reqs.remove_monitor_callback() removes monitor
        # callback when initializing an experiment for the second time with same name.
        # While recreating a class with same variable name python first calls __init__ and then __del__.

        # if self._reqs is not None:
        #     # TODO: confirm placement for finishing monitoring - should be at the end of the experiment
        #     self._reqs.remove_monitor_callback()

        if isinstance(self._monitor, Monitor):
            self._monitor.close_writer()

    @exp_exceptions
    def aggregator(self) -> Aggregator:
        """Retrieves aggregator class that will be used for aggregating model parameters.

        To set or update aggregator: [`set_aggregator`][fedbiomed.researcher.experiment.Experiment.set_aggregator].

        Returns:
            A class or an object that is an instance of [Aggregator][fedbiomed.researcher.aggregators.Aggregator]

        """
        return self._aggregator

    @exp_exceptions
    def agg_optimizer(self) -> Optional[Optimizer]:
        """Retrieves the optional Optimizer used to refine aggregated model updates.

        To set or update that optimizer:
        [`set_agg_optimizer`][fedbiomed.researcher.experiment.Experiment.set_agg_optimizer].

        Returns:
            An [Optimizer][fedbiomed.common.optimizers.Optimizer] instance,
            or None.
        """
        return self._agg_optimizer

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
    def model_args(self) -> dict:
        """Retrieves model arguments.

        Please see also [`set_model_args`][fedbiomed.researcher.experiment.Experiment.set_model_args]

        Returns:
            The arguments that are going to be passed to [`training_plans`][fedbiomed.common.training_plans]
                classes in built time on the node side.
        """
        return self._model_args

    @exp_exceptions
    def test_ratio(self) -> float:
        """Retrieves the ratio for validation partition of entire dataset.

        Please see also [`set_test_ratio`][fedbiomed.researcher.experiment.Experiment.set_test_ratio] to
            change/set `test_ratio`

        Returns:
            The ratio for validation part, `1 - test_ratio` is ratio for training set.
        """

        return self._training_args['test_ratio']

    @exp_exceptions
    def test_metric(self) -> Union[MetricTypes, str, None]:
        """Retrieves the metric for validation routine.

        Please see also [`set_test_metric`][fedbiomed.researcher.experiment.Experiment.set_test_metric]
            to change/set `test_metric`

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

        Please see also
            [`set_test_on_local_updates`][fedbiomed.researcher.experiment.Experiment.set_test_on_local_updates].

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
    def monitor(self) -> Monitor:
        """Retrieves the monitor object

        Monitor is responsible for receiving and parsing real-time training and validation feed-back from each node
        participate to federated training. See [`Monitor`][fedbiomed.researcher.monitor.Monitor]

        Returns:
            Monitor object that will always exist with experiment to retrieve feed-back from the nodes.
        """
        return self._monitor

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

        return self._training_replies

    @exp_exceptions
    def save_breakpoints(self) -> bool:
        """Retrieves the status of saving breakpoint after each round of training.

        Returns:
            `True`, If saving breakpoint is active. `False`, vice versa.
        """

        return self._save_breakpoints

    # a specific getter-like
    @exp_exceptions
    def info(self) -> Dict[str, Any]:
        """Prints out the information about the current status of the experiment.

        Lists  all the parameters/arguments of the experiment and informs whether the experiment can be run.

        Raises:
            FedbiomedExperimentError: Inconsistent experiment due to missing variables
        """

        # at this point all attributes are initialized (in constructor)
        info = super().info()
        info.update({
            'Arguments': [
                'Aggregator',
                'Strategy',
                'Aggregator Optimizer',
                'Model Arguments',
                'Rounds already run',
                'Rounds total',
                'Breakpoint State',
            ],
            # max 60 characters per column for values - can we do that with tabulate() ?
            'Values': ['\n'.join(findall('.{1,60}',
                                         str(e))) for e in [
                self._aggregator.aggregator_name if self._aggregator is not None else None,
                self._node_selection_strategy,
                self._agg_optimizer,
                self._model_args,
                self._round_current,
                self._round_limit,
                self._save_breakpoints,
            ]
            ]
        }
        )
        return info

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
        self.aggregator_args["aggregator_name"] = self._aggregator.aggregator_name
        if self._fds is not None:
            self._aggregator.set_fds(self._fds)

        return self._aggregator

    @exp_exceptions
    def set_agg_optimizer(
        self,
        agg_optimizer: Optional[Optimizer],
    ) -> Optional[Optimizer]:
        """Sets the optional researcher optimizer.

        Args:
            agg_optimizer: Optional fedbiomed Optimizer instance to be
                used so as to refine aggregated updates prior to applying them.
                If None, equivalent to using vanilla SGD with 1.0 learning rate.

        Returns:
            The optional researcher optimizer attached to this Experiment.

        Raises:
            FedbiomedExperimentError: if `optimizer` is of unproper type.
        """
        if not (
            agg_optimizer is None
            or isinstance(agg_optimizer, Optimizer)
        ):
            raise FedbiomedExperimentError(
                f"{ErrorNumbers.FB410.value}: 'agg_optimizer' must be an "
                f"Optimizer instance or None, not {type(agg_optimizer)}."
            )
        self._agg_optimizer = agg_optimizer
        return self._agg_optimizer

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

        # `Monitor` is not yet declared during object initialization
        if isinstance(self._monitor, Monitor):
            self._monitor.set_round(self._round_current + 1)

        # at this point self._round_current is an int
        return self._round_current

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

        return self._model_args

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
        return self._training_args['test_on_global_updates']

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

        # check pre-requisites are met for running a round
        if self._node_selection_strategy is None:
            msg = ErrorNumbers.FB411.value + ', missing `node_selection_strategy`'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        # Sample nodes for training
        training_nodes = self._node_selection_strategy.sample_nodes(self._round_current)

        self._raise_for_missing_job_prerequities()
        job = TrainingJob(reqs=self._reqs,
                          nodes=training_nodes,
                          keep_files_dir=self.experimentation_path())
        job.upload_workflow_code(self._training_plan)
        filename = job.save_params_to_file(self._training_plan)
        job.upload_parameters(filename)
        self._global_model = self._training_plan.get_model_params()

        self._aggregator.set_training_plan_type(self._training_plan.type())

        # Setup Secure Aggregation (it's a noop if not active)
        secagg_arguments = self.secagg_setup()

        # Check aggregator parameter(s) before starting a round
        self._aggregator.check_values(n_updates=self._training_args.get('num_updates'),
                                      training_plan=self._training_plan)

        aggr_args_thr_msg, aggr_args_thr_file = self._aggregator.create_aggregator_args(self._global_model,
                                                                                        training_nodes)

        # Collect auxiliary variables from the aggregates optimizer, if any.
        optim_aux_var = self._collect_optim_aux_var()

        logger.info('Sampled nodes in round ' + str(self._round_current) + ' ' + str(job.nodes))

        replies, _ = job.start_nodes_training_round(
            round_=self._round_current,
            training_args=self._training_args,
            model_args=self._model_args,
            data=self._fds,
            aggregator_args_thr_msg=aggr_args_thr_msg,
            aggregator_args_thr_files=aggr_args_thr_file,
            do_training=True,
            secagg_arguments=secagg_arguments,
            optim_aux_var=optim_aux_var,
        )
        self._training_replies[self._round_current] = replies

        # refining/normalizing model weights received from nodes
        model_params, weights, total_sample_size, encryption_factors = self._node_selection_strategy.refine(
            self._training_replies[self._round_current], self._round_current)

        self._aggregator.set_fds(self._fds)

        if self._secagg.active:
            flatten_params = self._secagg.aggregate(
                round_=self._round_current,
                encryption_factors=encryption_factors,
                total_sample_size=total_sample_size,
                model_params=model_params
            )
            # FIXME: Access TorchModel through non-private getter once it is implemented
            aggregated_params: Dict[str, Union[torch.tensor, np.ndarray]] = (
                self._training_plan._model.unflatten(flatten_params)
            )

        else:
            # aggregate models from nodes to a global model
            aggregated_params = self._aggregator.aggregate(model_params,
                                                           weights,
                                                           global_model=self._global_model,
                                                           training_plan=self._training_plan,
                                                           training_replies=self._training_replies,
                                                           node_ids=job.nodes,
                                                           n_updates=self._training_args.get('num_updates'),
                                                           n_round=self._round_current)

        # Optionally refine the aggregated updates using an Optimizer.
        self._process_optim_aux_var(job)
        aggregated_params = self._run_agg_optimizer(self._training_plan,
                                                    aggregated_params)

        self._training_plan.set_model_params(aggregated_params)
        aggregated_params_path = job.save_params_to_file(self._training_plan)
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
            aggr_args_thr_msg, aggr_args_thr_file = self._aggregator.create_aggregator_args(self._global_model,
                                                                                            training_nodes)
            job.start_nodes_training_round(round_=self._round_current,
                                           training_args=self._training_args,
                                           model_args=self._model_args,
                                           data=self._fds,
                                           aggregator_args_thr_msg=aggr_args_thr_msg,
                                           aggregator_args_thr_files=aggr_args_thr_file,
                                           do_training=False)

        return 1

    def _collect_optim_aux_var(
            self,
        ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Collect auxiliary variables of the held Optimizer, if any."""
        if self._agg_optimizer is None:
            return None
        return self._agg_optimizer.get_aux()

    def _process_optim_aux_var(
        self,
        job: TrainingJob
    ) -> None:
        """Process Optimizer auxiliary variables received during last round.

        Raises:
            FedbiomedExperimentError: if auxiliary variables were received,
                but `agg_optimizer` is None and thus cannot process them.
            FedbiomedOptimizerError: if the received auxiliary variables do
                not match the expectations of the `agg_optimizer` Optimizer.
        """
        # Collect auxiliary variables from participating nodes' replies.
        aux_var = job.extract_received_optimizer_aux_var_from_round(
            self._round_current,
            self._training_replies
        )
        # If an Optimizer is used, pass it the auxiliary variables (if any).
        if self._agg_optimizer is not None:
            self._agg_optimizer.set_aux(aux_var)
        # If no Optimizer is used but auxiliary variables were received, raise.
        elif aux_var:
            raise FedbiomedExperimentError(
                "Received auxiliary variables from 1+ node Optimizer, but "
                "no `agg_optimizer` was set for this Experiment to process "
                "them.\nThese variables come from the following plug-in "
                f"modules: {set(aux_var)}."
            )

    def _run_agg_optimizer(
        self,
        training_plan,
        aggregated_params: Dict[str, T],
    ) -> Dict[str, T]:
        """Optionally refine aggregated parameters into model updates.

        Args:
            aggregated_params: `Aggregator`-output model weights, equal
                to $$\\theta^t - aggregate(\\{\\theta_i^{t+1}\\}_{i=1}^I)$$.

        Returns:
            Updated model weights that should be used to replace the current
            `self._global_model`. If a researcher optimizer is set, they are
            obtained by taking a SGD(-based) step over the aggregated updates.
            Otherwise, the outputs are the same as the inputs.
        """
        # If no Optimizer is used, return the inputs.
        if self._agg_optimizer is None:
            return aggregated_params
        # Run any start-of-round routine.
        self._agg_optimizer.init_round()
        # Recover the aggregated model updates, wrapped as a Vector.
        # Optionally restrict weights that require updating to non-frozen ones.
        # aggregated_params = agg({w^t - sum_k(eta_{k,i,t} * grad_{k,i,t})}_i)
        # hence aggregated_params = w^t - agg(updates_i)
        # hence agg_gradients = agg_i(updates_i)
        names = set(
            training_plan.get_model_params(only_trainable=True)
        )
        init_params = Vector.build(
            {k: v for k, v in self._global_model.items() if k in names}
        )
        agg_gradients = init_params - Vector.build(
            {k: v for k, v in aggregated_params.items() if k in names}
        )
        # Take an Optimizer step to compute the updates.
        # When using vanilla SGD: agg_updates = - lrate * agg_gradients
        agg_updates = self._agg_optimizer.step(agg_gradients, init_params)
        # Return the model weights' new values after this step.
        weights = (init_params + agg_updates).coefs
        return {k: weights.get(k, v) for k, v in aggregated_params.items()}

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
          - agg_optimizer
          - node_selection_strategy
          - training_data
          - training_args
          - model_args
          - training_plan_path
          - training_plan_class
          - aggregated_params
          - job (attributes returned by the Job, aka job state)
          - secagg

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
            'breakpoint_version': str(__breakpoints_version__),
            'training_data': self._fds.data(),
            'training_args': self._training_args.dict(),
            'model_args': self._model_args,
            'training_plan_path': self._job.training_plan_file,  # only in Job we always model saved to a file
            # with current version
            'training_plan_class': self._job.training_plan_name,  # not always available properly
            # formatted in Experiment with current version
            'round_current': self._round_current,
            'round_limit': self._round_limit,
            'experimentation_folder': self._experimentation_folder,
            'aggregator': self._aggregator.save_state(breakpoint_path, global_model=self._global_model),  # aggregator state
            'agg_optimizer': self._save_optimizer(breakpoint_path),
            'node_selection_strategy': self._node_selection_strategy.save_state(),
            # strategy state
            'tags': self._tags,
            'aggregated_params': self._save_aggregated_params(
                self._aggregated_params, breakpoint_path),
            'job': self._job.save_state(breakpoint_path),  # job state
            'secagg': self._secagg.save_state()
        }

        # rewrite paths in breakpoint : use the links in breakpoint directory
        state['training_plan_path'] = create_unique_link(
            breakpoint_path,
            # - Need a file with a restricted characters set in name to be able to import as module
            'model_' + str("{:04d}".format(self._round_current - 1)), '.py',
            # - Prefer relative path, eg for using experiment result after
            # experiment in a different tree
            os.path.join('../..', os.path.basename(state["training_plan_path"]))
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
    def load_breakpoint(cls: Type[TExperiment],
                        breakpoint_folder_path: Union[str, None] = None) -> TExperiment:
        """
        Loads breakpoint (provided a breakpoint has been saved)
        so experience can be resumed. Usefull if training has crashed
        researcher side or if user wants to resume experiment.

        Args:
          cls: Experiment class
          breakpoint_folder_path: path of the breakpoint folder. Path can be absolute or relative eg:
            "var/experiments/Experiment_xxxx/breakpoints_xxxx". If None, loads latest breakpoint of the latest
            experiment. Defaults to None.

        Returns:
            Reinitialized experiment object. With given object-0.2119,  0.0796, -0.0759, user can then use `.run()`
                method to pursue model training.

        Raises:
            FedbiomedExperimentError: bad argument type, error when reading breakpoint or bad loaded breakpoint
                content (corrupted)
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
        breakpoint_folder_path, state_file = find_breakpoint_path(breakpoint_folder_path)
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
                                        f"{ErrorNumbers.FB413.value}: Breakpoint file was generated with version %s "
                                        f"which is incompatible with the current version %s.")

        # retrieve breakpoint training data
        bkpt_fds = saved_state.get('training_data')
        bkpt_fds = FederatedDataSet(bkpt_fds)
        # retrieve breakpoint sampling strategy
        bkpt_sampling_strategy_args = saved_state.get("node_selection_strategy")
        bkpt_sampling_strategy = cls._create_object(bkpt_sampling_strategy_args, data=bkpt_fds)
        # retrieve breakpoint researcher optimizer
        bkpt_optim = cls._load_optimizer(saved_state.get("agg_optimizer"))

        # initializing experiment
        loaded_exp = cls(tags=saved_state.get('tags'),
                         nodes=None,  # list of previous nodes is contained in training_data
                         training_data=bkpt_fds,
                         agg_optimizer=bkpt_optim,
                         node_selection_strategy=bkpt_sampling_strategy,
                         round_limit=saved_state.get("round_limit"),
                         training_plan_class=saved_state.get("training_plan_class"),
                         training_plan_path=saved_state.get("training_plan_path"),
                         model_args=saved_state.get("model_args"),
                         training_args=saved_state.get("training_args"),
                         save_breakpoints=True,
                         experimentation_folder=saved_state.get('experimentation_folder'),
                         secagg=SecureAggregation.load_state(saved_state.get('secagg')))

        # nota: we are initializing experiment with no aggregator: hence, by default,
        # `loaded_exp` will be loaded with FedAverage.

        # changing `Experiment` attributes
        loaded_exp._set_round_current(saved_state.get('round_current'))

        # TODO: checks when loading parameters
        training_plan = loaded_exp.training_plan()
        if training_plan is None:
            msg = ErrorNumbers.FB413.value + ' - load failed, ' + \
                'breakpoint file seems corrupted, `training_plan` is None'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)
        else:
            loaded_exp._aggregated_params = loaded_exp._load_aggregated_params(
                saved_state.get('aggregated_params')
            )

        # retrieve and change federator
        bkpt_aggregator_args = saved_state.get("aggregator")

        bkpt_aggregator = loaded_exp._create_object(bkpt_aggregator_args, training_plan=training_plan)
        loaded_exp.set_aggregator(bkpt_aggregator)

        # changing `Job` attributes
        loaded_exp._job.load_state(saved_state.get('job'))

        logger.info(f"Experimentation reload from {breakpoint_folder_path} successful!")
        return loaded_exp

    @staticmethod
    @exp_exceptions
    def _save_aggregated_params(aggregated_params_init: dict, breakpoint_path: str) -> Dict[int, dict]:
        """Extract and format fields from aggregated_params that need to be saved in breakpoint.

        Creates link to the params file from the `breakpoint_path` and use them to reference the params files.

        Args:
            aggregated_params_init (dict): ???
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
    def _load_aggregated_params(aggregated_params: Dict[str, dict]) -> Dict[int, Dict[str, Any]]:
        """Reconstruct experiment's aggregated params.

        Aggregated parameters structure from a breakpoint. It is identical to a classical `_aggregated_params`.

        Args:
            aggregated_params: JSON formatted aggregated_params extract from a breakpoint

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

        # JSON converted all keys from int to string, need to revert
        try:
            for key in list(aggregated_params):
                aggregated_params[int(key)] = aggregated_params.pop(key)
        except (TypeError, ValueError):
            msg = ErrorNumbers.FB413.value + ' - load failed. ' + \
                f'Bad key {str(key)} in aggregated params, should be convertible to int'
            logger.critical(msg)
            raise FedbiomedExperimentError(msg)

        for aggreg in aggregated_params.values():
            aggreg['params'] = Serializer.load(aggreg['params_path'])

        return aggregated_params

    @exp_exceptions
    def _save_optimizer(self, breakpoint_path: str) -> Optional[str]:
        """Save the researcher-side Optimizer attached to this Experiment.

        Args:
            breakpoint_path: Path to the breakpoint folder.

        Returns:
            Path to the optimizer's save file, or None if no Optimizer is used.
        """
        # Case when no researcher optimizer is used.
        if self._agg_optimizer is None:
            return None
        # Case when an Optimizer is used: save its state and return the path.
        state = self._agg_optimizer.get_state()
        path = os.path.join(breakpoint_path, f"optimizer_{uuid.uuid4()}.mpk")
        Serializer.dump(state, path)
        return path

    @staticmethod
    @exp_exceptions
    def _load_optimizer(state_path: Optional[str]) -> Optional[Optimizer]:
        """Load an optional researcher-side Optimizer from a breakpoint path.

        Args:
            state_path: Optional path to a breakpoint-attached Optimizer state
                dump file.

        Returns:
            Optimizer instantiated from the provided state file, or None.
        """
        # Case when no researcher optimizer is used.
        if state_path is None:
            return None
        # Case when an Optimizer is used: de-serialize its state and load it.
        state = Serializer.load(state_path)
        return Optimizer.load_state(state)

    @staticmethod
    @exp_exceptions
    def _create_object(args: Dict[str, Any], training_plan: Optional['FederatedPlan'] = None,
                       **object_kwargs: dict) -> Any:
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
        if "training_plan" in inspect.signature(object_instance.load_state).parameters:
            object_instance.load_state(args, training_plan=training_plan)
        else:
            object_instance.load_state(args)
        # note: exceptions for `load_state` should be handled in training plan

        return object_instance

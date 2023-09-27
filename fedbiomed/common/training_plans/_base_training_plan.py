# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Base class defining the shared API of all training plans."""
import random
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from fedbiomed.common import utils
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import (
    FedbiomedError, FedbiomedModelError, FedbiomedTrainingPlanError
)
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import Metrics, MetricTypes
from fedbiomed.common.models import Model
from fedbiomed.common.optimizers.generic_optimizers import BaseOptimizer
from fedbiomed.common.training_plans._federated_data_plan import FederatedDataPlan


class BaseTrainingPlan(FederatedDataPlan):
    """Base class for training plan

    All concrete, framework- and/or model-specific training plans
    should inherit from this class, and implement:
        * the `post_init` method:
            to process model and training hyper-parameters
        * the `training_routine` method:
            to train the model for one round
        * the `predict` method:
            to compute predictions over a given batch
        * (opt.) the `testing_step` method:
            to override the evaluation behavior and compute
            a batch-wise (set of) metric(s)

    Attributes:
        dataset_path: The path that indicates where dataset has been stored
        pre_processes: Preprocess functions that will be applied to the
            training data at the beginning of the training routine.
        training_data_loader: Data loader used in the training routine.
        testing_data_loader: Data loader used in the validation routine.
    """

    _model: Model
    _optimizer: BaseOptimizer

    # static attributes
    dataset_class = None

    def __init__(self) -> None:
        """Construct the base training plan."""
        # Arguments provided by the researcher; they will be populated by post_init
        super().__init__()
        self._model_args: Dict[str, Any] = None
        self._aggregator_args: Dict[str, Any] = None
        self._optimizer_args: Dict[str, Any] = None
        self._training_args: Dict[str, Any] = None

    @abstractmethod
    def model(self):
        """Gets model instance of the training plan"""

    def optimizer(self) -> Optional[BaseOptimizer]:
        """Get the BaseOptimizer wrapped by this training plan.

        Returns:
            BaseOptimizer wrapped by this training plan, or None if
            it has not been initialized yet.
        """
        # FUTURE: return `self._optimizer.optimizer` instead?
        # Currently, the legacy Scaffold implem. needs the BaseOptimizer,
        # but IMHO it really should remain a private backend component.
        return self._optimizer

    @abstractmethod
    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: Dict[str, Any],
            aggregator_args: Optional[Dict[str, Any]] = None,
        ) -> None:
        """Process model, training and optimizer arguments.

        Args:
            model_args: Arguments defined to instantiate the wrapped model.
            training_args: Arguments that are used in training routines
                such as epoch, dry_run etc.
                Please see [`TrainingArgs`][fedbiomed.common.training_args.TrainingArgs]
            aggregator_args: Arguments managed by and shared with the
                researcher-side aggregator.
        """
        super().post_init(model_args,training_args, aggregator_args)
        # Store various arguments provided by the researcher
        self._model_args = model_args
        self._aggregator_args = aggregator_args or {}
        self._optimizer_args = training_args.optimizer_arguments() or {}
        self._loader_args = training_args.loader_arguments() or {}
        self._training_args = training_args.pure_training_arguments()

        # Set random seed: the seed can be either None or an int provided by the researcher.
        # when it is None, both random.seed and np.random.seed rely on the OS to generate a random seed.
        rseed = training_args['random_seed']
        random.seed(rseed)
        np.random.seed(rseed)

    def get_model_params(self, only_trainable: bool = False,) -> Dict[str, Any]:
        """Return a copy of the model's trainable weights.

        The type of data structure used to store weights depends on the actual
        framework of the wrapped model.

        Args:
            only_trainable: Whether to ignore non-trainable model parameters
                from outputs (e.g. frozen neural network layers' parameters),
                or include all model parameters (the default).

        Returns:
            Model weights, as a dictionary mapping parameters' names to their value.
        """
        return self._model.get_weights(only_trainable=only_trainable)

    def set_model_params(self, params: Dict[str, Any]) -> None:
        """Assign new values to the model's trainable parameters.

        The type of data structure used to store weights depends on the actual
        framework of the wrapped model.

        Args:
            params: model weights, as a dictionary mapping parameters' names
                to their value.
        """
        return self._model.set_weights(params)

    def set_aggregator_args(self, aggregator_args: Dict[str, Any]):
        raise FedbiomedTrainingPlanError("method not implemented and needed")

    @abstractmethod
    def init_optimizer(self) -> Any:
        """Method for declaring optimizer by default

        Returns:
            either framework specific optimizer (or None) or
            FedBiomed [`Optimizers`][`fedbiomed.common.optimizers.Optimizer`]
        """

    def optimizer_args(self) -> Dict:
        """Retrieves optimizer arguments (to be overridden
        by children classes)

        Returns:
            Empty dictionary: (to be overridden in children classes)
        """
        return {}

    @staticmethod
    def _create_metric_result_dict(
            metric: Union[Dict[str, float], List[float], float, np.ndarray, torch.Tensor, List[torch.Tensor]],
            metric_name: str = 'Custom'
        ) -> Dict[str, float]:
        """Create a metrics result dictionary, feedable to a HistoryMonitor.

        Args:
            metric: Array-like or scalar
                metric value(s), or dictionary wrapping such values.
            metric_name: Name of the metric.
                If `metric` is a list, metric names will be set to
                (`metric_name_1`, ..., `metric_name_n`).
                If `metric` is a dict, `metric_name` will be ignored
                and the dict's keys will be used instead.

        Returns:
            Dictionary mapping <metric_name>:<metric values>, where
                <metric values> are floats taken or converted from `metric`.

        Raises:
            FedbiomedTrainingPlanError: if `metric` input is of unproper type.
        """
        # If `metric` is an array-like structure, convert it.
        if isinstance(metric, torch.Tensor):
            metric = metric.numpy()
        if isinstance(metric, np.ndarray):
            metric = list(metric) if metric.shape else float(metric)
        # If `metric` is a single value, return a {name: value} dict.
        if isinstance(metric, (int, float, np.integer, np.floating)) and not isinstance(metric, bool):
            return {metric_name: float(metric)}
        # If `metric` is a collection.
        if isinstance(metric, (dict, list)):
            if isinstance(metric, list):
                metric_names = [
                    f"{metric_name}_{i + 1}" for i in range(len(metric))
                ]
            elif isinstance(metric, dict):
                metric_names = list(metric)
            try:
                values = utils.convert_iterator_to_list_of_python_floats(metric)
            except FedbiomedError as exc:
                msg = (
                    f"{ErrorNumbers.FB605.value}: error when converting "
                    f"metric values to float - {exc}"
                )
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg) from exc
            return dict(zip(metric_names, values))
        # Raise if `metric` is of unproper input type.
        msg = (
            f"{ErrorNumbers.FB605.value}: metric value should be one of type "
            "int, float, numpy scalar, numpy.ndarray, torch.Tensor, or list "
            f"or dict wrapping such values; but received {type(metric)}"
        )
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    @abstractmethod
    def training_routine(
            self,
            history_monitor: Optional['HistoryMonitor'] = None,
            node_args: Optional[Dict[str, Any]] = None
        ) -> None:
        """Training routine, to be called once per round.

        Args:
            history_monitor: optional HistoryMonitor
                instance, recording training metadata.
            node_args: Command line arguments for node.
                These arguments can specify GPU use; however, this is not
                supported for scikit-learn models and thus will be ignored.
        """
        return None

    def testing_routine(
            self,
            metric: Optional[MetricTypes],
            metric_args: Dict[str, Any],
            history_monitor: Optional['HistoryMonitor'],
            before_train: bool
        ) -> None:
        """Evaluation routine, to be called once per round.

        !!! info "Note"
            If the training plan implements a `testing_step` method
            (the signature of which is func(data, target) -> metrics)
            then it will be used rather than the input metric.

        Args:
            metric: The metric used for validation.
                If None, use MetricTypes.ACCURACY.
            history_monitor: HistoryMonitor instance,
                used to record computed metrics and communicate them to
                the researcher (server).
            before_train: Whether the evaluation is being performed
                before local training occurs, of afterwards. This is merely
                reported back through `history_monitor`.
        """
        # TODO: Add preprocess option for testing_data_loader.
        if self.testing_data_loader is None:
            msg = f"{ErrorNumbers.FB605.value}: no validation dataset was set."
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        n_batches = len(self.testing_data_loader)
        n_samples = len(self.testing_data_loader.dataset)
        # Set up a batch-wise metrics-computation function.
        # Either use an optionally-implemented custom training routine.
        if hasattr(self, "testing_step"):
            evaluate = getattr(self, "testing_step")
            metric_name = "Custom"
        # Or use the provided `metric` (or its default value).
        else:
            if metric is None:
                metric = MetricTypes.ACCURACY
            metric_controller = Metrics()
            def evaluate(data, target):
                nonlocal metric, metric_args, metric_controller
                output = self._model.predict(data)
                if isinstance(target, torch.Tensor):
                    target = target.numpy()
                return metric_controller.evaluate(
                    target, output, metric=metric, **metric_args
                )
            metric_name = metric.name
        # Iterate over the validation dataset and run the defined routine.
        num_samples_observed_till_now: int = 0
        for idx, (data, target) in enumerate(self.testing_data_loader, 1):
            num_samples_observed_till_now += self._infer_batch_size(data)
            # Run the evaluation step; catch and raise exceptions.
            try:
                m_value = evaluate(data, target)
            except Exception as exc:
                msg = (
                    f"{ErrorNumbers.FB605.value}: An error occurred "
                    f"while computing the {metric_name} metric: {exc}"
                )
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg) from exc
            # Log the computed value.
            logger.debug(
                f"Validation: Batch {idx}/{n_batches} "
                f"| Samples {num_samples_observed_till_now}/{n_samples} "
                f"| Metric[{metric_name}]: {m_value}"
            )
            # Further parse, and report it (provided a monitor is set).
            if history_monitor is not None:
                m_dict = self._create_metric_result_dict(m_value, metric_name)
                history_monitor.add_scalar(
                    metric=m_dict,
                    iteration=idx,
                    epoch=None,
                    test=True,
                    test_on_local_updates=(not before_train),
                    test_on_global_updates=before_train,
                    total_samples=n_samples,
                    batch_samples=num_samples_observed_till_now,
                    num_batches=n_batches
                )

    def after_training_params(
        self,
        flatten: bool = False,
    ) -> Union[Dict[str, Any], List[float]]:
        """Return the wrapped model's parameters for aggregation.

        This method returns a dict containing parameters that need to be
        reported back and aggregated in a federated learning setting.

        It may also implement post-processing steps to make these parameters
        suitable for sharing with the researcher after training - hence its
        being used over `get_model_params` at the end of training rounds.

        Returns:
            The trained parameters to aggregate.
        """
        if flatten:
            return self._model.flatten()
        return self.get_model_params()

    def export_model(self, filename: str) -> None:
        """Export the wrapped model to a dump file.

        Args:
            filename: path to the file where the model will be saved.

        !!! info "Notes":
            This method is designed to save the model to a local dump
            file for easy re-use by the same user, possibly outside of
            Fed-BioMed. It is not designed to produce trustworthy data
            dumps and is not used to exchange models and their weights
            as part of the federated learning process.

            To save the model parameters for sharing as part of the FL process,
            use the `after_training_params` method (or `get_model_params` one
            outside of a training context) and export results using
            [`Serializer`][fedbiomed.common.serializer.Serializer].
        """
        self._model.export(filename)

    def import_model(self, filename: str) -> None:
        """Import and replace the wrapped model from a dump file.

        Args:
            filename: path to the file where the model has been exported.

        !!! info "Notes":
            This method is designed to load the model from a local dump
            file, that might not be in a trustworthy format. It should
            therefore only be used to re-load data exported locally and
            not received from someone else, including other FL peers.

            To load model parameters shared as part of the FL process, use the
            [`Serializer`][fedbiomed.common.serializer.Serializer] to read the
            network-exchanged file, and the `set_model_params` method to assign
            the loaded values into the wrapped model.
        """
        try:
            self._model.reload(filename)
        except FedbiomedModelError as exc:
            msg = (
                f"{ErrorNumbers.FB304.value}: failed to import a model from "
                f"a dump file: {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg) from exc

    def model_args(self) -> Dict[str, Any]:
        """Retrieve model arguments.

        Returns:
            Model arguments
        """
        return self._model_args

    def training_args(self) -> Dict[str, Any]:
        """Retrieve training arguments

        Returns:
            Training arguments
        """
        return self._training_args

    def optimizer_args(self) -> Dict[str, Any]:
        """Retrieves optimizer arguments

        Returns:
            Optimizer arguments
        """
        return self._optimizer_args

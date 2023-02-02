# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Base class defining the shared API of all training plans."""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from collections import OrderedDict

from torch.utils.data import DataLoader

from fedbiomed.common import utils
from fedbiomed.common.constants import ErrorNumbers, ProcessTypes
from fedbiomed.common.data import NPDataLoader
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedTrainingPlanError, FedbiomedUserInputError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import Metrics, MetricTypes
from fedbiomed.common.training_plans._training_iterations import MiniBatchTrainingIterationsAccountant
from fedbiomed.common.utils import get_class_source
from fedbiomed.common.utils import get_method_spec


class BaseTrainingPlan(metaclass=ABCMeta):
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

    def __init__(self) -> None:
        """Construct the base training plan."""
        self._dependencies: List[str] = []
        self.dataset_path: Union[str, None] = None
        self.pre_processes: Dict[
            str, Dict[ProcessTypes, Union[str, Callable[..., Any]]]
        ] = OrderedDict()
        self.training_data_loader: Union[DataLoader, NPDataLoader, None] = None
        self.testing_data_loader: Union[DataLoader, NPDataLoader, None] = None

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
        return None

    def add_dependency(self, dep: List[str]) -> None:
        """Add new dependencies to the TrainingPlan.

        These dependencies are used while creating a python module.

        Args:
            dep: Dependencies to add. Dependencies should be indicated as
                import statement strings, e.g. `"from torch import nn"`.
        """
        for val in dep:
            if val not in self._dependencies:
                self._dependencies.append(val)

    def set_dataset_path(self, dataset_path: str) -> None:
        """Dataset path setter for TrainingPlan

        Args:
            dataset_path: The path where data is saved on the node.
                This method is called by the node that executes the training.
        """
        self.dataset_path = dataset_path
        logger.debug(f"Dataset path has been set as {self.dataset_path}")

    def set_data_loaders(
            self,
            train_data_loader: Union[DataLoader, NPDataLoader, None],
            test_data_loader: Union[DataLoader, NPDataLoader, None]
        ) -> None:
        """Sets data loaders

        Args:
            train_data_loader: Data loader for training routine/loop
            test_data_loader: Data loader for validation routine
        """
        self.training_data_loader = train_data_loader
        self.testing_data_loader = test_data_loader

    def init_dependencies(self) -> List:
        """Default method where dependencies are returned

        Returns:
            Empty list as default
        """
        return []

    def _configure_dependencies(self) -> None:
        """ Configures dependencies """
        init_dep_spec = get_method_spec(self.init_dependencies)
        if len(init_dep_spec.keys()) > 0:
            raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605}: `init_dependencies` should not take any argument. "
                                             f"Unexpected arguments: {list(init_dep_spec.keys())}")

        dependencies: Union[Tuple, List] = self.init_dependencies()
        if not isinstance(dependencies, (list, tuple)):
            raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605}: Expected dependencies are l"
                                             f"ist or tuple, but got {type(dependencies)}")
        self.add_dependency(dependencies)

    def save_code(self, filepath: str) -> None:
        """Saves the class source/codes of the training plan class that is created byuser.

        Args:
            filepath: path to the destination file

        Raises:
            FedbiomedTrainingPlanError: raised when source of the model class cannot be assessed
            FedbiomedTrainingPlanError: raised when model file cannot be created/opened/edited
        """
        try:
            class_source = get_class_source(self.__class__)
        except FedbiomedError as e:
            msg = ErrorNumbers.FB605.value + \
                  " : error while getting source of the model class - " + \
                  str(e)
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        # Preparing content of the module
        content = ""
        for s in self._dependencies:
            content += s + "\n"

        content += "\n"
        content += class_source

        try:
            # should we write it in binary (for the sake of space optimization)?
            with open(filepath, "w") as file:
                file.write(content)
            logger.debug("Model file has been saved: " + filepath)
        except PermissionError:
            _msg = ErrorNumbers.FB605.value + f" : Unable to read {filepath} due to unsatisfactory privileges" + \
                   ", can't write the model content into it"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg)
        except MemoryError:
            _msg = ErrorNumbers.FB605.value + f" : Can't write model file on {filepath}: out of memory!"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg)
        except OSError:
            _msg = ErrorNumbers.FB605.value + f" : Can't open file {filepath} to write model content"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg)

    def training_data(self):
        """All subclasses must provide a training_data routine the purpose of this actual code is to detect
        that it has been provided

        Raises:
            FedbiomedTrainingPlanError: if called and not inherited
        """
        msg = ErrorNumbers.FB303.value + ": training_data must be implemented"
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def get_model_params(self) -> Union[OrderedDict, Dict]:
        """
        Retrieves parameters from a model defined in a training plan.
        Output format depends on the nature of the training plan (OrderedDict for
        a PyTorch training plan, np.ndarray for a sklearn training plan)

        Returns:
            Union[OrderedDict, np.ndarray]: model parameters. Object type depends on
            the nature of the TrainingPlan
        """
        msg = ErrorNumbers.FB303.value + ": get_model_parans method must be implemented in the TrainingPlan"
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def get_learning_rate(self) -> List[float]:
        raise FedbiomedTrainingPlanError("method not implemented")

    def set_aggregator_args(self, aggregator_args: Dict[str, Any]):
        raise FedbiomedTrainingPlanError("method not implemented and needed")

    def optimizer_args(self) -> Dict:
        """Retrieves optimizer arguments (to be overridden
        by children classes)

        Returns:
            Empty dictionary: (to be overridden in children classes)
        """
        return {}

    def add_preprocess(
            self,
            method: Callable,
            process_type: ProcessTypes
        ) -> None:
        """Register a pre-processing method to be executed on training data.

        Args:
            method: Pre-processing method to be run before training.
            process_type: Type of pre-processing that will be run.
                The expected signature of `method` and the arguments
                passed to it depend on this parameter.
        """
        if not callable(method):
            msg = (
                f"{ErrorNumbers.FB605.value}: error while adding "
                "preprocess, `method` should be callable."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        if not isinstance(process_type, ProcessTypes):
            msg = (
                f"{ErrorNumbers.FB605.value}: error while adding "
                "preprocess, `process_type` should be an instance "
                "of `fedbiomed.common.constants.ProcessTypes`."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # NOTE: this may be revised into a list rather than OrderedDict
        self.pre_processes[method.__name__] = {
            'method': method,
            'process_type': process_type
        }

    def _preprocess(self) -> None:
        """Executes registered data pre-processors."""
        for name, process in self.pre_processes.items():
            method = process['method']
            process_type = process['process_type']
            if process_type == ProcessTypes.DATA_LOADER:
                self._process_data_loader(method=method)
            else:
                logger.error(
                    f"Process type `{process_type}` is not implemented."
                    f"Preprocessor '{name}' will therefore be ignored."
                )

    def _process_data_loader(
            self,
            method: Callable[..., Any]
        ) -> None:
        """Handle a data-loader pre-processing action.

        Args:
            method: Process method that is to be executed.

        Raises:
            FedbiomedTrainingPlanError: If one of the following happens:
                - the method does not have 1 positional argument (dataloader)
                - running the method fails
                - the method does not return a dataloader of the same type as
                its input
        """
        # Check that the preprocessing method has a proper signature.
        argspec = utils.get_method_spec(method)
        if len(argspec) != 1:
            msg = (
                f"{ErrorNumbers.FB605.value}: preprocess method of type "
                "`PreprocessType.DATA_LOADER` sould expect one argument: "
                "the data loader wrapping the training dataset."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # Try running the preprocessor.
        try:
            data_loader = method(self.training_data_loader)
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB605.value}: error while running "
                f"preprocess method `{method.__name__}` -> {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        logger.debug(
            f"The process `{method.__name__}` has been successfully executed."
        )
        # Verify that the output is of proper type and assign it.
        if isinstance(data_loader, type(self.training_data_loader)):
            self.training_data_loader = data_loader
            logger.debug(
                "Data loader for training routine has been updated "
                f"by the process `{method.__name__}`."
            )
        else:
            msg = (
                f"{ErrorNumbers.FB605.value}: the return type of the "
                f"`{method.__name__}` preprocess method was expected "
                f"to be {type(self.training_data_loader)}, but was "
                f"{type(data_loader)}."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

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
                raise FedbiomedTrainingPlanError(msg)
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
                output = self.predict(data)
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
                raise FedbiomedTrainingPlanError(msg)
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

    @abstractmethod
    def predict(
            self,
            data: Any,
        ) -> np.ndarray:
        """Return model predictions for a given batch of input features.

        This method is called as part of `testing_routine`, to compute
        predictions based on which evaluation metrics are computed. It
        will however be skipped if a `testing_step` method is attached
        to the training plan, than wraps together a custom routine to
        compute an output metric directly from a (data, target) batch.

        Args:
            data: Array-like (or tensor) structure containing batched
                input features.
        Returns:
            Output predictions, converted to a numpy array (as per the
                `fedbiomed.common.metrics.Metrics` specs).
        """
        return NotImplemented

    @staticmethod
    def _infer_batch_size(data: Union[dict, list, tuple, 'torch.Tensor', 'np.ndarray']) -> int:
        """Utility function to guess batch size from data.

        This function is a temporary fix needed to handle the case where
        Opacus changes the batch_size dynamically, without communicating
        it in any way.

        This will be improved by issue #422.

        Returns:
            the batch size for the input data
        """
        if isinstance(data, dict):
            # case `data` is a dict (eg {'modality1': data1, 'modality2': data2}):
            # compute length of the first modality
            return BaseTrainingPlan._infer_batch_size(next(iter(data.values())))
        elif isinstance(data, (list, tuple)):
            return BaseTrainingPlan._infer_batch_size(data[0])
        else:
            # case `data` is a torch.Tensor or a np.ndarray
            batch_size = len(data)
            return batch_size







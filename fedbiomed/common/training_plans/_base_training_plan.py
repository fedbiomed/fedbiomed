# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Base class defining the shared API of all training plans."""
import random
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from fedbiomed.common import utils
from fedbiomed.common.constants import ErrorNumbers, ProcessTypes
from fedbiomed.common.data import NPDataLoader
from fedbiomed.common.exceptions import (
    FedbiomedError, FedbiomedModelError, FedbiomedTrainingPlanError
)
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import Metrics, MetricTypes
from fedbiomed.common.models import Model
from fedbiomed.common.optimizers.generic_optimizers import BaseOptimizer
from fedbiomed.common.utils import get_class_source
from fedbiomed.common.utils import get_method_spec
from fedbiomed.common.training_plans._training_iterations import MiniBatchTrainingIterationsAccountant


class PreProcessDict(TypedDict):
    """Dict structure to specify a pre-processing transform."""

    method: Callable[..., Any]
    process_type: ProcessTypes


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

    _model: Optional[Model]
    _optimizer: Optional[BaseOptimizer]

    def __init__(self) -> None:
        """Construct the base training plan."""
        self._dependencies: List[str] = []
        self.dataset_path: Union[str, None] = None
        self.pre_processes: Dict[str, PreProcessDict] = OrderedDict()
        self.training_data_loader: Union[DataLoader, NPDataLoader, None] = None
        self.testing_data_loader: Union[DataLoader, NPDataLoader, None] = None

        # Arguments provided by the researcher; they will be populated by post_init
        self._model_args: Dict[str, Any] = None
        self._aggregator_args: Dict[str, Any] = None
        self._optimizer_args: Dict[str, Any] = None
        self._loader_args: Dict[str, Any] = None
        self._training_args: Dict[str, Any] = None

        self._error_msg_import_model: str = f"{ErrorNumbers.FB605.value}: Training Plan's Model is not initialized.\n" +\
                                            "To %s a model, you should do it through `fedbiomed.researcher.federated_workflows.Experiment`'s interface" +\
                                            " and not directly from Training Plan"

    @abstractmethod
    def model(self):
        """Gets model instance of the training plan"""

    # FIXME: re-implement Model as a subclass of Torch.nn.module, SGDClassifier, etc.
    # to avoid having a distinct getter for the class, see #1049

    def get_model_wrapper_class(self) -> Optional[Model]:
        """Gets training plan's model wrapper class.

        Returns:
            the wrapper class for the model, or None
            if model is not instantiated.
        """
        return self._model

    @property
    def dependencies(self):
        return self._dependencies

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

    def set_optimizer(self, optimizer: BaseOptimizer):
        self._optimizer = optimizer

    @abstractmethod
    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: Dict[str, Any],
            aggregator_args: Optional[Dict[str, Any]] = None,
            initialize_optimizer: bool = True
        ) -> None:
        """Process model, training and optimizer arguments.

        Args:
            model_args: Arguments defined to instantiate the wrapped model.
            training_args: Arguments that are used in training routines
                such as epoch, dry_run etc.
                Please see [`TrainingArgs`][fedbiomed.common.training_args.TrainingArgs]
            aggregator_args: Arguments managed by and shared with the
                researcher-side aggregator.
            initialize_optimizer: whether to initialize the optimizer or not. Defaults
                to True.
        """

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

    def _add_dependency(self, dep: List[str]) -> None:
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

    def init_dependencies(self) -> List[str]:
        """Default method where dependencies are returned

        Returns:
            Empty list as default
        """
        return []

    def _configure_dependencies(self) -> None:
        """ Configures dependencies """
        init_dep_spec = get_method_spec(self.init_dependencies)
        if len(init_dep_spec.keys()) > 0:
            raise FedbiomedTrainingPlanError(
                f"{ErrorNumbers.FB605}: `init_dependencies` should not take any argument. "
                f"Unexpected arguments: {list(init_dep_spec.keys())}"
            )
        dependencies = self.init_dependencies()
        if not isinstance(dependencies, (list, tuple)):
            raise FedbiomedTrainingPlanError(
                f"{ErrorNumbers.FB605}: Expected dependencies are a list or "
                "tuple of str, but got {type(dependencies)}"
            )
        self._add_dependency(dependencies)

    def source(self) -> str:

        try:
            class_source = get_class_source(self.__class__)
        except FedbiomedError as exc:
            msg = f"{ErrorNumbers.FB605.value}: error while getting source of the model class: {exc}"
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg) from exc

        # Preparing content of the module
        content = "\n".join(self._dependencies)
        content += "\n"
        content += class_source

        return content

    def save_code(self, filepath: str, from_code: Union[str, None] = None) -> None:
        """Saves the class source/codes of the training plan class that is created byuser.

        Args:
            filepath: path to the destination file

        Raises:
            FedbiomedTrainingPlanError: raised when source of the model class cannot be assessed
            FedbiomedTrainingPlanError: raised when model file cannot be created/opened/edited
        """
        if from_code is None:
            content = self.source()
        else:
            if not isinstance(from_code, str):
                raise FedbiomedTrainingPlanError(f"{ErrorNumbers.FB605}: Expected type str for `from_code`, "
                                                 "got: {type(from_code)}")
            content = from_code

        try:
            # should we write it in binary (for the sake of space optimization)?
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(content)
            logger.debug("Model file has been saved: " + filepath)
        except PermissionError as exc:
            _msg = ErrorNumbers.FB605.value + f" : Unable to read {filepath} due to unsatisfactory privileges" + \
                   ", can't write the model content into it"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg) from exc
        except MemoryError as exc:
            _msg = ErrorNumbers.FB605.value + f" : Can't write model file on {filepath}: out of memory!"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg) from exc
        except OSError as exc:
            _msg = ErrorNumbers.FB605.value + f" : Can't open file {filepath} to write model content"
            logger.critical(_msg)
            raise FedbiomedTrainingPlanError(_msg) from exc

    def training_data(self):
        """All subclasses must provide a training_data routine the purpose of this actual code is to detect
        that it has been provided

        Raises:
            FedbiomedTrainingPlanError: if called and not inherited
        """
        msg = f"{ErrorNumbers.FB303.value}: training_data must be implemented"
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def get_model_params(self,
                         only_trainable: bool = False,
                         exclude_buffers: bool = True) -> Dict[str, Any]:
        """Return a copy of the model's trainable weights.

        The type of data structure used to store weights depends on the actual
        framework of the wrapped model.

        Args:
            only_trainable: Whether to ignore non-trainable model parameters
                from outputs (e.g. frozen neural network layers' parameters),
                or include all model parameters (the default).
            exclude_buffers: Whether to ignore buffers (the default), or
                include them.

        Returns:
            Model weights, as a dictionary mapping parameters' names to their value.
        """
        return self._model.get_weights(only_trainable=only_trainable, exclude_buffers=exclude_buffers)

    def set_model_params(self, params: Dict[str, Any]) -> None:
        """Assign new values to the model's trainable parameters.

        The type of data structure used to store weights depends on the actual
        framework of the wrapped model.

        Args:
            params: model weights, as a dictionary mapping parameters' names
                to their value.
        """
        self._model.set_weights(params)

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
            raise FedbiomedTrainingPlanError(msg) from exc
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

        # If `metric` is a single value, including [val], return a {name: value} dict.
        if isinstance(metric, (int, float, np.integer, np.floating)) and not \
                isinstance(metric, bool):
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
                raise FedbiomedTrainingPlanError(
                    f"{ErrorNumbers.FB605.value}: error when converting "
                    f"metric values to float - {exc}") from exc
            return dict(zip(metric_names, values))

        raise FedbiomedTrainingPlanError(
            f"{ErrorNumbers.FB605.value}: metric value should be one of type "
            "int, float, numpy scalar, numpy.ndarray, torch.Tensor, or list "
            f"or dict wrapping such values; but received {type(metric)}")

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
            before_train: bool,
        ) -> None:
        """Evaluation routine, to be called once per round.

        !!! info "Note"
            If the training plan implements a `testing_step` method
            (the signature of which is func(data, target) -> metrics)
            then it will be used rather than the input metric.

        Args:
            metric: The metric used for validation.
                If None, use MetricTypes.ACCURACY.
            metric_args: dicitonary containing additinal arguments for setting up metric,
                that maps <argument_name; argument_value> ad that will be passed to the
                metric function as positinal arguments.
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

        n_samples = len(self.testing_data_loader.dataset)
        n_batches = max(len(self.testing_data_loader) , 1)

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
            # Reporting

            if idx % self.training_args()['log_interval'] == 0 or idx == 1 or idx == n_batches:
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
        exclude_buffers = not self._training_args['share_persistent_buffers']
        if flatten:
            return self._model.flatten(exclude_buffers=exclude_buffers)
        return self.get_model_params(exclude_buffers=exclude_buffers)

    def export_model(self, filename: str) -> None:
        """Export the wrapped model to a dump file.

        Args:
            filename: path to the file where the model will be saved.

        Raises:
            FedBiomedTrainingPlanError: raised if model has not be initialized through the
            `post_init` method. If you need to export the model, you must do it through
            [`Experiment`][`fedbiomed.researcher.federated_workflows.Experiment`]'s interface.

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
        if self._model is None:
            raise FedbiomedTrainingPlanError(self._error_msg_import_model % "export")
        self._model.export(filename)

    def import_model(self, filename: str) -> None:
        """Import and replace the wrapped model from a dump file.

        Args:
            filename: path to the file where the model has been exported.

        Raises:
            FedBiomedTrainingPlanError: raised if model has not be initialized through the
            `post_init` method. If you need to export the model from the Training Plan, you
            must do it through [`Experiment`][`fedbiomed.researcher.federated_workflows.Experiment`]'s
            interface.

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
        if self._model is None:
            raise FedbiomedTrainingPlanError(self._error_msg_import_model % "import")
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

    def loader_args(self) -> Dict[str, Any]:
        """Retrieve loader arguments

        Returns:
            Loader arguments
        """
        return self._loader_args

    def optimizer_args(self) -> Dict[str, Any]:
        """Retrieves optimizer arguments

        Returns:
            Optimizer arguments
        """
        return self._optimizer_args



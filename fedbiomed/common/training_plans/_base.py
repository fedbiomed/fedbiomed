# coding: utf-8

"""Base class defining the shared API of all training plans."""

import functools
import importlib
import json
import os
import sys
import tempfile
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import declearn
import numpy as np
import python_minifier
import torch

from fedbiomed.common import utils
from fedbiomed.common.constants import ErrorNumbers, ProcessTypes
from fedbiomed.common.data import DataLoaderTypes, DataManager, TypeDataLoader
from fedbiomed.common.exceptions import (
    FedbiomedError, FedbiomedTrainingPlanError
)
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import Metrics, MetricTypes
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.utils import get_class_source


IMPORT_IDX = 0  # global counter to use unique names for generated .py files


class TrainingPlan(metaclass=ABCMeta):
    """Base class for training plans.

    All concrete, framework- and/or model-specific training plans
    should inherit from this class, and implement:
        * the `_model_cls` class attribute:
            to define the type of declearn Model to use
        * the `_data_type` class attribute:
            to define the type of data loader expected
        * the `predict` method:
            to compute predictions over a given batch
        * the `training_data` method:
            to define how to set up the `fedbiomed.data.DataManager`
            wrapping the training (and, by split, validation) data
        * (opt.) the `testing_step` method:
            to override the evaluation behavior and compute
            a batch-wise (set of) metric(s)

    Attributes:
        model: declearn Model instance wrapping the model being trained.
        optim: declearn Optimizer in charge of node-side optimization.
        pre_processes: Preprocess functions that will be applied to the
            training data at the beginning of the training routine.
        training_data_loader: Data loader used in the training routine.
        testing_data_loader: Data loader used in the validation routine.
    """

    _model_cls: Type[declearn.model.api.Model]
    _data_type: DataLoaderTypes

    def __init__(
            self,
            model: Union[Any, Dict[str, Any]],
            optim: Union[declearn.optimizer.Optimizer, Dict[str, Any]],
            **kwargs: Any
        ) -> None:
        """Construct the base training plan.

        Args:
            model: Base model object to be interfaced through a declearn
                Model (the class of which is set by `self._model_cls`),
                or config dict of the latter declearn Model class.
            optim: declearn.optimizer.Optimizer instance of config dict.
            **kwargs: Any additional keyword parameter to the declearn
                Model class constructor. Unused if `model` is a dict.
        """
        self.model = self._build_model(model, **kwargs)
        self.optim = self._build_optim(optim)
        self._training_args: Optional[TrainingArgs] = None

        self._dependencies: List[str] = []
        self.pre_processes: Dict[
            str, Dict[str, Union[ProcessTypes, Callable[..., Any]]]
        ] = OrderedDict()
        self.training_data_loader: Optional[TypeDataLoader] = None
        self.testing_data_loader: Optional[TypeDataLoader] = None

    def _build_model(
            self,
            model: Union[Any, Dict[str, Any]],
            **kwargs: Any,
        ) -> declearn.model.api.Model:
        """Build a class-based declearn Model based on input arguments.

        Args:
            model: Base model object to be interfaced through a declearn
                Model (the class of which is set by `self._model_cls`),
                or config dict of the latter declearn Model class.
            **kwargs: Any additional keyword parameter to the declearn
                Model class constructor. Unused if `model` is a dict.

        Raises:
            FedbiomedTrainingPlanError: If the Model instantiation fails.
        """
        try:
            if isinstance(model, dict):
                model = self._model_cls.from_config(model)
            else:
                model = self._model_cls(model, **kwargs)
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB304.value}: failed to wrap up the provided "
                f"model using {self._model_cls}: {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg) from exc
        # Return the instantiated model.
        return model

    def _build_optim(
            self,
            optim: Union[declearn.optimizer.Optimizer, Dict[str, Any]],
        ) -> declearn.optimizer.Optimizer:
        """Validate or build a declearn Optimizer.

        Args:
            optim: declearn.optimizer.Optimizer instance of config dict.

        Raises:
            FedbiomedTrainingPlanError: In case of type error of if the
                Optimizer instantiation from config fails.
        """
        try:
            if isinstance(optim, dict):
                optim = declearn.optimizer.Optimizer.from_config(optim)
            if not isinstance(optim, declearn.optimizer.Optimizer):
                raise TypeError(
                    "Wrong input type for training plan's 'optim': "
                    f"expected declearn.optimizer.Optimizer, got {type(optim)}"
                )
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB304.value}: failed to wrap up the provided "
                f"optimizer: {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg) from exc
        # Return the instantiated model.
        return optim

    def data_loader_type(self) -> DataLoaderTypes:
        """Getter for the type of DataLoader required by this TrainingPlan."""
        return self._data_type

    # TODO-PAUL: merge this into `__init__`?
    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: TrainingArgs,
        ) -> None:
        """Set arguments for the model, training and the optimizer.

        Args:
            model_args: Arguments defined to instantiate the wrapped model.
            training_args: Arguments that are used in training routines
                such as epoch, dry_run etc.
                Please see [`TrainingArgs`][fedbiomed.common.training_args.TrainingArgs]
        """
        try:
            self.model.initialize(model_args)
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB304.value}: failed to initialize the "
                f"wrapped model using provided `model_args`: {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg) from exc
        self._training_args = training_args

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

    #@abstractmethod
    def training_data(
            self,
            dataset_path: str
        ) -> DataManager:
        """Instantiate and return a DataManager suitable for this plan.

        All subclasses must provide a training_data routine.

        This method is called once per round by each node that performs
        training and evaluation according to this plan.

        Args:
            dataset_path: The path where data is saved on the node.

        Raises:
            FedbiomedTrainingPlanError: if called and not inherited.
        """
        msg = ErrorNumbers.FB303.value + ": training_data must be implemented"
        logger.critical(msg)
        raise FedbiomedTrainingPlanError(msg)

    def set_data_loaders(
            self,
            train_data_loader: TypeDataLoader,
            test_data_loader: TypeDataLoader,
        ) -> None:
        """Data loaders setter for TrainingPlan.

        This method is used to provide with the final data loaders
        that are to be used in the training and testing routines.

        Args:
            train_data_loader: Data loader for training routine/loop.
            test_data_loader: Data loader for validation routine.
        """
        for loader in (train_data_loader, test_data_loader):
            if not isinstance(loader, self._data_type.value):
                msg = (
                    f"{ErrorNumbers.FB304.value}: unproper data loader type:"
                    f" required {self._data_type.value}, got {type(loader)}"
                )
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)
        self.training_data_loader = train_data_loader
        self.testing_data_loader = test_data_loader

    def save_code(self, filepath: str) -> str:
        """Save the training plan's source code and parameters to a JSON file.

        Args:
            filepath: Path to the destination file.

        Raises:
            FedbiomedTrainingPlanError: if the model file cannot be written to.
        """
        # TODO: garbage-collect this (after correcting external calls)
        self.save_to_json(filepath)
        return filepath

    def save_to_json(self, path: str) -> None:
        """Save the training plan's source code and parameters to a JSON file.

        Args:
            filepath: Path to the destination file.

        Raises:
            FedbiomedTrainingPlanError: if the model file cannot be written to.
        """
        # Assemble the class's source code.
        source = "\n".join((
            "\n".join(self._dependencies),  # import statements
            get_class_source(type(self)),  # class definition
        ))
        # Minify it.
        source = python_minifier.minify(
            source,
            remove_annotations=False,
            combine_imports=False,
            remove_pass=False,
            hoist_literals=False,
            remove_object_base=True,
            rename_locals=False,
        )
        # Wrap up the class's source and its parameters into a config dict.
        # NOTE: pre-processes are *not* saved (as in current Fed-BioMed)
        config = {
            "clsname": self.__class__.__name__,
            "source": source,
            "dependencies": self._dependencies,
            "model": self.model.get_config(),
            "optim": self.optim.get_config(),
        }
        # Write out the config dict to a JSON file.
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(config, file, default=declearn.utils.json_pack)
            logger.debug("Model file has been saved: " + path)
        except (OSError, MemoryError, PermissionError) as exc:
            msg = (
                f"{ErrorNumbers.FB605.value}: Unable to write model "
                f"file to {path} due to: {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg) from exc

    @staticmethod
    def load_from_json(path: str) -> "TrainingPlan":
        """Reload a training plan from its JSON dump."""
        global IMPORT_IDX  # REVISE: improve this (global is not great)
        # Import the JSON-serialized config.
        try:
            with open(path, "r", encoding="utf-8") as file:
                config = json.load(
                    file, object_hook=declearn.utils.json_unpack
                )
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB304.value}: failed to load training plan "
                f"config from {path}: {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg) from exc
        # Execute the source code and gather the training plan subclass.
        # TODO: improve this part to enhance security.
        with tempfile.TemporaryDirectory() as folder:
            # Write the source code to a temporary .py file.
            try:
                # Use a unique name (otherwise next import attempt would fail).
                name = f"tplan_{IMPORT_IDX:03d}"
                path = os.path.join(folder, f"{name}.py")
                with open(path, "w", encoding="utf-8") as file:
                    file.write(config["source"])
                sys.path.append(folder)  # undone in next "finally" block
            except Exception as exc:
                msg = (
                    f"{ErrorNumbers.FB304.value}: failed to write the "
                    f"training plan subclass source code to a file: {exc}"
                )
                raise FedbiomedTrainingPlanError(msg) from exc
            # Import the file as a module and gather the class.
            try:
                mod = importlib.import_module(name)
                IMPORT_IDX += 1  # update number of used file names
                cls = getattr(mod, config["clsname"])
            except Exception as exc:
                msg = (
                    f"{ErrorNumbers.FB304.value}: failed to recreate the "
                    f"training plan subclass from its source code: {exc}"
                )
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg) from exc
            # Ensure the temporary folder is removed from `sys.path`.
            finally:
                sys.path.remove(folder)
        # Instantiate the training plan.
        if not (isinstance(cls, type) and issubclass(cls, TrainingPlan)):
            msg = (
                f"{ErrorNumbers.FB304.value}: Reloaded element from config "
                "and source code is not a training plan subclass."
            )
            raise FedbiomedTrainingPlanError(msg)
        return cls(model=config["model"], optim=config["optim"])

    def save_weights(
            self,
            path: str,
        ) -> None:
        """Save the wrapped model's weights to a JSON file.

        Args:
            path: Path to the destination file.
        """
        try:
            weights = self.model.get_weights()
            with open(path, "w", encoding="utf-8") as file:
                json.dump(weights, file, default=declearn.utils.json_pack)
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB304.value}: Failed to save model "
                f"weights to JSON: {exc}"
            )
            logger.error(msg)
            raise FedbiomedTrainingPlanError(msg) from exc

    def load_weights(
            self,
            path: str,
            assign: bool = True,
        ) -> declearn.model.api.NumpyVector:
        """Reload the wrapped model's weights from a JSON file.

        Args:
            path: Path to the source file.
            assign: Whether to assign the reloaded weights to
                the model, or merely load and return them.

        Returns:
            weights: Reloaded weights, formatted as a declearn
                NumpyVector.
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                weights = json.load(
                    file, object_hook=declearn.utils.json_unpack
                )
            if not isinstance(weights, declearn.model.api.NumpyVector):
                raise TypeError(
                    "Reloaded weights should be a declearn NumpyVector"
                )
            if assign:
                self.model.set_weights(weights)
            return weights
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB304.value}: Failed to reload model "
                f"weights from JSON: {exc}"
            )
            logger.error(msg)
            raise FedbiomedTrainingPlanError(msg) from exc

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
            msg = (  # type: ignore  # unreachable warning
                f"{ErrorNumbers.FB605.value}: error while adding "
                "preprocess, `method` should be callable."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        if not isinstance(process_type, ProcessTypes):
            msg = (  # type: ignore  # unreachable warning
                f"{ErrorNumbers.FB605.value}: error while adding "
                "preprocess, `process_type` should be an instance "
                "of `fedbiomed.common.constants.ProcessType`."
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
                self._process_data_loader(method=method)  # type: ignore
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
                f"preprocess method `{method.__name__}`: {exc}"
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

    def training_routine(
            self,
            history_monitor: Optional['HistoryMonitor'] = None,
            node_args: Optional[Dict[str, Any]] = None
        ) -> None:
        """Training routine, to be called once per round.

        !!! info "Note"
            The training behavior may be adjusted by extending or
            overriding the `_training_step` method, which defines
            the step-wise training operation.

        Args:
            history_monitor: optional HistoryMonitor
                instance, recording training metadata.
            node_args: Node-specific command-line arguments.
                These arguments can notably specify GPU use
                for compatible model frameworks.
        """
        # Run training pre-checks.
        if not isinstance(self.training_data_loader, self._data_type.value):
            msg = (
                f"{ErrorNumbers.FB310.value}: BaseTrainingPlan cannot "
                "be trained without a `training_data_loader` of type "
                f"{self._data_type.value}."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        if self._training_args is None:
            msg = (
                f"{ErrorNumbers.FB304.value}: BaseTrainingPlan cannot "
                "be trained without `training_args` having been set."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # Run optional preprocessing operations.
        self._preprocess()
        # Set up loss reporting.
        log_interval = self._training_args.get("log_interval", 10)
        record_loss = self._setup_loss_reporting(history_monitor)
        # Set up effort constraints (max. number of epochs and/or steps).
        epochs = declearn.main.utils.Constraint(
            self._training_args.get("epochs", 1.), name="n_epochs"
        )
        nsteps = declearn.main.utils.Constraint(
            self._training_args.get("num_updates"), name="num_updates"
        )
        if bool(self._training_args.get("dry_run")):
            nsteps.limit = 1
        # Process node arguments.
        self._process_training_node_args(node_args or {})
        # Iterate over epochs and step-wise batches to traing the model.
        while not epochs.saturated:
            record_loss.keywords["epoch"] = int(epochs.value)
            for idx, (inp, tgt) in enumerate(self.training_data_loader, 1):
                rec = record_loss if (log_interval % idx == 0) else None
                self._training_step(idx, inp, tgt, rec)
                # Update effort constraints and break when one is met.
                nsteps.increment()
                if nsteps.saturated:
                    break
            epochs.increment()

    def _setup_loss_reporting(
            self,
            history_monitor: Optional["HistoryMonitor"]
        ) -> functools.partial:
        """Set up a function that performs loss reporting.

        Args:
            history_monitor: Optional history monitor for loss tracking.

        Returns:
            record_loss: functools.partial wrapping a function that takes
                (loss, batch_size, epoch, step) arguments and performs
                logging and optional history-monitoring.
        """
        # Gather the (fixed) number of samples and batches in the dataset.
        if self.training_data_loader is None:
            raise TypeError("No training dataset available.")
        n_batches = len(self.training_data_loader)
        n_samples = len(self.training_data_loader.dataset)  # type: ignore
        # Define a routine to log and opt. record loss at set batch indices.
        def record_loss(
                loss: float,
                batch_size: int,
                epoch: int,
                step: int,
            ) -> None:
            """Log and optionally monitor the computed loss."""
            nonlocal history_monitor, n_batches, n_samples
            logger.debug(
                f"Train Epoch: {epoch} Batch: {step}/{n_batches}"
                f"\tLoss: {loss:.6f}"
            )
            if history_monitor is not None:
                history_monitor.add_scalar(
                    metric={"loss": loss},
                    train=True,
                    epoch=epoch,
                    iteration=step,
                    batch_samples=batch_size,
                    num_batches=n_batches,
                    total_samples=n_samples,
                )
        # Wrap the routine as a functools.partial to enable kwargs setting.
        return functools.partial(record_loss)

    def _process_training_node_args(
            self,
            node_args: Dict[str, Any],
        ) -> None:
        """Process node-specific arguments prior to training.

        Args:
            node_args: Node-specific command-line arguments.
        """
        if node_args:
            logger.warning(
                "`BaseTrainingPlan._process_training_node_args` called: "
                "this has no effect and means some arguments may have been "
                "left un-handled."
            )

    def _training_step(
            self,
            idx: int,
            inputs: Any,
            target: Any,
            record_loss: Optional[functools.partial] = None,
        ) -> None:
        """Backend method to run a single training step.

        This method should always be called as part of `training_routine`
        and is merely factored out of it to enable its extension by model
        or framework-specific subclasses.
        """
        batch = (inputs, target, None)
        # Optionally report on the batch training loss.
        if record_loss is not None:
            loss = self.model.compute_loss([batch])
            record_loss(loss=loss, batch_size=len(inputs), step=idx)
        # Run the training step.
        self.optim.run_train_step(self.model, batch)

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
            metric_args: Keyword arguments for the metric's computation.
            history_monitor: HistoryMonitor instance,
                used to record computed metrics and communicate them to
                the researcher (server).
            before_train: Whether the evaluation is being performed
                before local training occurs, of afterwards. This is merely
                reported back through `history_monitor`.
        """
        if self.testing_data_loader is None:
            msg = f"{ErrorNumbers.FB605.value}: no validation dataset was set."
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # Set up a function to record batch-wise metrics.
        if history_monitor is not None:
            record_metrics = functools.partial(
                history_monitor.add_scalar,
                epoch=None,
                test=True,
                test_on_local_updates=(not before_train),
                test_on_global_updates=before_train,
                total_samples=len(self.testing_data_loader.dataset),
                num_batches=len(self.testing_data_loader),
            )
        # Set up a batch-wise metrics-computation function.
        evaluate, metric_name = self._setup_evaluate_step(metric, metric_args)
        # Iterate over the validation dataset and run the defined routine.
        for idx, (data, target) in enumerate(self.testing_data_loader, 1):
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
                f"Validation: Batch {idx}/{len(self.testing_data_loader)} "
                f"| Metric[{metric_name}]: {m_value}"
            )
            # Further parse, and report it (provided a monitor is set).
            if history_monitor is not None:
                m_dict = self._create_metric_result_dict(m_value, metric_name)
                record_metrics(
                    metric=m_dict,
                    iteration=idx,
                    batch_samples=len(target),
                )

    def _setup_evaluate_step(
            self,
            metric: Optional[MetricTypes],
            metric_args: Dict[str, Any],
        ) -> Tuple[Callable[[Any, Any], Any], str]:
        """Set up a batch-wise metrics-computation function.

        Args:
            metric: The metric used for validation.
                If None, use MetricTypes.ACCURACY.
            metric_args: Keyword arguments for the metric's computation.

        Returns:
            evaluate: Function computing a metric from (data, target) inputs.
            metric_name: Name associated with the metric being computed.
        """
        # Either use an optionally-implemented custom training routine.
        if hasattr(self, "testing_step"):
            evaluate = getattr(self, "testing_step")
            metric_name = "Custom"
        # Or use the provided `metric` (or its default value).
        else:
            if metric is None:
                metric = MetricTypes.ACCURACY
            metric_controller = Metrics()
            def evaluate(data, target):  # type: ignore
                nonlocal metric, metric_args, metric_controller
                output = self.predict(data)
                if isinstance(target, torch.Tensor):
                    target = target.detach().numpy()
                return metric_controller.evaluate(
                    target, output, metric=metric, **metric_args
                )
            metric_name = metric.name
        # Return the evaluation function and the metric's name.
        return evaluate, metric_name

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
    def _create_metric_result_dict(
            metric: Union[
                Dict[str, float], List[float], float,
                np.ndarray, torch.Tensor, List[torch.Tensor]
            ],
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
        if isinstance(metric, bool):
            pass
        elif isinstance(metric, (int, float, np.integer, np.floating)):
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

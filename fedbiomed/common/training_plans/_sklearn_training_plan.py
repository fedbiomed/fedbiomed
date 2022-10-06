"""TrainingPlan definitions for the scikit-learn ML framework.

This module implements the base class for all implementations of
Fed-BioMed training plans wrapping scikit-learn models.
"""

from abc import ABCMeta, abstractmethod
from io import StringIO
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import Metrics, MetricTypes
from fedbiomed.common.utils import get_method_spec
from fedbiomed.node.history_monitor import HistoryMonitor

from ._base_training_plan import BaseTrainingPlan


class _Capturer(list):
    """Capturing class for console outputs.

    This class is meant to be used as a context manager, to capture
    console outputs of scikit-learn models during their training in
    verbose mode.
    """

    def __enter__(self) -> '_Capturer':
        sys.stdout.flush()
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *_: Any) -> None:
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # Remove it from memory
        sys.stdout = self._stdout


class SKLearnTrainingPlan(BaseTrainingPlan, metaclass=ABCMeta):
    """Base class for Fed-BioMed wrappers of sklearn classes.

    Classes that inherit from this must meet the following conditions:
    - Specify a `_model_cls` class attribute that defines the type
      of scikit-learn model being wrapped for training.
    - Have a `model` attribute with an instance of that class.
    - Populate a `params_list` attribute during initialization with
      the names of `model`'s attributes that store weights which are
      to be trained locally and shared for aggregation.
    - Implement a `set_init_params` method that creates the model's initial
      trainable weights attributes.
    - Implement a `training_routine_hook` method that:
        1. Sets `data` and `target` attributes as outputs of a data loader.
        2. Calls `partial_fit` or a similar method of the wrapped model, so
           as to train it based on the input data.
    - Implement an `evaluate_loss` method, that calls the `_evaluate_loss_core`
      method of this base class.

    Attributes:
        params: parameters of the model, both learnable and non-learnable
        model_args: model arguments provided by researcher
        param_list: names of the parameters that will be used for aggregation
        dataset_path: the path to the dataset on the node
    """

    _model_cls: Type[BaseEstimator]        # wrapped model class
    _model_dep: Tuple[str, ...] = tuple()  # model-specific dependencies

    def __init__(self) -> None:
        """Initialize the SKLearnTrainingPlan."""
        super().__init__()
        self._model = self._model_cls()
        self._model_args = {}  # type: Dict[str, Any]
        self._training_args = {}  # type: Dict[str, Any]
        self._param_list = []  # type: List[str]
        self.__type = TrainingPlans.SkLearnTrainingPlan
        self._is_classification = False
        self.dataset_path = None
        self.add_dependency([
            "import inspect",
            "import numpy as np",
            "import pandas as pd",
            "from fedbiomed.common.training_plans import SKLearnTrainingPlan",
            "from fedbiomed.common.data import DataManager",
        ])
        self.add_dependency(list(self._model_dep))

    def post_init(
            self,
            model_args: Dict[str, Any],
            training_args: Dict[str, Any]
        ) -> None:
        """Process model, training and optimizer arguments.

        Args:
            model_args: Model arguments.
            training_args: Training arguments.
        """
        self._model_args = model_args
        self._model_args.setdefault("verbose", 1)
        self._training_args = training_args
        # Override default model parameters based on `self._model_args`.
        params = {
            key: self._model_args.get(key, val)
            for key, val in self._model.get_params()
        }
        self._model.set_params(**params)
        # Set up additional parameters (normally created by `self._model.fit`).
        self.set_init_params()

    @abstractmethod
    def set_init_params(self) -> None:
        """Initialize the model's trainable parameters."""
        return None

    def model_args(self) -> Dict[str, Any]:
        """Retrieve model arguments.

        Returns:
            Model arguments
        """
        return self._model_args

    def training_args(self) -> Dict[str, Any]:
        """Retrieve training arguments.

        Returns:
            Training arguments
        """
        return self._training_args

    def model(self) -> BaseEstimator:
        """Retrieve the wrapped scikit-learn model instance.

        Returns:
            Scikit-learn model instance
        """
        return self._model

    def training_routine(
            self,
            history_monitor: Optional[HistoryMonitor] = None,
            node_args: Optional[Dict[str, Any]] = None
        ) -> None:
        """Training routine, to be called once per round.

        Args:
            history_monitor (HistoryMonitor or None): optional HistoryMonitor
              instance, recording training metadata.
            node_args (dict or None): Command line arguments for node.
              These arguments can specify GPU use; however, this is not
              supported for scikit-learn models and thus will be ignored.
        """
        if self._model is None:
            raise FedbiomedTrainingPlanError('Wrapped model is None.')
        # Run preprocessing operations.
        self._preprocess()
        # Warn if GPU-use was expected (as it is not supported).
        if node_args is not None and node_args.get('gpu_only', False):
            logger.warning(
                'Node would like to force GPU usage, but sklearn training '
                'plan does not support it. Training on CPU.'
            )
        # Run the training loop.
        epochs = self._training_args.get("epochs", 1)
        for epoch in range(epochs):
            # Run the model-specific training routine.
            with _Capturer() as output:
                try:
                    self.training_routine_hook()
                except Exception as exc:
                    msg = (
                        f"{ErrorNumbers.FB605.value}: error while fitting "
                        f"the model: {exc}"
                    )
                    logger.critical(msg)
                    raise FedbiomedTrainingPlanError(msg)
            # Optionally log training outputs.
            # TODO: This part should be changed after mini-batch
            #       implementation is completed.
            if (history_monitor is not None) and self._model_args["verbose"]:
                loss = self.parse_training_loss(output, epoch)
                name = getattr(self._model, "loss", "")
                name = "Loss" + (f" {name}" if name else "")
                history_monitor.add_scalar(
                    metric={name: float(loss)},
                    iteration=1,
                    epoch=epoch,
                    train=True,
                    num_batches=1,
                    total_samples=len(self.training_data_loader[0]),  # FIXME
                    batch_samples=len(self.training_data_loader[0])   # FIXME
                )
                # TODO: For clustering; passes inertia value as scalar.
                # It should be implemented when KMeans implementation is ready.
                # history_monitor.add_scalar(
                #     'Inertia', self._model.inertia_, -1 , epoch
                # )
                # Need to find a way for Bayesian approaches.

    @abstractmethod
    def training_routine_hook(self) -> None:
        """Model-specific training routine for an epoch.

        This method needs to be implemented by SKLearnTrainingPlan
        child classes, and is called as part of `training_routine`
        (that notably enforces exception catching).
        """
        return None

    @abstractmethod
    def parse_training_loss(
            self,
            output: List[str],
            epoch: int
        ) -> float:
        """Model-specific loss-parsing from model training outputs.

        This method needs to be implemented by SKLearnTrainingPlan
        child classes, and is called as part of `training_routine`.
        It may use the `_parse_training_losses` staticmethod.
        """
        return NotImplemented

    @staticmethod
    def _parse_training_losses(
            output: List[str],
            epoch: int
        ) -> List[float]:
        """Evaluate the wrapped model's loss on the training dataset.

        Args:
            output (StringIO): Capture text output of the model's training.
            epoch (int): Epoch number (as part of the current round).

        Returns:
            list[float]: List of loss values parsed from `output`.
        """
        losses = []
        for line in output:
            # Try parsing the loss value (or go on to the next line).
            split = line.split("loss: ")
            if len(split) == 1:
                continue
            try:
                loss = float(split[-1])
            except ValueError as exc:
                logger.error(f"Value error during monitoring: {exc}")
            # Record and log (at debug level) the loss value.
            losses.append(loss)
            logger.debug(
                f"Train Epoch: {epoch} [Batch All Samples]"
                f"\tLoss: {loss:.6f}"
            )
        return losses

    def testing_routine(self,
                        metric: Union[MetricTypes, None],
                        metric_args: Dict[str, Any],
                        history_monitor,
                        before_train: bool):
        """
        Validation routine for SGDSkLearnModel. This method is called by the Round class if validation
        is activated for the Federated training round

        Args:
            metric (MetricType, None): The metric that is going to be used for validation. Should be
                an instance of MetricTypes. If it is None and there is no `testing_step` is defined
                by researcher method will raise an Exception. Defaults to ACCURACY.

            history_monitor (HistoryMonitor): History monitor class of node side to send validation results
                to researcher.

            before_train (bool): If True, this means validation is going to be performed after loading model parameters
              without training. Otherwise, after training.

        """
        # Use accuracy as default metric
        if metric is None:
            metric = MetricTypes.ACCURACY

        if self.testing_data_loader is None:
            msg = ErrorNumbers.FB605.value + ": can not find dataset for validation."
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

        # Check validation data loader is exists
        data, target = self.testing_data_loader

        # At the first round model won't have classes_ attribute
        if self._is_classification and not hasattr(self._model, 'classes_'):
            classes = self._classes_from_concatenated_train_test()
            setattr(self._model, 'classes_', classes)

        # Build metrics object
        metric_controller = Metrics()
        tot_samples = len(data)

        # Use validation method defined by user
        if hasattr(self, 'testing_step') and callable(self.testing_step):
            try:
                m_value = self.testing_step(data, target)
            except Exception as err:
                msg = ErrorNumbers.FB605.value + \
                      ": error - " + \
                      str(err)
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)

            # If custom validation step returns None
            if m_value is None:
                msg = ErrorNumbers.FB605.value + \
                      ": metric function has returned None"
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)

            metric_name = 'Custom'

        # If metric is defined use pre-defined validation for Fed-BioMed
        else:
            if metric is None:
                metric = MetricTypes.ACCURACY
                logger.info(f"No `testing_step` method found in TrainingPlan and `test_metric` is not defined "
                            f"in the training arguments `: using default metric {metric.name}"
                            " for model validation")
            else:
                logger.info(
                    f"No `testing_step` method found in TrainingPlan: using defined metric {metric.name}"
                    " for model validation.")

            try:
                pred = self._model.predict(data)
            except Exception as e:
                msg = ErrorNumbers.FB605.value + \
                      ": error during predicting validation data set - " + \
                      str(e)
                logger.critical(msg)
                raise FedbiomedTrainingPlanError(msg)

            m_value = metric_controller.evaluate(target, pred, metric=metric, **metric_args)
            metric_name = metric.name

        metric_dict = self._create_metric_result_dict(m_value, metric_name=metric_name)

        # For logging in node console
        logger.debug('Validation: [{}/{}] | Metric[{}]: {}'.format(len(target), tot_samples,
                                                                metric.name, m_value))

        # Send scalar values via general/feedback topic
        if history_monitor is not None:
            history_monitor.add_scalar(metric=metric_dict,
                                       iteration=1,  # since there is only one
                                       epoch=None,  # no epoch
                                       test=True,  # means that for sending validation metric
                                       test_on_local_updates=False if before_train else True,
                                       test_on_global_updates=before_train,
                                       total_samples=tot_samples,
                                       batch_samples=len(target),
                                       num_batches=1)

    def _classes_from_concatenated_train_test(self) -> np.ndarray:
        """
        Method for getting all classes from validatino and target dataset. This action is required
        in case of some class only exist in training subset or validation subset

        Returns:
            np.ndarray: numpy array containing unique values from the whole dataset (training + validation dataset)
        """

        target_test = self.testing_data_loader[1] if self.testing_data_loader is not None else np.array([])
        target_train = self.training_data_loader[1] if self.training_data_loader is not None else np.array([])

        target_test_train = np.concatenate((target_test, target_train))

        return np.unique(target_test_train)

    def _process_data_loader(
            self,
            method: Callable[..., Any]
        ) -> None:
        """Handle a data-loader pre-processing action.

        Args:
            method (Callable) : Process method that is to be executed.

        Raises:
            FedbiomedTrainingPlanError:
              - if the method does not have 1 positional argument (dataloader)
              - if running the method fails
              - if the method does not return a dataloader of the same type as
               its input
        """
        # Check that the preprocessing method has a proper signature.
        argspec = get_method_spec(method)
        if len(argspec) != 2:
            msg = (
                f"{ErrorNumbers.FB605.value}: preprocess method of type "
                "`PreprocessType.DATA_LOADER` should expect two arguments: "
                "the inputs and targets composing the training dataset."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # Try running the preprocessor.
        try:
            inputs, target = self.training_data_loader
            outputs = method(inputs, target)
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
        if (len(outputs) == 2) and all(isinstance(x, np.ndarray) for x in outputs):
            self.training_data_loader = outputs
            logger.debug(
                "Data loader for training routine has been updated "
                f"by the process `{method.__name__}`."
            )
        else:
            msg = (
                f"{ErrorNumbers.FB605.value}: the return type of the "
                f"`{method.__name__}` preprocess method was expected "
                f"to be a pair of numpy arrays, but was {type(outputs)}."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

    def _compute_support(self, targets: np.ndarray) -> np.ndarray:
        """
        Computes support, i.e. the number of items per
        classes. It is designed from the way scikit learn linear model
        `fit_binary` and `_prepare_fit_binary` have been implemented.

        Args:
            targets (np.ndarray): targets that contain labels
            used for training models

        Returns:
            np.ndarray: support
        """
        support = np.zeros((len(self._model.classes_),))

        # to see how multi classification is done in sklearn, please visit:
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L324   # noqa
        # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L738   # noqa
        for i, aclass in enumerate(self._model.classes_):
            # in sklearn code, in `fit_binary1`, `i`` seems to be
            # iterated over model.classes_
            # (https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/linear_model/_stochastic_gradient.py#L774)
            # We cannot directly know for each loss that has been logged from scikit learn
            #  which labels it corresponds to. This is our best guest
            idx = targets == aclass
            support[i] = np.sum(targets[targets.astype(int)[idx]])

        return support

    def save(
            self,
            filename: str,
            params: Union[None, Dict[str, np.ndarray], Dict[str, Any]] = None
        ) -> None:
        """Save the wrapped model and its trainable parameters.

        This method is designed for parameter communication. It
        uses the joblib.dump function, which in turn uses pickle
        to serialize the model. Note that unpickling objects can
        lead to arbitrary code execution; hence use with care.

        Args:
            filename (str): Path to the output file.
            params (dict or None): Model parameters to enforce and save.
              This may either be a {name: array} parameters dict, or a
              nested dict that stores such a parameters dict under the
              'model_params' key (in the context of the Round class).

        Notes:
            Save can be called from Job or Round.
              From Round it is called with params (as a complex dict).
              From Job it is called with no params in constructor, and
              with params in update_parameters.
        """
        # Optionally overwrite the wrapped model's weights.
        if params:
            if isinstance(params.get('model_params'), dict):  # in a Round
                params = params["model_params"]
            for key, val in params.items():
                setattr(self._model, key, val)
        # Save the wrapped model (using joblib, hence pickle).
        with open(filename, "wb") as file:
            joblib.dump(self._model, file)

    def load(
            self,
            filename: str,
            to_params: bool = False
        ) -> Union[BaseEstimator, Dict[str, Dict[str, np.ndarray]]]:
        """Load a scikit-learn model dump, overwriting the wrapped model.

        This method uses the joblib.load function, which in turn uses
        pickle to deserialize the model. Note that unpickling objects
        can lead to arbitrary code execution; hence use with care.

        This function updates the `_model` private attribute with the
        loaded instance, and returns either that same model or a dict
        wrapping its trainable parameters.

        Args:
            filename (str): The path to the pickle file to load.
            to_params (bool): Whether to return the model's parameters
              wrapped as a dict rather than the model instance.

        Notes:
            Load can be called from a Job or Round:
              From Round it is called to return the model.
              From Job it is called with to return its parameters dict.

        Returns:
            dictionary with the loaded parameters
        """
        # Deserialize the dump, type-check the instance and assign it.
        with open(filename, "rb") as file:
            model = joblib.load(file)
        if not isinstance(model, self._model_cls):
            msg = (
                f"{ErrorNumbers.FB304.value}: reloaded model does not conform "
                f"to expectations: should be of type {self._model_cls}, not "
                f"{type(model)}."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        self._model = model
        # Optionally return the model's pseudo state dict instead of it.
        if to_params:
            params = {k: getattr(self._model, k) for k in self._param_list}
            return {"model_params": params}
        return self._model

    def get_model(self) -> BaseEstimator:
        """Get the wrapped scikit-learn model.

        Returns:
            sklearn.base.BaseEstimator: the scikit-learn model instance.
        """
        return self._model

    def type(self) -> TrainingPlans:
        """Getter for training plan type """
        return self.__type

    def after_training_params(self) -> Dict[str, np.ndarray]:
        """Return the wrapped model's trainable parameters' current values.

        This method returns a dict containing parameters that need
        to be reported back and aggregated in a federated learning
        setting.

        Returns:
            dict[str, np.ndarray]: the trained parameters to aggregate.
        """
        return {key: getattr(self._model, key) for key in self._param_list}

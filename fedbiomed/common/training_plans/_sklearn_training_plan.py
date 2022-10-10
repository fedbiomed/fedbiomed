"""TrainingPlan definitions for the scikit-learn ML framework.

This module implements the base class for all implementations of
Fed-BioMed training plans wrapping scikit-learn models.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.data import NPDataLoader
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.node.history_monitor import HistoryMonitor

from ._base_training_plan import BaseTrainingPlan


class SKLearnTrainingPlan(BaseTrainingPlan, metaclass=ABCMeta):
    """Base class for Fed-BioMed wrappers of sklearn classes.

    Classes that inherit from this abstract class must:
    - Specify a `_model_cls` class attribute that defines the type
      of scikit-learn model being wrapped for training.
    - Implement a `set_init_params` method that:
      - sets and assigns the model's initial trainable weights attributes.
      - populates the `_param_list` attribute with names of these attributes.
    - Implement a `_training_routine` method that performs a training round
      based on `self.train_data_loader` (which is a `NPDataLoader`).

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
        args = (train_data_loader, test_data_loader)
        if not all(isinstance(data, NPDataLoader) for data in args):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan expects "
                "NPDataLoader instances as training and testing data "
                f"loaders, but received {type(train_data_loader)} "
                f"and {type(train_data_loader)} respectively."
            )
            logger.error(msg)
            raise FedbiomedTrainingPlanError(msg)
        self.training_data_loader = train_data_loader
        self.testing_data_loader = test_data_loader

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
        if not isinstance(self._model, BaseEstimator):
            msg = (
                f"{ErrorNumbers.FB317.value}: model should be a scikit-learn "
                f"estimator, but is of type {type(self._model)}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        if not isinstance(self.training_data_loader, NPDataLoader):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan cannot "
                "be trained without a NPDataLoader as `training_data_loader`."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # Run preprocessing operations.
        self._preprocess()
        # Warn if GPU-use was expected (as it is not supported).
        if node_args is not None and node_args.get('gpu_only', False):
            logger.warning(
                'Node would like to force GPU usage, but sklearn training '
                'plan does not support it. Training on CPU.'
            )
        # Run the model-specific training routine.
        try:
            self._training_routine(history_monitor)
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB605.value}: error while fitting "
                f"the model: {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

    @abstractmethod
    def _training_routine(
            self,
            history_monitor: Optional[HistoryMonitor] = None
        ) -> None:
        """Model-specific training routine backend.

        Args:
            history_monitor (HistoryMonitor or None): optional HistoryMonitor
              instance, recording the loss value during training.

        This method needs to be implemented by SKLearnTrainingPlan
        child classes, and is called as part of `training_routine`
        (that notably enforces preprocessing and exception catching).
        """
        return None

    def testing_routine(
            self,
            metric: Optional[MetricTypes],
            metric_args: Dict[str, Any],
            history_monitor: Optional[HistoryMonitor],
            before_train: bool
        ) -> None:
        """Evaluation routine, to be called once per round.

        Note: if the training plan implements a `testing_step` method
              (the signature of which is func(data, target) -> metrics)
              then it will be used rather than the input metric.

        Args:
            metric (MetricType, None): The metric used for validation.
              If None, use MetricTypes.ACCURACY.
            history_monitor (HistoryMonitor): HistoryMonitor instance,
              used to record computed metrics and communicate them to
              the researcher (server).
            before_train (bool): Whether the evaluation is being performed
              before local training occurs, of afterwards. This is merely
              reported back through `history_monitor`.
        """
        # Check that the testing data loader is of proper type.
        if not isinstance(self.testing_data_loader, NPDataLoader):
            msg = (
                f"{ErrorNumbers.FB310.value}: SKLearnTrainingPlan cannot be "
                "evaluated without a NPDataLoader as `testing_data_loader`."
            )
            logger.error(msg)
            raise FedbiomedTrainingPlanError(msg)
        # If required, make up for the lack of specifications regarding target
        # classification labels.
        if self._is_classification and not hasattr(self._model, 'classes_'):
            classes = self._classes_from_concatenated_train_test()
            setattr(self._model, 'classes_', classes)
        # If required, select the default metric (accuracy or mse).
        if metric is None:
            if self._is_classification:
                metric = MetricTypes.ACCURACY
            else:
                metric = MetricTypes.MEAN_SQUARE_ERROR
        # Delegate the actual evalation routine to the parent class.
        super().testing_routine(
            metric, metric_args, history_monitor, before_train
        )

    def predict(
            self,
            data: Any,
            target: Any,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Return model predictions for a given batch of input features.

        This method is called as part of `testing_routine`, to compute
        predictions based on which evaluation metrics are computed. It
        will however be skipped if a `testing_step` method is attached
        to the training plan, than wraps together a custom routine to
        compute an output metric directly from a (data, target) batch.

        Args:
            data: Array-like (or tensor) structure containing batched
              input features.
            target: Array-like (or tensor) structure containing batched
              target values (not to be used for prediction, but to be
              processed into a returned numpy array).

        Returns:
            np.ndarray: Output predictions, converted to a numpy array
              (as per the `fedbiomed.common.metrics.Metrics` specs).
            np.ndarray: Target output values, converted from `target`.
        """
        return self._model.predict(data), target

    def _classes_from_concatenated_train_test(self) -> np.ndarray:
        """Return unique target labels from the training and testing datasets.

        Returns:
            np.ndarray: numpy array containing the unique values from the
              targets wrapped in the training and testing NPDataLoader
              instances.
        """
        target = np.array([])
        for loader in (self.training_data_loader, self.testing_data_loader):
            if isinstance(loader, NPDataLoader):
                target = np.concatenate(target, loader.get_target())
        return np.unique(target)

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

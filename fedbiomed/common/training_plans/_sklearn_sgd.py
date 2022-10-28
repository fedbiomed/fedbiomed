# coding: utf-8

"""Training Plan designed to wrap scikit-learn SGD Classifier/Regressor."""

import functools
from typing import Any, Dict, Optional, Union

import declearn
import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import SGDClassifier, SGDRegressor  # type: ignore

from fedbiomed.common.data import DataLoaderTypes
from fedbiomed.common.logger import logger

from ._base import TrainingPlan


class SklearnSGDTrainingPlan(TrainingPlan):
    """Base class for training plans using sklearn SGD Classifier/Regressor.

    All concrete sklearn-sgd training plans inheriting this class
    should implement:
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

    _model_cls=declearn.model.sklearn.SklearnSGDModel
    _data_type=DataLoaderTypes.NUMPY

    def __init__(
            self,
            model: Union[SGDClassifier, SGDRegressor, Dict[str, Any]],
            optim: Union[declearn.optimizer.Optimizer, Dict[str, Any]],
            **kwargs: Any
        ) -> None:
        """Construct the torch training plan.

        Args:
            model: Base `sklearn.linear_model.SGDClassifier` or `SGDRegressor`
                object to be interfaced using a declearn `SklearnSGDModel`,
                or config dict of the latter.
            optim: declearn.optimizer.Optimizer instance of config dict.
        """
        super().__init__(model, optim, **kwargs)

    def predict(
            self,
            data: ArrayLike,
        ) -> np.ndarray:
        """Return model predictions for a given batch of input features.

        This method is called as part of `testing_routine`, to compute
        predictions based on which evaluation metrics are computed. It
        will however be skipped if a `testing_step` method is attached
        to the training plan, than wraps together a custom routine to
        compute an output metric directly from a (data, target) batch.

        Args:
            data: numpy-compatible array-like structure containing batched
                input features.

        Returns:
            Output predictions, converted to a numpy array (as per the
                `fedbiomed.common.metrics.Metrics` specs).
        """
        model = (
            getattr(self.model, "_model")
        )  # type: Union[SGDClassifier, SGDRegressor]
        preds = model.predict(data)  # type: np.ndarray
        return preds

    def _process_training_node_args(
            self,
            node_args: Dict[str, Any],
        ) -> None:
        # Warn if GPU-use was expected (as it is not supported).
        if node_args.get("gpu_only", False):
            logger.warning(
                "Node would like to force GPU usage, but sklearn training "
                "plan does not support it. Training on CPU."
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
        target = target.ravel()  # scikit-learn expects 1d-array targets
        super()._training_step(idx, inputs, target, record_loss)

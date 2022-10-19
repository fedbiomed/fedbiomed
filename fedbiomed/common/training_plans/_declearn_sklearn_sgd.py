# coding: utf-8

"""Training Plan designed to wrap scikit-learn SGD Classifier/Regressor."""

from typing import Any, Dict, Union

import declearn
import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import SGDClassifier, SGDRegressor

from fedbiomed.common.data import NPDataLoader
from fedbiomed.common.logger import logger

from ._declearn_training_plan import BaseTrainingPlan


class SklearnSGDTrainingPlan(BaseTrainingPlan):
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
    _data_type=NPDataLoader

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
        return model.predict(data)

    def _process_training_node_args(
            self,
            node_args: Dict[str, Any],
        ) -> None:
        # Warn if GPU-use was expected (as it is not supported).
        if node_args.get('gpu_only', True):
            logger.warning(
                'Node would like to force GPU usage, but sklearn training '
                'plan does not support it. Training on CPU.'
            )

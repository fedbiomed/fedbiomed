# coding: utf-8

"""Training Plan designed to wrap scikit-learn SGD Classifier/Regressor."""

import functools
from abc import ABCMeta
from typing import Any, Dict, Optional, Union

import declearn
import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import SGDClassifier, SGDRegressor  # type: ignore

from fedbiomed.common.data import DataLoaderTypes
from fedbiomed.common.logger import logger

from ._base import TrainingPlan


class SklearnSGDTrainingPlan(TrainingPlan, metaclass=ABCMeta):
    """Base class for training plans using sklearn SGD Classifier/Regressor.

    All concrete sklearn-sgd training plans inheriting this class
    should implement:
        * the `training_data` method:
            to define how to set up the `fedbiomed.data.DataManager`
            wrapping the training (and, by split, validation) data
        * (opt.) the `init_model` method:
            to build the scikit-learn model to be used, in a more
            restrictive manner than the default implemented here
        * (opt.) the `init_optim` method:
            to build the optimizer that is to be used (by default,
            use the optimizer config passed through training args)
        * (opt.) the `testing_step` method:
            to override the evaluation behavior and compute
            a batch-wise (set of) metric(s)

    Attributes:
        model: declearn SklearnSGDModel wrapping the model being trained.
        optim: declearn Optimizer in charge of node-side optimization.
        pre_processes: Preprocess functions that will be applied to the
            training data at the beginning of the training routine.
        training_data_loader: Data loader used in the training routine.
        testing_data_loader: Data loader used in the validation routine.
    """

    _model_cls=declearn.model.sklearn.SklearnSGDModel
    _data_type=DataLoaderTypes.NUMPY

    def init_model(
            self,
            model_args: Dict[str, Any],
        ) -> Union[SGDClassifier, SGDRegressor]:
        """Build and return a scikit-learn SGDClassifier or SGDRegressor model.

        !!! info "Note"
            This method provides with a default builder for either a classifier
            or regressor model. It may be overridden in user-defined subclasses
            in order to restrict ~ automate the model kind and hyper-parameter
            choices.

        Args:
            model_args: Dict containing hyper-parameters to specify the model.
                It should contain "kind" (either "classifier" or "regressor")
                to specify the type of model, as well as any keyword argument
                defined as part of the SGDClassifier or SGDRegressor API.

        Returns:
            model: sklearn SGDClassifier or SGDRegressor instance.
        """
        # Gather the only required hyper-parameter: "kind".
        kind = model_args.get("kind", None)
        if kind is None:
            raise KeyError("Missing required 'model_args' field: 'kind'.")
        if kind not in ("classifier", "regressor"):
            raise TypeError(
                "'kind' field in 'model_args' should be 'classifier' "
                f"or 'regressor', not '{kind}'"
            )
        # GAther all other supported keyword arguments.
        names = {
            "loss", "penalty", "alpha", "l1_ratio",
            "epsilon", "fit_intercept", "n_jobs"
        }
        kwargs = {key: val for key, val in model_args.items() if key in names}
        # SGDClassifier case.
        if kind == "classifier":
            sk_cls = SGDClassifier
        # SGDRegressor case.
        elif kind == "regressor":
            kwargs.pop("n_jobs")  # unsupported for SGDRegressor
            sk_cls = SGDRegressor
        # Instantiate the sklearn model and return it.
        return sk_cls(**kwargs,)

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

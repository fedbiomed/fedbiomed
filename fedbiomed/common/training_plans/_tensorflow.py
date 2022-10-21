# coding: utf-8

"""Training Plan designed to wrap tensorflow `keras.Layer` models."""

from typing import Any, Dict, Optional, Union

import declearn
import declearn.model.tensorflow
import numpy as np
import tensorflow as tf  # type: ignore

from fedbiomed.common.data import NPDataLoader

from ._base import TrainingPlan


class TensorflowTrainingPlan(TrainingPlan):
    """Base class for training plans wrapping `tf.keras.Layer` models.

    All concrete torch training plans inheriting this class should implement:
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

    _model_cls=declearn.model.tensorflow.TensorflowModel
    _data_type=NPDataLoader  # FIXME: implement a dedicated data loader

    def __init__(
            self,
            model: Union[tf.keras.layers.Layer, Dict[str, Any]],
            optim: Union[declearn.optimizer.Optimizer, Dict[str, Any]],
            loss: Optional[Union[str, tf.keras.losses.Loss]] = None,
            **kwargs: Any
        ) -> None:
        """Construct the tensorflow training plan.

        Args:
            model: Base `tf.keras.Layer` object to be interfaced using
                a declearn `TensorflowModel`, or config dict of the latter.
            optim: declearn.optimizer.Optimizer instance of config dict.
            loss: Optional `tf.keras.losses.Loss` or name of one, defining
                the model's loss (unused if `model` is a config dict).
                If a function (name) is provided rather than an object, it
                will be converted to a Loss instance, and an exception may
                be raised if that fails.
        """
        super().__init__(model, optim, loss=loss, **kwargs)

    def predict(
            self,
            data: tf.Tensor,
        ) -> np.ndarray:
        """Return model predictions for a given batch of input features.

        This method is called as part of `testing_routine`, to compute
        predictions based on which evaluation metrics are computed. It
        will however be skipped if a `testing_step` method is attached
        to the training plan, than wraps together a custom routine to
        compute an output metric directly from a (data, target) batch.

        Args:
            data: tensorflow.Tensor containing batched input features.

        Returns:
            Output predictions, converted to a numpy array (as per the
                `fedbiomed.common.metrics.Metrics` specs).
        """
        model = getattr(self.model, "_model")  # type: tf.keras.Model
        pred = model(data, training=False).numpy()  # type: np.ndarray
        return pred

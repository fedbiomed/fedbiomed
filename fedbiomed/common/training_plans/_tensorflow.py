# coding: utf-8

"""Training Plan designed to wrap tensorflow `keras.Layer` models."""

from abc import ABCMeta, abstractmethod
from typing import Any, Union

import declearn
import declearn.model.tensorflow
import numpy as np
import tensorflow as tf  # type: ignore

from fedbiomed.common.data import DataLoaderTypes

from ._base import TrainingPlan


class TensorflowTrainingPlan(TrainingPlan, metaclass=ABCMeta):
    """Base class for training plans wrapping `tf.keras.Layer` models.

    All concrete tensorflow training plans inheriting this class should
    implement:
        * the `training_data` method:
            to define how to set up the `fedbiomed.data.DataManager`
            wrapping the training (and, by split, validation) data
        * the `init_model` method:
            to build the model to be used, as a tensorflow.keras.Layer
        * the `init_loss` method:
            to build the keras loss object to be used
        * (opt.) the `init_optim` method:
            to build the optimizer that is to be used (by default,
            use the optimizer config passed through training args)
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
    _data_type=DataLoaderTypes.NUMPY  # FIXME: implement a dedicated data loader

    @abstractmethod
    def init_loss(self) -> Union[str, tf.keras.losses.Loss]:
        """Return the loss function to use, as a keras object or name of one.

        Returns:
            loss: Keras Loss instance, or name of one. If a function (name)
                is provided, it will be converted to a Loss instance, and
                an exception may be raised if that fails.
        """
        return NotImplemented

    def _wrap_base_model(
            self,
            model: Any,
        ) -> declearn.model.api.Model:
        if not isinstance(model, tf.keras.layers.Layer):
            raise TypeError("The base model should be a keras Layer.")
        loss = self.init_loss()
        return self._model_cls(model, loss=loss)

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

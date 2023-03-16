# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""'Model' abstract base class defining an API to interface framework-specific models."""

from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Union, Type

import torch
from sklearn.base import BaseEstimator
from declearn.model.api import Vector

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.logger import logger


class Model(metaclass=ABCMeta):
    """Model abstraction, that wraps and handles both native models

    Attributes:
        model: native model, written with frameworks supported by Fed-BioMed.
        model_args: model arguments stored as a dictionary, that provides additional
            arguments for building/using models. Defaults to None.
    """

    _model_type: ClassVar[Type[Any]]

    def __init__(self, model: Union[BaseEstimator, torch.nn.Module]):
        """Constructor of Model abstract class

        Args:
            model: native model wrapped, of child-class-specific type.
        """
        self._validate_model_type(model)
        self.model: Any = model
        self.model_args: Optional[Dict[str, Any]] = None

    def set_model(self, model: Union[BaseEstimator, torch.nn.Module]):
        self._validate_model_type(model)
        self.model = model
        
    def _validate_model_type(self, model: Union[BaseEstimator, torch.nn.Module]):
        if not isinstance(model, self._model_type):
            err_msg = (
                f"{ErrorNumbers.FB622.value}: unproper 'model' input type: "
                f"expected '{self._model_type}', but 'got {type(model)}'."
            )
            logger.critical(err_msg)
            raise FedbiomedModelError(err_msg)

    @abstractmethod
    def init_training(self):
        """Initializes parameters before model training"""

    @abstractmethod
    def train(self, inputs: Any, targets: Any, **kwargs) -> None:
        """Trains model given inputs and targets data

        !!! warning "Warning"
            Please run `init_training` method before running `train` method,
            so to initialize parameters needed for model training"

        !!! warning "Warning"
            This function may not update weights. You may need to call `apply_updates`
            to apply updates to the model

        Args:
            inputs (Any): input (training) data.
            targets (Any): target values.
        """

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Returns model predictions given input values

        Args:
            inputs (Any): input values.

        Returns:
            Any: predictions.
        """

    @abstractmethod
    def apply_updates(self, updates: Any):
        """Applies updates to the model.

        Args:
            updates (Any): model updates.
        """

    @abstractmethod
    def get_weights(self,
                    as_vector: bool = False,
                    only_trainable: bool = False,) -> Union[Dict[str, Any], Vector]:
        """Return a copy of the model's trainable weights.

        Args:
            as_vector: Whether to wrap returned weights into a declearn Vector.
            only_trainable (bool, optional): whether to gather weights only on trainable layers (ie
                non-frozen layers) or all layers (trainable and frozen). Defaults to False, (trainable and
                frozen ones)

        Returns:
            Model weights, as a dictionary mapping parameters' names to their
                value, or as a declearn Vector structure wrapping such a dict.
        """

    @abstractmethod
    def get_gradients(self, as_vector: bool = False) -> Union[Dict[str, Any], Vector]:
        """Return computed gradients attached to the model.

        Args:
            as_vector: Whether to wrap returned gradients into a declearn Vector.

        Returns:
            Gradients, as a dictionary mapping parameters' names to their gradient's
                value, or as a declearn Vector structure wrapping such a dict.
        """

    @abstractmethod
    def load(self, filename: str) -> None:
        """Loads model from a file.

        Args:
            filename: path towards the file where the model has been saved.
        """

    @abstractmethod
    def save(self, filename: str) -> None:
        """Saves model into a file.

        Args:
            filename: path to the file, where will be saved the model.
        """

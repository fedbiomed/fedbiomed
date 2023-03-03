# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import torch
from sklearn.base import BaseEstimator

from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.constants import ErrorNumbers


class Model(metaclass=ABCMeta):
    """Model abstraction, that wraps and handles both native models
    
    Attributes:
        model: native model, written with frameworks supported by Fed-BioMed.
        model_args: model arguments stored as a dictionary, that provides additional
            arguments for building/using models. Defaults to None.
    """
    model: Union[torch.nn.Module, BaseEstimator]
    model_args: Dict[str, Any]
    
    def __init__(self, model: Union[torch.nn.Module, BaseEstimator]):
        """Constructor of Model abstract class

        Args:
            model (Union[torch.nn.Module, BaseEstimator]): native model wrapped
        """
        self.model = model
        self.model_args = None

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
    def get_weights(self, return_type: Callable = None) -> Any:
        """Returns weights of the model.

        Args:
            return_type: Function that converts the dictionary mapping layers to model weights into another data
                structure. `return_type` should be used mainly with `declearn`'s `Vector`s. Defaults to None.

        Returns:
            Any: model's weights.
        """
    @abstractmethod
    def get_gradients(self, return_type: Callable = None) -> Any:
        """Returns computed gradients after training a model

        Args:
            return_type (Callable, optional): _description_. Defaults to None.
        """

    @abstractmethod
    def load(self, filename: str):
        """Loads model from a file.

        Args:
            filename: path towards the file where the model has been saved.
        """
    @abstractmethod
    def save(self, filename: str):
        """Saves model into a file.

        Args:
            filename: path to the file, where will be saved the model.
        """
    @staticmethod
    def _validate_return_type(return_type: Optional[Callable] = None) -> None:
        """Checks that `return_type` argument is either a callable or None.

        Otherwise, raises an error

        Args:
            return_type: callable that will be used to convert a dictionary into another data structure
                (e.g. a declearn Vector). Defaults to None.

        Raises:
            FedbiomedModelError: raised if `return_type` argument is neither a callable nor `None`.
        """
        if not (return_type is None or callable(return_type)):
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}. Argument `return_type` should be either None or callable, "
                f"but got {type(return_type)} instead"
            )

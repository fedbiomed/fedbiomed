# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""'Model' abstract base class defining an API to interface framework-specific models."""

from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict, Generic, Optional, Union, Type, TypeVar, List

from declearn.model.api import Vector

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.logger import logger


# Generic type variables for annotations: specify types that are abstract
# at this level, but have to be coherent when defined by children classes.
_MT = TypeVar("_MT")  # model type
_DT = TypeVar("_DT")  # data array type
_VT = TypeVar("_VT", bound=Vector)  # declearn vector type


class Model(Generic[_MT, _DT, _VT], metaclass=ABCMeta):
    """Model abstraction, that wraps and handles both native models

    Attributes:
        model: native model, written with frameworks supported by Fed-BioMed.
        model_args: model arguments stored as a dictionary, that provides additional
            arguments for building/using models. Defaults to None.
    """

    _model_type: ClassVar[Type[Any]]

    def __init__(self, model: _MT):
        """Constructor of Model abstract class

        Args:
            model: native model wrapped, of child-class-specific type.
        """
        if not isinstance(model, self._model_type):
            err_msg = (
                f"{ErrorNumbers.FB622.value}: unproper 'model' input type: "
                f"expected '{self._model_type}', but 'got {type(model)}'."
            )
            logger.critical(err_msg)
            raise FedbiomedModelError(err_msg)
        self.model: Any = model
        self.model_args: Optional[Dict[str, Any]] = None

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
    def get_weights(self, as_vector: bool = False) -> Union[Dict[str, _DT], _VT]:
        """Return a copy of the model's trainable weights.

        Args:
            as_vector: Whether to wrap returned weights into a declearn Vector.

        Returns:
            Model weights, as a dictionary mapping parameters' names to their
                value, or as a declearn Vector structure wrapping such a dict.
        """

    @abstractmethod
    def set_weights(self, weights: Union[Dict[str, _DT], _VT]) -> None:
        """Assign new values to the model's trainable weights.

        Args:
            weights: Model weights, as a dict mapping parameters' names to their
                value, or as a declearn Vector structure wrapping such a dict.
        """

    @abstractmethod
    def get_gradients(self, as_vector: bool = False) -> Union[Dict[str, Any], _VT]:
        """Return computed gradients attached to the model.

        Args:
            as_vector: Whether to wrap returned gradients into a declearn Vector.

        Returns:
            Gradients, as a dictionary mapping parameters' names to their gradient's
                value, or as a declearn Vector structure wrapping such a dict.
        """

    @abstractmethod
    def flatten(self) -> List[float]:
        """Flattens model weights

        Returns:
            List of model weights as float.
        """

    @abstractmethod
    def export(self, filename: str) -> None:
        """Export the wrapped model to a dump file.

        Args:
            filename: path to the file where the model will be saved.

        !!! info "Notes":
            This method is designed to save the model to a local dump
            file for easy re-use by the same user, possibly outside of
            Fed-BioMed. It is not designed to produce trustworthy data
            dumps and is not used to exchange models and their weights
            as part of the federated learning process.
        """

    def reload(self, filename: str) -> None:
        """Import and replace the wrapped model from a dump file.

        Args:
            filename: path to the file where the model has been exported.

        !!! info "Notes":
            This method is designed to load the model from a local dump
            file, that might not be in a trustworthy format. It should
            therefore only be used to re-load data exported locally and
            not received from someone else, including other FL peers.

        Raises:
            FedbiomedModelError: if the reloaded instance is of unproper type.
        """
        model = self._reload(filename)
        if not isinstance(model, self._model_type):
            err_msg = (
                f"{ErrorNumbers.FB622.value}: unproper type for imported model"
                f": expected '{self._model_type}', but 'got {type(model)}'."
            )
            logger.critical(err_msg)
            raise FedbiomedModelError(err_msg)
        self.model = model

    @abstractmethod
    def _reload(self, filename: str) -> _MT:
        """Model-class-specific backend to the `reload` method.

        Args:
            filename: path to the file where the model has been exported.

        Returns:
            model: reloaded model instance to be wrapped, that will be type-
                checked as part of the calling `reload` method.
        """

    @abstractmethod
    def unflatten(
            self,
            weights_vector: List[float]
    ) -> None:
        """Revert flatten model weights back model-dict form.

        Args:
            weights_vector: Vectorized model weights to convert dict

        Returns:
            Model dictionary
        """

        if not isinstance(weights_vector, list) or not all([isinstance(w, float) for w in weights_vector]):
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622} `weights_vector should be 1D list of float containing flatten model parameters`"
            )

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""'Model' abstract base class defining an API to interface framework-specific models."""

from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict, Generic, Type, TypeVar, List

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.logger import logger


# Generic type variables for annotations: specify types that are abstract
# at this level, but have to be coherent when defined by children classes.
_MT = TypeVar("_MT")  # model type
DT = TypeVar("DT")  # data array type


class Model(Generic[_MT, DT], metaclass=ABCMeta):
    """Model abstraction, that wraps and handles both native models

    Attributes:
        model: native model, written in a framework supported by Fed-BioMed.
    """

    _model_type: ClassVar[Type[Any]]

    def __init__(self, model: _MT):
        """Constructor of Model abstract class

        Args:
            model: native model wrapped, of child-class-specific type.
        """
        self._validate_model_type(model)
        self.model: Any = model

    def set_model(self, model: _MT) -> None:
        """Replace the wrapped model with a new one.

        Args:
            model: New model instance that needs assignment as the `model`
                attribute.
        """
        self._validate_model_type(model)
        self.model = model

    def _validate_model_type(self, model: _MT) -> None:
        if not isinstance(model, self._model_type):
            err_msg = (
                f"{ErrorNumbers.FB622.value}: unproper 'model' input type: "
                f"expected '{self._model_type}', but 'got {type(model)}'."
            )
            logger.critical(err_msg)
            raise FedbiomedModelError(err_msg)

    @abstractmethod
    def init_training(self):
        """Initialize parameters before model training."""

    @abstractmethod
    def train(self, inputs: Any, targets: Any, **kwargs) -> None:
        """Perform a training step given inputs and targets data.

        !!! warning "Warning"
            Please run `init_training` method before running `train` method,
            so to initialize parameters needed for model training"

        !!! warning "Warning"
            This function usually does not update weights. You need to call
            `apply_updates` to ensure updates are applied to the model.

        Args:
            inputs: input (training) data.
            targets: target values.
        """

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Return model predictions given input values.

        Args:
            inputs: input values.

        Returns:
            Any: predictions.
        """

    @abstractmethod
    def apply_updates(self, updates: Dict[str, DT]):
        """Applies updates to the model.

        Args:
            updates: model updates.
        """

    @abstractmethod
    def get_weights(self, only_trainable: bool = False, exclude_buffers: bool = True) -> Dict[str, DT]:
        """Return a copy of the model's trainable weights.

        Args:
            only_trainable: Whether to ignore non-trainable model parameters
                from outputs (e.g. frozen neural network layers' parameters),
                or include all model parameters (the default).
            exclude_buffers: Whether to ignore buffers (the default), or 
                include them.

        Returns:
            Model weights, as a dict mapping parameters' names to their value.
        """

    @abstractmethod
    def set_weights(self, weights: Dict[str, DT]) -> None:
        """Assign new values to the model's trainable weights.

        Args:
            weights: Model weights, as a dict mapping parameters' names
                to their value.
        """

    @abstractmethod
    def get_gradients(self) -> Dict[str, DT]:
        """Return computed gradients attached to the model.

        Returns:
            Gradients, as a dict mapping parameters' names to their
                gradient's value.
        """

    @abstractmethod
    def flatten(self,
                only_trainable: bool = False,
                exclude_buffers: bool = True) -> List[float]:
        """Flattens model weights

        Args:
            only_trainable: Whether to ignore non-trainable model parameters
                from outputs (e.g. frozen neural network layers' parameters),
                or include all model parameters (the default).
            exclude_buffers: Whether to ignore buffers (the default), or 
                include them.

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

    @abstractmethod
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

    @staticmethod
    def _assert_dict_inputs(params: Dict[str, Any]) -> None:
        """Raise a FedbiomedModelError if `params` is not a dict."""
        if not isinstance(params, dict):
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}: Got an object with type "
                f"'{type(params)}' while expecting a dict."
            )

    @abstractmethod
    def unflatten(
            self,
            weights_vector: List[float],
            only_trainable: bool = False,
            exclude_buffers: bool = True
    ) -> None:
        """Revert flatten model weights back model-dict form.

        Args:
            weights_vector: Vectorized model weights to convert dict
            only_trainable: Whether to ignore non-trainable model parameters
                from outputs (e.g. frozen neural network layers' parameters),
                or include all model parameters (the default).
            exclude_buffers: Whether to ignore buffers (the default), or 
                include them.

        Returns:
            Model dictionary
        """

        if not isinstance(weights_vector, list) or not all([isinstance(w, float) for w in weights_vector]):
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622} `weights_vector should be 1D list of float containing flatten model parameters`"
            )

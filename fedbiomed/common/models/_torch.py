# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Torch interfacing Model class."""

from typing import Dict, Iterable, Tuple, Union

import numpy as np
import torch
from declearn.model.torch import TorchVector

from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.models import Model


class TorchModel(Model):
    """PyTorch model wrapper that ease the handling of a pytorch model

    Attributes:
        model: torch.nn.Module. Pytorch model wrapped.
        init_params: OrderedDict. Model initial parameters. Set when calling `init_training`
    """

    _model_type = torch.nn.Module
    model: torch.nn.Module  # merely for the docstring builder

    def __init__(self, model: torch.nn.Module) -> None:
        """Instantiates the wrapper over a torch Module instance."""
        super().__init__(model)
        self.init_params: Dict[str, torch.Tensor] = {}

    def get_gradients(
        self,
        as_vector: bool = False,
    ) -> Union[Dict[str, torch.Tensor], TorchVector]:
        """Return the gradients attached to the model, opt. as a declearn TorchVector.

        Args:
            as_vector: Whether to wrap returned gradients into a declearn Vector.

        Returns:
            Gradients, as a dictionary mapping parameters' names to their gradient's
                torch tensor, or as a declearn TorchVector wrapping such a dict.
        """
        gradients = {
            name: param.grad.detach().clone()
            for name, param in self.model.named_parameters()
            if (param.requires_grad and param.grad is not None)
        }
        if len(gradients) < len(list(self.model.named_parameters())):
            # FIXME: this will be triggered when having some frozen weights even if training was properly conducted
            logger.warning(
                "Warning: can not retrieve all gradients from the model. Are you sure you have "
                "trained the model beforehand?"
            )
        if as_vector:
            return TorchVector(gradients)
        return gradients

    def get_weights(
        self,
        as_vector: bool = False,
        only_trainable: bool = False,
    ) -> Union[Dict[str, torch.Tensor], TorchVector]:
        """Return the model's parameters, optionally as a declearn TorchVector.

        Args:
            only_trainable (bool, optional): whether to gather weights only on trainable layers (ie
                non-frozen layers) or all layers (trainable and frozen). Defaults to False, (trainable and
                frozen ones)
            as_vector: Whether to wrap returned weights into a declearn Vector.

        Returns:
            Model weights, as a dictionary mapping parameters' names to their
                torch tensor, or as a declearn TorchVector wrapping such a dict.
        """
        parameters = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad or not only_trainable
        }
        if as_vector:
            return TorchVector(parameters)
        return parameters

    def apply_updates(
        self,
        updates: Union[TorchVector, Dict[str, torch.Tensor]],
    ) -> None:
        """Apply incoming updates to the wrapped model's parameters.

        Args:
            updates: model updates to be added to the model.
        """
        iterator = self._get_iterator_model_params(updates)
        with torch.no_grad():
            for name, update in iterator:
                param = self.model.get_parameter(name)
                param.add_(update.to(param.device))

    def add_corrections_to_gradients(
        self,
        corrections: Union[TorchVector, Dict[str, torch.Tensor]],
    ) -> None:
        """Adds values to attached gradients in the model

        Args:
            corrections: corrections to be added to model's gradients
        """
        iterator = self._get_iterator_model_params(corrections)
        for name, update in iterator:
            param = self.model.get_parameter(name)
            if param.grad is not None:
                param.grad.add_(update.to(param.grad.device))

    @staticmethod
    def _get_iterator_model_params(
        model_params: Union[Dict[str, torch.Tensor], TorchVector],
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Returns an iterable from model_params, whether it is a
        dictionary or a declearn's TorchVector

        Args:
            model_params: model parameters

        Raises:
            FedbiomedModelError: if argument `model_params` type is neither
                a TorchVector nor a dictionary.

        Returns:
            Iterable[Tuple]: iterable containing model parameters, that returns layer name and its value
        """
        if isinstance(model_params, TorchVector):
            iterator = model_params.coefs.items()
        elif isinstance(model_params, dict):
            iterator = model_params.items()
        else:
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}. Got a {type(model_params)} "
                f"while expecting TorchVector or OrderedDict/Dict"
            )
        return iterator

    def predict(
        self,
        inputs: torch.Tensor,
    ) -> np.ndarray:
        """Computes prediction given input data.

        Args:
            inputs: input data

        Returns:
            Model predictions returned as a numpy array
        """
        self.model.eval()  # pytorch switch for model inference-mode
        with torch.no_grad():
            pred = self.model(inputs)
        return pred.cpu().numpy()

    def send_to_device(
        self,
        device: torch.device,
    ) -> None:
        """Sends model to device

        Args:
            device: device set for using GPU or CPU.
        """
        self.model.to(device)

    def init_training(self) -> None:
        """Initializes and sets attributes before the training.

        Initializes `init_params` as a copy of the initial parameters of the model
        """
        # initial aggregated model parameters
        self.init_params = {
            key: param.data.detach().clone()
            for key, param in self.model.named_parameters()
        }
        self.model.train()  # pytorch switch for training
        self.model.zero_grad()

    def train(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> None:
        # TODO: should we pass loss function here? and do the backward prop?
        if not self.init_params:
            raise FedbiomedModelError(
                f"{ErrorNumbers.FB622.value}. Training has not been initialized, please initialize it beforehand"
            )

    def load(self, filename: str) -> None:
        """Loads model from a file.

        Args:
            filename: path towards the file where the model has been saved.

        Returns:
            model_parameters stored in an OrderedDict.
        """
        # loads model from a file
        params = torch.load(filename)
        self.model.load_state_dict(params)

    def save(self, filename: str) -> None:
        """Saves model into a file.

        Args:
            filename: path to the file, where will be saved the model.
        """
        torch.save(self.model.state_dict(), filename)

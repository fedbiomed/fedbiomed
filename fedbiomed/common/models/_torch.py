# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Torch interfacing Model class."""

from typing import Dict

import numpy as np
import torch

from fedbiomed.common.exceptions import FedbiomedModelError
from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.models import Model


class TorchModel(Model):
    """PyTorch model wrapper that ease the handling of a pytorch model

    Attributes:
        model: torch.nn.Module. Pytorch model wrapped.
        init_params: OrderedDict. Model initial parameters.
            Set when calling `init_training`.
    """

    _model_type = torch.nn.Module
    model: torch.nn.Module  # merely for the docstring builder

    def __init__(self, model: torch.nn.Module) -> None:
        """Instantiates the wrapper over a torch Module instance."""
        super().__init__(model)
        self.init_params: Dict[str, torch.Tensor] = {}

    def get_gradients(
        self,
    ) -> Dict[str, torch.Tensor]:
        """Return the gradients attached to the model.

        Returns:
            Gradients, as a dict mapping parameters' names to their gradient's
                torch tensor.
        """
        gradients = {
            name: param.grad.detach().clone()
            for name, param in self.model.named_parameters()
            if (param.requires_grad and param.grad is not None)
        }
        if len(gradients) < len(list(self.model.named_parameters())):
            # FIXME: this will be triggered when having some frozen weights
            #        even if training was properly conducted
            logger.warning(
                "Warning: can not retrieve all gradients from the model. "
                "Are you sure you have trained the model beforehand?"
            )
        return gradients

    def get_weights(
        self,
        only_trainable: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return the model's parameters.

        Args:
            only_trainable: Whether to ignore non-trainable model parameters
                from outputs (e.g. frozen neural network layers' parameters),
                or include all model parameters (the default).

        Returns:
            Model weights, as a dictionary mapping parameters' names to their
                torch tensor.
        """
        parameters = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad or not only_trainable
        }
        return parameters

    def set_weights(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        """Sets model weights.

        Args:
            weights: Model weights, as a dict mapping parameters' names
                to their torch tensor.
        """
        self._assert_dict_inputs(weights)
        incompatible = self.model.load_state_dict(weights, strict=False)
        if incompatible.missing_keys:
            logger.warning(
                "'TorchModel.set_weights' received inputs that did not cover all"
                "model parameters; missing weights: %s",
                incompatible.missing_keys
            )
        if incompatible.unexpected_keys:
            logger.warning(
                "'TorchModel.set_weights' received inputs with unexpected names: %s",
                incompatible.unexpected_keys
            )

    def apply_updates(
        self,
        updates: Dict[str, torch.Tensor],
    ) -> None:
        """Apply incoming updates to the wrapped model's parameters.

        Args:
            updates: model updates to be added to the model.
        """
        self._assert_dict_inputs(updates)
        with torch.no_grad():
            for name, update in updates.items():
                param = self.model.get_parameter(name)
                param.add_(update.to(param.device))

    def add_corrections_to_gradients(
        self,
        corrections: Dict[str, torch.Tensor],
    ) -> None:
        """Add values to the gradients currently attached to the model.

        Args:
            corrections: corrections to be added to the model's gradients.
        """
        self._assert_dict_inputs(corrections)
        for name, update in corrections.items():
            param = self.model.get_parameter(name)
            if param.grad is not None:
                param.grad.add_(update.to(param.grad.device))

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

        !!! warning "Warning":
            This method uses `torch.save`, which relies on pickle and
            is therefore hard to trust by third-party loading methods.
        """
        torch.save(self.model, filename)

    def _reload(self, filename: str) -> None:
        """Model-class-specific backend to the `reload` method.

        Args:
            filename: path to the file where the model has been exported.

        Returns:
            model: reloaded model instance to be wrapped, that will be type-
                checked as part of the calling `reload` method.
        """
        return torch.load(filename)

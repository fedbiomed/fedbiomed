# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Diffenrential Privacy controller."""

from typing import Dict, Tuple, Union
from fedbiomed.common.optimizers.generic_optimizers import NativeTorchOptimizer

import torch
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.validators import ModuleValidator
from torch.nn import Module
from torch.utils.data import DataLoader

from fedbiomed.common.validator import ValidateError
from fedbiomed.common.training_args import DPArgsValidator
from fedbiomed.common.exceptions import FedbiomedDPControllerError
from fedbiomed.common.constants import ErrorNumbers


class DPController:
    """Controls DP action during training."""

    def __init__(self, dp_args: Union[Dict, None] = None) -> None:
        """Constructs DPController with given model.

        Args:
            dp_args: Arguments for differential privacy
        """
        self._privacy_engine = PrivacyEngine()
        self._dp_args = dp_args or {}
        self._is_active = dp_args is not None
        # Configure/validate dp arguments
        if self._is_active:
            self._configure_dp_args()

    def before_training(self,
                        optimizer: NativeTorchOptimizer,
                        loader: DataLoader) -> Tuple[NativeTorchOptimizer, DPDataLoader]:
        """DP action before starting training.

        Args:
            optimizer: NativeTorchOptimizer for training
            loader: Data loader for training

        Returns:
            Differential privacy applied Optimizer and data loader
        """


        if self._is_active:
            if not isinstance(optimizer.optimizer, torch.optim.Optimizer):
                raise FedbiomedDPControllerError(
                    f"{ErrorNumbers.FB616.value}: "
                    f"Optimizer must be an instance of torch.optim.Optimizer, but got {optimizer}"
                    "\nDeclearn optimizers are not yet compatible with Differential Privacy"
            )
            if not isinstance(loader, DataLoader):
                raise FedbiomedDPControllerError(
                    f"{ErrorNumbers.FB616.value}: "
                    "Data loader must be an instance of torch.utils.data.DataLoader"
                )
            try:
                optimizer._model.model, optimizer.optimizer, loader = self._privacy_engine.make_private(
                    module=optimizer._model.model,
                    optimizer=optimizer.optimizer,
                    data_loader=loader,
                    noise_multiplier=float(self._dp_args['sigma']),
                    max_grad_norm=float(self._dp_args['clip'])
                )
            except Exception as e:
                raise FedbiomedDPControllerError(
                    f"{ErrorNumbers.FB616.value}: "
                    f"Error while running privacy engine: {e}"
                )
        return optimizer, loader

    def after_training(self, params: Dict) -> Dict:
        """DP actions after the training.

        Args:
            params: Contains model parameters after training with DP
        Returns:
            `params` fixed model parameters after applying differential privacy
        """
        if self._is_active:
            params = self._postprocess_dp(params)
        return params

    def _configure_dp_args(self) -> None:
        """Initialize arguments to perform DP training. """
        self._dp_args = DPArgsValidator.populate_with_defaults(
            self._dp_args, only_required=False
        )
        try:
            DPArgsValidator.validate(self._dp_args)
        except ValidateError as e:
            raise FedbiomedDPControllerError(
                f"{ErrorNumbers.FB616.value}: DP arguments are not valid: {e}"
            )
        if self._dp_args['type'] == 'central':
            self._dp_args['sigma_CDP'] = self._dp_args['sigma']
            self._dp_args['sigma'] = 0.

    def validate_and_fix_model(self, model: Module) -> Module:
        """Validate and Fix model to be DP-compliant.

        Args:
            model: An instance of [`Module`][torch.nn.Module]

        Returns:
            Fixed or validated model
        """
        if self._is_active and not ModuleValidator.is_valid(model):
            try:
                model = ModuleValidator.fix(model)
            except Exception as e:
                raise FedbiomedDPControllerError(
                    f"{ErrorNumbers.FB616.value}: "
                    f"Error while making model DP-compliant: {e}"
                )
        return model

    def _assess_budget_locally(self, loader: DataLoader) -> Tuple[float, float]:
        """Computes eps and alpha for budget privacy.

        TODO: This function is not used any where on the node side. For future implementation

        Args:
            loader: Pytorch data loader that is going to be used for training

        Returns:
            eps: Calculated epsilon value for privacy budget
            alpha: Calculated epsilon alpha for privacy budget
        """
        # To be used by the nodes to assess budget locally
        eps, alpha = self._privacy_engine.accountant.get_privacy_spent(delta=.1 / len(loader))
        return eps, alpha

    def _postprocess_dp(self, params: Dict) -> Dict:
        """Postprocess of model's parameters after training with DP.

        **Postprocess of DP parameters implies**
        - If central DP is enabled, model's parameters are perturbed according
          to the provided DP parameters.
        - When the Opacus `PrivacyEngine` is attached to the model, parameters'
          names are modified by the addition of `_module.`. This modification
          should be undone before communicating to the server for aggregation.
          This is needed in order to correctly perform download/upload of
          model's parameters in the following rounds

        Args:
            params: Contains model parameters after training with DP
        Returns:
            Contains (post processed) parameters
        """
        # Rename parameters when needed.
        params = {
            key.replace('_module.', ''): param
            for key, param in params.items()
        }
        # When using central DP, postprocess the parameters.
        if self._dp_args['type'] == 'central':
            sigma = self._dp_args['sigma_CDP']
            for key, param in params.items():
                noise = sigma * self._dp_args['clip'] * torch.randn_like(param)
                params[key] = param + noise
        return params

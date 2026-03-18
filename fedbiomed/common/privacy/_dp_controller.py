# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Diffenrential Privacy controller."""

from typing import Dict, Tuple, Union

import torch
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.validators import ModuleValidator
from torch.nn import Module
from torch.utils.data import DataLoader

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedDPControllerError
from fedbiomed.common.logger import logger
from fedbiomed.common.optimizers.generic_optimizers import NativeTorchOptimizer
from fedbiomed.common.training_args import DPArgsValidator
from fedbiomed.common.validator import ValidateError


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
        logger.debug(
            "Initializing DP controller: active=%s provided_args=%s",
            self._is_active,
            sorted(self._dp_args.keys()),
        )
        # Configure/validate dp arguments
        if self._is_active:
            self._configure_dp_args()

    def before_training(
        self, optimizer: NativeTorchOptimizer, loader: DataLoader
    ) -> Tuple[NativeTorchOptimizer, DPDataLoader]:
        """DP action before starting training.

        Args:
            optimizer: NativeTorchOptimizer for training
            loader: Data loader for training

        Returns:
            Differential privacy applied Optimizer and data loader
        """
        # Before training is called once per node per round.
        # The effects remain throughout the batch steps inside that round
        #
        # dp_type=local: Per-batch dp, inside the local training loop of a single round
        # This is done per batch thanks to the Opacus `PrivacyEngine` `make_private` function that is attached to the model, optimizer and data loader.
        #
        # dp_type=central: Noise is added once after training, in the `after_training` function.
        # before_training() still runs once at the start of the round, but with sigma=0.0 after normalization, so the training-time noise path is disabled, and it only does clipping.

        if self._is_active:
            logger.debug(
                "Applying DP before training: optimizer_type=%s loader_type=%s dp_type=%s",
                type(optimizer.optimizer).__name__,
                type(loader).__name__,
                self._dp_args.get("type"),
            )
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
                optimizer._model.model, optimizer.optimizer, loader = (
                    self._privacy_engine.make_private(
                        module=optimizer._model.model,
                        optimizer=optimizer.optimizer,
                        data_loader=loader,
                        noise_multiplier=float(self._dp_args["sigma"]),
                        max_grad_norm=float(self._dp_args["clip"]),
                    )
                )
                logger.debug(
                    "DP privacy engine attached successfully: dp_type=%s loader_type=%s",
                    self._dp_args.get("type"),
                    type(loader).__name__,
                )
            except Exception as e:
                raise FedbiomedDPControllerError(
                    f"{ErrorNumbers.FB616.value}: "
                    f"Error while running privacy engine: {e}"
                ) from e
        return optimizer, loader

    def after_training(self, params: Dict) -> Dict:
        """DP actions after the training.

        Args:
            params: Contains model parameters after training with DP
        Returns:
            `params` fixed model parameters after applying differential privacy
        """
        if self._is_active:
            logger.debug(
                "Applying DP after training: dp_type=%s parameter_count=%d",
                self._dp_args.get("type"),
                len(params),
            )
            params = self._postprocess_dp(params)
        return params

    def _configure_dp_args(self) -> None:
        """Initialize arguments to perform DP training."""
        self._dp_args = DPArgsValidator.populate_with_defaults(
            self._dp_args, only_required=False
        )
        try:
            DPArgsValidator.validate(self._dp_args)
        except ValidateError as e:
            raise FedbiomedDPControllerError(
                f"{ErrorNumbers.FB616.value}: DP arguments are not valid: {e}"
            ) from e
        logger.debug(
            "Validated DP arguments: dp_type=%s clip=%s sigma_set=%s",
            self._dp_args.get("type"),
            self._dp_args.get("clip"),
            "sigma" in self._dp_args,
        )
        if self._dp_args["type"] == "central":
            self._dp_args["sigma_CDP"] = self._dp_args["sigma"]
            self._dp_args["sigma"] = 0.0
            logger.debug("Configured central DP arguments: sigma moved to sigma_CDP")

    def validate_and_fix_model(self, model: Module) -> Module:
        """Validate and Fix model to be DP-compliant.

        Args:
            model: An instance of [`Module`][torch.nn.Module]

        Returns:
            Fixed or validated model
        """
        if self._is_active and not ModuleValidator.is_valid(model):
            logger.debug("DP model validation failed, attempting automatic fix")
            try:
                model = ModuleValidator.fix(model)
            except Exception as e:
                raise FedbiomedDPControllerError(
                    f"{ErrorNumbers.FB616.value}: "
                    f"Error while making model DP-compliant: {e}"
                ) from e
            logger.debug("DP model automatic fix applied successfully")
        elif self._is_active:
            logger.debug("DP model validation passed without modification")
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
        eps, alpha = self._privacy_engine.accountant.get_privacy_spent(
            delta=0.1 / len(loader)
        )
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
        # dp_type=central: Noise is added here once after training.
        # dp_type=local: Noise is added per batch during training.
        logger.debug(
            "Postprocessing DP parameters: dp_type=%s parameter_count=%d",
            self._dp_args.get("type"),
            len(params),
        )
        # Rename parameters when needed.
        params = {key.replace("_module.", ""): param for key, param in params.items()}
        # When using central DP, postprocess the parameters.
        if self._dp_args["type"] == "central":
            logger.debug("Applying central DP noise during postprocessing")
            sigma = self._dp_args["sigma_CDP"]
            for key, param in params.items():
                noise = sigma * self._dp_args["clip"] * torch.randn_like(param)
                params[key] = param + noise
        return params

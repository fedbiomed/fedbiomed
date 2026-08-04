# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedExperimentError


class FedCombatModelWrapper(nn.Module):
    """
    A wrapper for the Fed-ComBat model to automatically handle the biases
    """

    def __init__(
        self,
        biological_model: nn.Module,
        bias_model: nn.Module,
    ):
        """Constructor of the wrapper

        Args:
            biological_model: PyTorch model representing the biological effects in Fed-ComBat.
            bias_model: PyTorch model representing the bias effects in Fed-ComBat.

        Raises:
            FedbiomedExperimentError: if the provided biological model contains bias parameters."""
        super().__init__()
        self.biological_model = biological_model
        self.bias_model = bias_model

        if not self._check_model_no_bias():
            raise FedbiomedExperimentError(
                f"{ErrorNumbers.FB420.value}: Fed-ComBat harmonization model initialization failed. "
                "Provided biological model for Fed-ComBat contains bias. "
                "This will result in a biased harmonization. "
                "Please provide a model without bias parameters."
            )

    def forward(self, x):
        biological_effects = self.biological_model(x)
        bias_column = x.new_ones((x.shape[0], 1))
        bias = self.bias_model(bias_column)
        return biological_effects + bias

    def _check_model_no_bias(self) -> bool:
        """Tests whether the given model has any trainable bias parameters.

        Returns:
            True if the model doesn't have a bias, False if it does.
        """
        # Check common module types that can carry an explicit bias parameter.
        for module in self.biological_model.modules():
            if isinstance(
                module,
                (
                    nn.Linear,
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                ),
            ):
                # If bias is not None, the layer uses a bias parameter.
                if getattr(module, "bias", None) is not None:
                    return False

        # Fallback: look for any parameter whose name suggests it is a bias.
        for name, param in self.biological_model.named_parameters():
            if "bias" in name and param is not None:
                return False

        return True

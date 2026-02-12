import torch
import torch.nn as nn

from fedbiomed.common.logger import logger


class FedComBatModelWrapper(nn.Module):
    """
    A wrapper for the Fed-ComBat model to automatically handle the biases
    """

    def __init__(
        self,
        biological_model: nn.Module,
        n_covariates: int,
        n_phenotypes: int,
    ):
        super().__init__()
        self.n_phenotypes = n_phenotypes
        self.n_covariates = n_covariates
        self.biological_model = biological_model
        # TODO: warning does not work. The message should be sent to the researcher to raise the warning
        if not self._check_model_no_bias():
            logger.warning(
                "Provided biological model for Fed-ComBat contains bias. "
                "This will result in a biased harmonization. "
                "Please provide a model without bias parameters"
            )

        self.local_bias = nn.Linear(1, self.n_phenotypes)

    def forward(self, x):
        biological_effects = self.biological_model(x)
        bias_column = torch.ones((x.shape[0], 1))
        bias = self.local_bias(bias_column)
        return biological_effects + bias

    def _check_model_no_bias(self) -> bool:
        """Tests whether the given model has a bias parameter.

        Returns:
            True if the model doesn't have a bias, False if it does
        """
        # This is not a bulletproof way to test whether there is a bias or not.
        with torch.no_grad():
            zero_covariates = torch.zeros((1, self.n_covariates))
            preds_zeros = self.biological_model(zero_covariates)
            return torch.all(preds_zeros == 0)

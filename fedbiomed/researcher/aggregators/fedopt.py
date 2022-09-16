"""
"""

from typing import Dict

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging


class FedOpt(Aggregator):
    """
    Defines the FedOpt strategy
    """

    def __init__(self, strategy: str = "FedAdam", server_lr: float = 1e-2, beta1: float = 0.9, beta2: float = 0.999, tau: float = 1e-3):
        """Construct `FedOpt` object as an instance of [`Aggregator`]
        [fedbiomed.researcher.aggregators.Aggregator].

        Args:
            strategy: specific aggregation strategy from fed opt. Defaults is FedAdam.
            tau: adaptivity hyperparameter for the Adam/Yogi optimizer. Defaults to 1e-3.
            server_learning_rate : The learning rate used by the server optimizer. Defaults to 1.
            beta1: between 0 and 1, momentum parameter. Defaults to 0.9.
            beta2: between 0 and 1, second moment parameter. Defaults to 0.999.
        """
        assert strategy in ["FedAdam", "FedAdagrad", "FedYogi"], "One strategy among FedAdam, FedAdagrad, FedYogi must be set"
        super(FedOpt, self).__init__()
        self.aggregator_name = strategy
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau

    def aggregate(self, model_params: list, weights: list) -> Dict:
        """ Aggregates  local models sent by participating nodes into a global model, following Federated Averaging
        strategy.

        Args:
            model_params: contains each model layers
            weights: contains all weights of a given layer.

        Returns:
            Aggregated parameters
        """
        weights = self.normalize_weights(weights)
        return federated_averaging(model_params, weights)

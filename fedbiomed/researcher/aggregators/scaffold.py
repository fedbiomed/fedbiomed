"""
"""

from typing import Dict

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging


class Scaffold(Aggregator):
    """
    Defines the Scaffold strategy
    """

    def __init__(self, server_lr):
        """Construct `Scaffold` object as an instance of [`Aggregator`]
        [fedbiomed.researcher.aggregators.Aggregator].
        """
        super(Scaffold, self).__init__()
        self.aggregator_name = "Scaffold"
        self.server_lr = server_lr

    def aggregate(self, model_params: list, weights: list) -> Dict:
        """ Aggregates local models sent by participating nodes into a global model, using Federated Averaging
        also present in Scaffold for the aggregation step.

        Args:
            model_params: contains each model layers
            weights: contains all weights of a given layer.

        Returns:
            Aggregated parameters
        """
        weights = self.normalize_weights(weights)
        return federated_averaging(model_params, weights)

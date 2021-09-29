from typing import Dict

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging


class FedAverage(Aggregator):
    """ Defines the Federated averaging strategy """

    def __init__(self):
        super(FedAverage, self).__init__()

    def aggregate(self, model_params: list, weights: list) -> Dict:
        """aggregates  local models sent by participating nodes into
        a global model, following Federated Averaging strategy.

        Args:
            model_params (list): contains each model layers
            weights (list): contains all weigths of a given
            layer.

        Returns:
            Dict: [description]
        """
        weights = self.normalize_weights(weights)
        return federated_averaging(model_params, weights)

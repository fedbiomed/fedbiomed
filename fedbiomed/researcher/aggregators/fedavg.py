"""
"""

from typing import Dict

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging, perturb


class FedAverage(Aggregator):
    """
    Defines the Federated averaging strategy
    """

    def __init__(self, Central_DP_params = None):
        """
        constructor
        """
        super(FedAverage, self).__init__()
        self.aggregator_name = "FedAverage"

        self.Central_DP_params = {}
        if Central_DP_params is not None:
            if not isinstance(Central_DP_params['sigma'], float):
                raise TypeError("FedAvg noise parameter sigma must be float")
            else:
                self.Central_DP_params['sigma'] = Central_DP_params['sigma']

            if not isinstance(Central_DP_params['clip_threshold'], float):
                raise TypeError("FedAvg clip_threshold parameter must be float")
            else:
                self.Central_DP_params['clip_threshold'] = Central_DP_params['clip_threshold']
        

    def aggregate(self, model_params: list, weights: list) -> Dict:
        """
        aggregates  local models sent by participating nodes into
        a global model, following Federated Averaging strategy.

        Args:
            model_params (list): contains each model layers
            weights (list): contains all weigths of a given
            layer.
            sigma (float): sigma for global Gaussian DP mechanism

        Returns:
            Dict: [description]
        """
        weights = self.normalize_weights(weights)

        avg = federated_averaging(model_params, weights)

        if self.Central_DP_params:
           avg = perturb(avg, self.Central_DP_params)
        return avg

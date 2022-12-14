# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
"""

from typing import Dict

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging


class FedAverage(Aggregator):
    """
    Defines the Federated averaging strategy
    """

    def __init__(self):
        """Construct `FedAverage` object as an instance of [`Aggregator`]
        [fedbiomed.researcher.aggregators.Aggregator].
        """
        super(FedAverage, self).__init__()
        self.aggregator_name = "FedAverage"

    def aggregate(self, model_params: list, weights: list, *args, **kwargs) -> Dict:
        """ Aggregates  local models sent by participating nodes into a global model, following Federated Averaging
        strategy.

        Args:
            model_params: contains each model layers
            weights: contains all weights of a given layer.

        Returns:
            Aggregated parameters
        """
        model_params_processed = [list(model_param.values())[0] for model_param in model_params] # model params are contained in a dictionary with node_id as key, we just retrieve the params
        weights_processed = [weight if isinstance(weight, float) else list(weight.values())[0] for weight in weights]
        weights_processed = self.normalize_weights(weights_processed)

        return federated_averaging(model_params_processed, weights_processed)

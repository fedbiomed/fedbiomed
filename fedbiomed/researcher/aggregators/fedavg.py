# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
"""

from typing import Dict

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedAggregatorError
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

        weights is a list of single-item dictionaries, each dictionary has the node id as key, and the weight as value.
        model_params is a list of single-item dictionaries, each dictionary has the node is as key,
        and a framework-specific representation of the model parameters as value.

        Args:
            model_params: contains each model layers
            weights: contains all weights of a given layer.

        Returns:
            Aggregated parameters
        """
        model_params_processed = list()
        weights_processed = list()
        for model_param in model_params:
            node_id_list = list(model_param.keys())
            if len(node_id_list) != 1:
                msg = f'{ErrorNumbers.FB401.value}. Unexpected format for model parameters. '\
                      f'Expected a dictionary with exactly 1 key, instead got {len(node_id_list)} keys.' \
                      f'The full list of keys in model parameters is {[list(m.keys()) for m in model_params]}'
                raise FedbiomedAggregatorError(msg)
            node_id = node_id_list[0]  # guaranteed to be the one and only element of the list
            model_params_processed.append(list(model_param.values())[0])
            # we are reordering the model weights so list of parameters
            # matches list of weights
            weight = self.get_weights_from_node_id(node_id, weights)
            weights_processed.append(weight)
        weights_processed = self.normalize_weights(weights_processed)

        return federated_averaging(model_params_processed, weights_processed)

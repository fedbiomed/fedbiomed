# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
"""

from typing import Dict, Union, Mapping

import torch # used by typing
import numpy # used by typing

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging, federated_standardization


class FedAverage_FedStd(Aggregator):
    """
    Defines the Federated averaging strategy
    """

    def __init__(self):
        """Construct `FedAverage` object as an instance of [`Aggregator`]
        [fedbiomed.researcher.aggregators.Aggregator].
        """
        super(FedAverage_FedStd, self).__init__()
        self.aggregator_name = "FedAverage_FedStd"

    def aggregate(
            self,
            model_params: Dict[str, Dict[str, Union['torch.Tensor', 'numpy.ndarray']]],
            weights: Dict[str, float],
            *args,
            **kwargs
    ) -> Mapping[str, Union['torch.Tensor', 'numpy.ndarray']]:
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

        model_params_processed = []
        mean_std_processed = []
        weights_processed = []

        for node_id, params in model_params.items():

            if node_id not in weights:
                raise FedbiomedAggregatorError(
                    f"{ErrorNumbers.FB401.value}. Can not find corresponding calculated weight for the "
                    f"node {node_id}. Aggregation is aborted."
                )

            weight = weights[node_id]
            weights_processed.append(weight)

            keys_fedstd = ['mean','std','size']
            mean_std_subdict = {k:params[k] for k in keys_fedstd if set(keys_fedstd).issubset(params)}
            mean_std_processed.append(mean_std_subdict)
            for key in keys_fedstd:
                params.pop(key, None)
            model_params_processed.append(params)

        if any([x < 0. or x > 1. for x in weights_processed]) or sum(weights_processed) == 0:
            raise FedbiomedAggregatorError(
                f"{ErrorNumbers.FB401.value}. Aggregation aborted due to sum of the weights is equal to 0 {weights}. "
                f"Sample sizes received from nodes might be corrupted."
            )

        agg_params = federated_averaging(model_params_processed, weights_processed)
        global_mean_std = federated_standardization(mean_std_processed, weights)
        agg_params.update(global_mean_std)

        return agg_params

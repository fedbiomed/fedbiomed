import copy
from typing import Dict, List

import torch


def federated_averaging(model_params: List[Dict[str, torch.Tensor]],
                        weights: List[float]) -> Dict[str, torch.Tensor]:
    """Defines Federated Averaging (FedAvg) strategy for model
    aggregation.

    Args:
        model_params (List[Dict[str, torch.Tensor]]): list that contains nodes'
        model parameters: each model is stored as an OrderedDict
        (maps model layer name to the model weights)
        weights (List[float]): weights for performing weighted sum
        in FedAvg strategy (depneding on the dataset size of each node).
        Items in the list must always sum up to 1

    Returns:
        Dict[str, torch.Tensor]: final model with aggregated layers, 
        as an OrderedDict object.
    """
    assert len(model_params) > 0, 'An empty list of models was passed.'
    assert len(weights) == len(model_params), 'List with number of observations must have ' \
                                              'the same number of elements that list of models.'

    # Compute proportions
    proportions = [n_k / sum(weights) for n_k in weights]

    # Empty model parameter dictionary
    avg_params = copy.deepcopy(model_params[0])
    for key, val in avg_params.items():
        avg_params[key] = torch.zeros_like(val)

    # Compute average
    for model, weight in zip(model_params, proportions):
        for key in avg_params.keys():
            avg_params[key] += weight * model[key]

    return avg_params

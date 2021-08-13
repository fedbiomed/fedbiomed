import copy
from typing import Dict, List

import torch


def federated_averaging(model_params: List[Dict], weights: List) -> Dict:
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

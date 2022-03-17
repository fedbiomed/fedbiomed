"""
???

should be defined in fedagv.py

???
"""

import copy
from typing import Dict, List

import torch
import numpy as np


def initialize(val):
    """
    ???
    """

    if isinstance(val, torch.Tensor):
        return ('tensor' , torch.zeros_like(val).float())
    elif isinstance(val, np.ndarray) or isinstance(val, list):
        return ('array' , np.zeros(len(val), dtype = float))


def federated_averaging(model_params: List[Dict[str, torch.Tensor]],
                        weights: List[float]) -> Dict[str, torch.Tensor]:
    """
    Defines Federated Averaging (FedAvg) strategy for model
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
        (t, avg_params[key] ) = initialize(val)
    if t == 'tensor':
        for model, weight in zip(model_params, proportions):
            for key in avg_params.keys():
                avg_params[key] += weight * model[key]

    if t == 'array':
        for key in avg_params.keys():
            matr = np.array([ d[key] for d in model_params ])
            avg_params[key] = np.average(matr, weights=np.array(weights), axis=0)

    return avg_params

def perturb(model_params: List[Dict[str, torch.Tensor]], DP_params: Dict[float, float]) -> Dict[str, torch.Tensor]:

    perturbed_params = {}

    per_param_norms = [params.view(len(params), -1).norm(2,dim=-1) for key, params in model_params.items()]
    per_sample_norms = torch.cat(per_param_norms,dim=0).norm(2)#torch.stack(per_param_norms, dim=1).norm(2,dim=1)
    per_sample_clip_factor = (DP_params['clip_threshold'] / (per_sample_norms + 1e-6)).clamp(max=1.0)
    for key in model_params.keys():
        perturbed_params[key] = model_params[key].mul(per_sample_clip_factor) \
                        + torch.sqrt(torch.tensor([2]))*DP_params['sigma'] * torch.randn_like(model_params[key])
    return perturbed_params


import copy
from typing import Dict, List
import torch
import numpy as np

def initialize(val):
    if isinstance(val, torch.Tensor):
        return ('tensor' , torch.zeros_like(val) )
    elif isinstance(val, np.ndarray) or isinstance(val, list):
        return ('array' , np.zeros(len(val)))

def federated_averaging(model_params: List[Dict], weights: List) -> Dict:
    assert len(model_params) > 0, 'An empty list of models was passed.'
    assert len(weights) == len(model_params), 'List with number of observations must have ' \
                                              'the same number of elements that list of models.'

    # Compute proportions
    proportions = [n_k / sum(weights) for n_k in weights]

    # Empty model parameter dictionary
    avg_params = copy.deepcopy(model_params[0])
    #print('before for ',model_params)
    for key, val in avg_params.items():
        (t, avg_params[key] ) = initialize(val)
    if t == 'tensor':
        for model, weight in zip(model_params, proportions):
            for key in avg_params.keys():
                avg_params[key] += weight * model[key]

    if t == 'array':
        for key in avg_params.keys():
            matr = np.array([ d[key] for d in model_params ])
            avg_params[key] = np.average(matr,weights=np.array(weights),axis=0)

    #print('after for',avg_params)
    return avg_params

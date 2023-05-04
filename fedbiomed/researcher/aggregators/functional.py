# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Dict, List, Mapping, Tuple, Union

import torch
import numpy as np


def initialize(val: Union[torch.Tensor, np.ndarray]) -> Tuple[str, Union[torch.Tensor, np.ndarray]]:
    """Initialize tensor or array vector. """
    if isinstance(val, torch.Tensor):
        return 'tensor', torch.zeros_like(val).float()

    if isinstance(val, (list, np.ndarray)):
        val = np.array(val)
        return 'array', np.zeros(val.shape, dtype = float)


def federated_averaging(model_params: List[Dict[str, Union[torch.Tensor, np.ndarray]]],
                        weights: List[float]) -> Mapping[str, Union[torch.Tensor, np.ndarray]]:
    """Defines Federated Averaging (FedAvg) strategy for model aggregation.

    Args:
        model_params: list that contains nodes' model parameters; each model is stored as an OrderedDict (maps
            model layer name to the model weights)
        weights: weights for performing weighted sum in FedAvg strategy (depending on the dataset size of each node).
            Items in the list must always sum up to 1

    Returns:
        Final model with aggregated layers, as an OrderedDict object.
    """
    assert len(model_params) > 0, 'An empty list of models was passed.'
    assert len(weights) == len(model_params), 'List with number of observations must have ' \
                                              'the same number of elements that list of models.'

    # Compute proportions
    proportions = [n_k / sum(weights) for n_k in weights]
    return weighted_sum(model_params, proportions)


def weighted_sum(model_params: List[Dict[str, Union[torch.Tensor, np.ndarray]]],
                 proportions: List[float]) -> Mapping[str, Union[torch.Tensor, np.ndarray]]:
    """Performs weighted sum operation

    Args:
        model_params (List[Dict[str, Union[torch.Tensor, np.ndarray]]]): list that contains nodes'
            model parameters; each model is stored as an OrderedDict (maps model layer name to the model weights)
        proportions (List[float]): weights of all items whithin model_params's list

    Returns:
        Mapping[str, Union[torch.Tensor, np.ndarray]]: model resulting from the weigthed sum 
                                                       operation
    """
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
            avg_params[key] = np.average(matr, weights=np.array(proportions), axis=0)

    return avg_params

def federated_standardization(model_params: List[Dict[str, torch.Tensor]],
                        weights: List[float]) -> Dict[str, torch.Tensor]:
    """Defines strategy for evaluating the global mean and standard deviation.

    Args:
        model_params: list that contains nodes' model parameters; each model is stored as an OrderedDict (maps
            model layer name to the model weights). Expected model parameters are:
            - mean, a torch.Tensor containing the local mean (expected size = num features)
            - std, a torch.Tensor containing the local std (expected size = num features)
            - size, a torch.Tensor containing the number of locally observed samples per feature (expected size = num features)

        weights: not needed here

    Returns:
        A dictionary containing federated mean, federated standard deviation and total number of samples.
    """
    assert len(model_params) > 0, 'An empty list of models was passed.'
    assert len(weights) == len(model_params), 'List with number of observations must have ' \
                                              'the same number of elements that list of models.'

    # Recover lists of local means and local stds
    mean_cl = [mod_par['mean'] for mod_par in model_params]
    std_cl = [mod_par['std'] for mod_par in model_params]
    N_cl = [mod_par['size'] for mod_par in model_params]

    #Evaluate global mean and global std
    cl = len(N_cl)
    N_tot = sum([N_cl[c] for c in range(cl)])
    fed_mean = sum([N_cl[i]*mean_cl[i]/N_tot for i in range(cl)])
    fed_std = torch.sqrt(sum([((N_cl[i]-1)*(std_cl[i]**2)+N_cl[i]*(mean_cl[i]**2))/(N_tot-cl) for i in range(cl)])-(N_tot/(N_tot-cl))*(fed_mean**2))

    # with np
    #fed_mean = sum([N_cl[i]*np.array(mean_cl[i])/N_tot for i in range(cl)])
    #fed_std = np.sqrt(sum([((N_cl[i]-1)*np.array(std_cl[i])**2+\
    #                            N_cl[i]*np.array(mean_cl[i])**2)/(N_tot-cl) for i in range(cl)])\
    #                        -(N_tot/(N_tot-cl))*fed_mean**2)

    fed_standardization_params = {'mean': fed_mean, 'std': fed_std, 'size': N_tot}

    return fed_standardization_params

def init_correction_states(model_params: Dict, node_ids: Dict) -> Dict:
    init_params = {key: initialize(tensor)[1] for key, tensor in model_params.items()}
    client_correction = {node_id: copy.deepcopy(init_params) for node_id in node_ids}
    return client_correction

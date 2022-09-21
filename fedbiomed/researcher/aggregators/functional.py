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
    """Initialize tensor or array vector. """

    if isinstance(val, torch.Tensor):
        return ('tensor' , torch.zeros_like(val).float())
    elif isinstance(val, np.ndarray) or isinstance(val, list):
        return ('array' , np.zeros(len(val), dtype = float))


def federated_averaging(model_params: List[Dict[str, torch.Tensor]],
                        weights: List[float]) -> Dict[str, torch.Tensor]:
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


def update_momentum_fedopt(strategy_info: Dict[str, str], momentum: Dict[str, torch.Tensor], delta_aggregated_params: Dict[str, torch.Tensor]):
    """ Update the momentum associated to each model parameter.

    Args:
        strategy_info: contains information about the chosen strategy (name, beta1, beta2, tau, server lr)
        momentum: momentum simply adds a fraction m of the previous weight update to the current one
                  each parameter of the prediction model has a corresponding tensor with momentum
        delta_aggregated_params: contains prediction model parameters and corresponding aggregated weights

    Returns:
        dictionary containing momentum
    """
    strategy = strategy_info['strategy']
    beta1 = strategy_info['beta1']
    if strategy in ["FedAdam", "FedAdagrad", "FedYogi"]:
        # Update momentum
        for param in momentum:
            momentum[param] = (
                beta1 * momentum[param]
                + (1 - beta1) * delta_aggregated_params[param]
            )
    return momentum


def update_second_moment_fedopt(strategy_info: Dict[str, str], second_moment: Dict[str, torch.Tensor], delta_aggregated_params: Dict[str, torch.Tensor]):
    """ Update the second moment associated to each model parameter.

    Args:
        strategy_info: contains information about the chosen strategy (name, beta1, beta2, tau, server lr)
        second_moment: term that helps to give different learning rates for different parameters
                       second moment is uncentered variance (meaning we don’t subtract the mean during variance calculation)
        delta_aggregated_params: contains prediction model parameters and corresponding aggregated weights
    
    Returns:
        dictionary containing second moments
    """
    strategy = strategy_info['strategy']
    beta2 = strategy_info['beta2']
    # Update second moment
    if strategy == "FedAdam":
        for param in second_moment:
            second_moment[param] = (
                beta2 * second_moment[param]
                + (1 - beta2)
                * delta_aggregated_params[param]
                * delta_aggregated_params[param]
            )
    elif strategy == "FedAdagrad":
        for param in second_moment:
            second_moment[param] = (
                second_moment[param]
                + delta_aggregated_params[param]
                * delta_aggregated_params[param]
            )
    elif strategy == "FedYogi":
        for param in second_moment:
            sign = torch.sign(
                second_moment[param]
                - delta_aggregated_params[param]
                * delta_aggregated_params[param]
            )
            second_moment[param] = (
                second_moment[param]
                - (1 - beta2)
                * delta_aggregated_params[param]
                * delta_aggregated_params[param]
                * sign
            )
    return second_moment


def calculate_param_updates_fedopt(strategy_info: Dict[str, str], updates: Dict[str, torch.Tensor], momentum: Dict[str, torch.Tensor], second_moment: Dict[str, torch.Tensor], tau_array: Dict[str, torch.Tensor]):
    """ Calculate the updates associated to each model parameter. These updates will be added to the initial server state,
    giving us the new server state at the end of the round.

    Args:
        strategy_info: contains information about the chosen strategy (name, beta1, beta2, tau, server lr)
        momentum: momentum simply adds a fraction m of the previous weight update to the current one
                  each parameter of the prediction model has a corresponding tensor with momentum
        second_moment: term that helps to give different learning rates for different parameters
                       second moment is uncentered variance (meaning we don’t subtract the mean during variance calculation)
        tau_array: adaptivity hyperparameter associated to each model parameter

    Returns:
        dictionary containing updates
    """
    strategy = strategy_info['strategy']
    server_lr = strategy_info['server_lr']
    if strategy in ["FedAdam", "FedAdagrad", "FedYogi"]:
        for param in updates:
            updates[param] = (
                server_lr
                * momentum[param]
                / (torch.sqrt(second_moment[param]) + tau_array[param])
            )
    return updates
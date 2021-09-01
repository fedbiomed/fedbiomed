from typing import Dict

import torch

from fedbiomed.researcher.aggregators.functional import federated_averaging


class Aggregator:
    """
    Defines methods for aggregating strategy (eg FedAvg, FedProx, SCAFFOLD, ...).
    """
    def __init__(self):
        pass

    @staticmethod
    def normalize_weights(weights) -> list:
        # Load list of weights assigned to each client and
        # normalize these weights so they sum up to 1
        weights = torch.tensor(weights).float()
        weights /= weights.sum()

        return weights

    def aggregate(self,  model_params: list, weights: list) -> Dict: # pragma: no cover
        """Strategy to aggregate models"""
        pass

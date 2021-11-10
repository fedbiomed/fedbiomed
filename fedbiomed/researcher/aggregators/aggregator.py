from typing import Dict, Any

import torch

from fedbiomed.researcher.aggregators.functional import federated_averaging


class Aggregator:
    """
    Defines methods for aggregating strategy
    (eg FedAvg, FedProx, SCAFFOLD, ...).
    """
    def __init__(self):
        pass

    @staticmethod
    def normalize_weights(weights) -> list:
        # Load list of weights assigned to each node and
        # normalize these weights so they sum up to 1
        norm = [w/sum(weights) for w in weights]
        return norm

    def aggregate(self,  model_params: list, weights: list) -> Dict: # pragma: no cover
        """Strategy to aggregate models"""
        pass

    def save_state(self) -> Dict[str, Any]:
        return None

    def load_state(self):
        pass

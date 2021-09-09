from typing import Dict

import torch

from fedbiomed.researcher.aggregators.functional import federated_averaging


class Aggregator:
    def __init__(self):
        pass

    @staticmethod
    def normalize_weights(weights) -> list:
        # Load list of weights assigned to each client and
        # normalize these weights so they sum up to 1
        norm = [w/sum(weights) for w in weights]
        return norm

    def aggregate(self,  model_params: list, weights: list) -> Dict: # pragma: no cover
        """Strategy to aggregate models"""
        pass

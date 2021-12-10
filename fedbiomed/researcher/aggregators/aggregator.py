"""
top class for all aggregators
"""

from typing import Dict, Any

class Aggregator:
    """
    Defines methods for aggregating strategy
    (eg FedAvg, FedProx, SCAFFOLD, ...).
    """
    def __init__(self):
        self.aggregator_params = None

    @staticmethod
    def normalize_weights(weights) -> list:
        """
        Load list of weights assigned to each node and
        normalize these weights so they sum up to 1
        """
        norm = [w/sum(weights) for w in weights]
        return norm

    def aggregate(self,  model_params: list, weights: list) -> Dict: # pragma: no cover
        """Strategy to aggregate models"""

    def save_state(self) -> Dict[str, Any]:
        """
        use for breakpoints. save the aggregator state
        """
        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "parameters": self.aggregator_params
        }
        return state

    def load_state(self, state: Dict[str, Any]=None):
        """
        use for breakpoints. load the aggregator state
        """

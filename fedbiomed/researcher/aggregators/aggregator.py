"""Aggregator abstract base class."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

from declearn.model.api import Vector


class Aggregator(metaclass=ABCMeta):
    """Abstract base class to implement federated updates' aggregation."""

    def __init__(self) -> None:
        """Instantiate the aggregator."""
        self._aggregator_params = None

    @staticmethod
    def normalize_weights(weights: List[float]) -> List[float]:
        """
        Load list of weights assigned to each node and
        normalize these weights so they sum up to 1

        assuming that all values are >= 0.0
        """
        _l = len(weights)
        if _l == 0:
            return []
        _s = sum(weights)
        if _s == 0:
            norm = [ 1.0 / _l ] * _l
        else:
            norm = [_w / _s for _w in weights]
        return norm

    @abstractmethod
    def aggregate(
            self,
            model_params: List[Vector],
            weights: List[float]
        ) -> Vector:
        """Aggregate model parameters.

        Args:
            model_params: List of model parameters received from each node.
            weights: List of node-wise weights.

        Returns:
            params: Aggregated parameters, as a declearn Vector.
        """
        return NotImplemented

    def save_state(self) -> Dict[str, Any]:
        """
        use for breakpoints. save the aggregator state
        """
        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "parameters": self._aggregator_params
        }
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        use for breakpoints. load the aggregator state
        """
        self._aggregator_params = state['parameters']

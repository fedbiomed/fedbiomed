"""Aggregator abstract base class."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Union

from declearn.model.api import Model, Vector
from declearn.optimizer import Optimizer
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.common.logger import logger


class Aggregator(metaclass=ABCMeta):
    """Abstract base class to implement federated updates' aggregation."""

    aggregator_name: str

    def __init__(
        self,
        optim: Union[Optimizer, Dict[str, Any], None] = None,
    ) -> None:
        """Instantiate the aggregator."""
        if isinstance(optim, Dict):
            self.optim = Optimizer(**optim)
        elif isinstance(optim, Optimizer):
            self.optim = optim
        elif not optim:
            self.optim = Optimizer(lrate=1)
        else:
            msg = "optim must be an one of Dict, Optimizer, or None"
            logger.critical(msg)
            raise FedbiomedAggregatorError(msg)

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
            norm = [1.0 / _l] * _l
        else:
            norm = [_w / _s for _w in weights]
        return norm

    @abstractmethod
    def aggregate(
        self,
        global_model: Model,
        local_model_params: List[Vector],
        weights: List[float],
    ) -> Vector:
        """Aggregate local model parameters and update global model.

        Args:
            global_model: Reference Model handled by the researcher, the
                weights from which are to be updated.
            local_model_params: List of model parameters received from each node.
            weights: List of node-wise weights.

        Returns:
            params: Aggregated parameters, as a declearn NumpyVector.
        """
        return NotImplemented

    def save_state(self) -> Dict[str, Any]:
        """
        use for breakpoints. save the aggregator state
        """
        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            # TODO add in Optimizer serialization
        }
        logger.warning(
            "`Aggregator.save_state` implementation is yet to be completed"
        )
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        use for breakpoints. load the aggregator state
        """
        # TODO add in Optimizer deserialization

        logger.warning(
            "`Aggregator.load_state` implementation is yet to be completed"
        )

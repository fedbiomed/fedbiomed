"""
top class for all aggregators
"""


from typing import Dict, Any

from fedbiomed.common.constants  import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.common.logger     import logger


class Aggregator:
    """
    Defines methods for aggregating strategy
    (eg FedAvg, FedProx, SCAFFOLD, ...).
    """
    def __init__(self):
        self._aggregator_params = None

    @staticmethod
    def normalize_weights(weights: list) -> list:
        """
        Load list of weights assigned to each node and
        normalize these weights so they sum up to 1

        assuming that all values are >= 0.0
        """
        print("WEIGTHS", weights)
        _l = len(weights)
        if _l == 0:
            return []
        _s = sum(weights)
        if _s == 0:
            norm = [ 1.0 / _l ] * _l
        else:
            norm = [_w / _s for _w in weights]
        return norm

    def aggregate(self, model_params: list, weights: list, *args, **kwargs) -> Dict:
        """
        Strategy to aggregate models

        Args:
            model_params: List of model parameters received from each node
            weights: Weight for each node-model-parameter set

        Raises:
            FedbiomedAggregatorError: If the method is not defined by inheritor
        """
        msg = ErrorNumbers.FB401.value + \
            ": aggreate method should be overloaded by the choosen strategy"
        logger.critical(msg)
        raise FedbiomedAggregatorError(msg)

    def scaling(self, model_param: dict, *args, **kwargs) -> dict:
        """Should be overwritten by child if a scaling operation is involved in aggregator"""
        return model_param

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

    def load_state(self, state: Dict[str, Any] = None):
        """
        use for breakpoints. load the aggregator state
        """
        self._aggregator_params = state['parameters']

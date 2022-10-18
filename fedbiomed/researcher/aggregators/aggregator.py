"""
top class for all aggregators
"""


import copy
import os
from typing import Dict, Any, List, Optional, Tuple

from fedbiomed.common.constants  import ErrorNumbers, TrainingPlans
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.common.logger     import logger
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.filetools import copy_file


class Aggregator:
    """
    Defines methods for aggregating strategy
    (eg FedAvg, FedProx, SCAFFOLD, ...).
    """
    def __init__(self):
        self._aggregator_args: dict = None
        self._fds: FederatedDataSet = None
        self._training_plan_type: TrainingPlans = None

    @staticmethod
    def normalize_weights(weights: list) -> list:
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
    
    def set_fds(self, fds: FederatedDataSet) -> FederatedDataSet:
        self._fds = fds
        return self._fds
    
    def set_training_plan_type(self, training_plan_type: TrainingPlans) -> TrainingPlans:
        self._training_plan_type = training_plan_type
        return self._training_plan_type

    def create_aggregator_args(self, *args, **kwargs) -> Tuple[dict, dict]:
        return {}, {}

    def scaling(self, model_param: dict, *args, **kwargs) -> dict:
        """Should be overwritten by child if a scaling operation is involved in aggregator"""
        return model_param

    def save_state(self,
                   training_plan: BaseTrainingPlan,
                   breakpoint_path: Optional[str] = None,
                   args_to_save_to_file: Optional[List[str]] = None, *args) -> Dict[str, Any]:
        """
        use for breakpoints. save the aggregator state
        """
        if args_to_save_to_file is None:
            args_to_save_to_file = []
        aggregator_args = copy.deepcopy(self._aggregator_args)
        if breakpoint_path is not None and aggregator_args is not None:
            for arg_name, aggregator_arg in aggregator_args.items():
                if arg_name in args_to_save_to_file and isinstance(aggregator_arg, dict):
                    
                
                    for node_id, node_arg in aggregator_arg.items():
                        filename = os.path.join(breakpoint_path, arg_name + '_' + node_id + '.pt')
                        training_plan.save(filename, node_arg)
                        aggregator_args[arg_name][node_id] = filename  # replacing value by a path towards a file
            
        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "parameters": aggregator_args
        }
        return state

    def load_state(self, state: Dict[str, Any] = None, *args):
        """
        use for breakpoints. load the aggregator state
        """
        self._aggregator_args = state['parameters']

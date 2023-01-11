# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
top class for all aggregators
"""


import os
from typing import Dict, Any, List, Optional, Tuple

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.common.logger import logger
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.researcher.datasets import FederatedDataSet


class Aggregator:
    """
    Defines methods for aggregating strategy
    (eg FedAvg, FedProx, SCAFFOLD, ...).
    """
    def __init__(self):
        self._aggregator_args: dict = None
        self._fds: FederatedDataSet = None
        self._training_plan_type: TrainingPlans = None

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

    def check_values(self, *args, **kwargs) -> True:
        return True

    def set_fds(self, fds: FederatedDataSet) -> FederatedDataSet:
        self._fds = fds
        return self._fds
    
    def set_training_plan_type(self, training_plan_type: TrainingPlans) -> TrainingPlans:
        self._training_plan_type = training_plan_type
        return self._training_plan_type

    def create_aggregator_args(self, *args, **kwargs) -> Tuple[dict, dict]:
        """Returns aggregator arguments that are expecting by the nodes
        
        Returns:
        dict: contains `Aggregator` parameters that will be sent through MQTT message
                service
        dict: contains parameters that will be sent through file exchange message.
                Both dictionaries are mapping node_id to 'Aggregator` parameters specific 
                to each Node.
        """
        return self._aggregator_args or {}, {}

    # def scaling(self, model_param: dict, *args, **kwargs) -> dict:
    #     """Should be overwritten by child if a scaling operation is involved in aggregator"""
    #     return model_param

    def save_state(self,
                   training_plan: Optional[BaseTrainingPlan] = None,
                   breakpoint_path: Optional[str] = None,
                   **aggregator_args_create) -> Dict[str, Any]:
        """
        use for breakpoints. save the aggregator state
        """
        aggregator_args_thr_msg, aggregator_args_thr_files = self.create_aggregator_args(**aggregator_args_create)
        if aggregator_args_thr_msg:
            if self._aggregator_args is None:
                self._aggregator_args = {}
            self._aggregator_args.update(aggregator_args_thr_msg)
            #aggregator_args = copy.deepcopy(self._aggregator_args)
            if breakpoint_path is not None and aggregator_args_thr_files:
                for node_id, node_arg in aggregator_args_thr_files.items():
                    if isinstance(node_arg, dict):

                        for arg_name, aggregator_arg in node_arg.items():
                            if arg_name != 'aggregator_name': # do not save `aggregator_name` as a file
                                filename = self._save_arg_to_file(training_plan, breakpoint_path,
                                                                  arg_name, node_id, aggregator_arg)
                                self._aggregator_args.setdefault(arg_name, {})
                                    

                                self._aggregator_args[arg_name][node_id] = filename  # replacing value by a path towards a file
                    else:
                        filename = self._save_arg_to_file(training_plan, breakpoint_path, arg_name, node_id, node_arg)
                        self._aggregator_args[arg_name] = filename
        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "parameters": self._aggregator_args
        }
        return state

    def _save_arg_to_file(self, training_plan: BaseTrainingPlan, breakpoint_path: str, arg_name: str,
                          node_id: str, arg: Any) -> str:
        
        filename = os.path.join(breakpoint_path, arg_name + '_' + node_id + '.pt')
        training_plan.save(filename, arg)
        return filename

    def load_state(self, state: Dict[str, Any] = None, **kwargs):
        """
        use for breakpoints. load the aggregator state
        """
        self._aggregator_args = state['parameters']

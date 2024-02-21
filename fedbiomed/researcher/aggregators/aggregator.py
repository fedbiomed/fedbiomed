# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
top class for all aggregators
"""


import os
import functools
import uuid
import math
from typing import Any, Dict, Optional, Tuple, List

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
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
            ": aggregate method should be overloaded by the choosen strategy"
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

    def create_aggregator_args(self, *args, **kwargs) -> Dict:
        """Returns aggregator arguments that are expecting by the nodes

        Args:
            args: ignored
            kwargs: ignored

        Returns:
            contains `Aggregator` parameters/argument that will be shared with the nodes 
        """
        return self._aggregator_args or {}

    def save_state_breakpoint(
        self,
        breakpoint_path: Optional[str] = None,
        **aggregator_args_create: Any,
    ) -> Dict[str, Any]:
        """
        use for breakpoints. save the aggregator state
        """
        aggregator_args = self.create_aggregator_args(**aggregator_args_create)
        if aggregator_args:

            if self._aggregator_args is None:
                self._aggregator_args = {}
            self._aggregator_args.update(aggregator_args)

        if breakpoint_path:
            filename = self._save_arg_to_file(breakpoint_path, 'aggregator_args', uuid.uuid4(), self._aggregator_args)

        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "parameters": filename if breakpoint_path else self._aggregator_args
        }

        return state

    def _save_arg_to_file(self, breakpoint_path: str, arg_name: str, node_id: str, arg: Any) -> str:

        filename = os.path.join(breakpoint_path, f"{arg_name}_{node_id}.mpk")
        Serializer.dump(arg, filename)
        return filename

    def load_state_breakpoint(self, state: Dict[str, Any], **kwargs) -> None:
        """
        use for breakpoints. load the aggregator state
        """
        if not isinstance(state["parameters"], Dict):
            self._aggregator_args = Serializer.load(state['parameters'])
        else:
            self._aggregator_args = state['parameters']

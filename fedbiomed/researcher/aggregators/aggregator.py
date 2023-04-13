# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
top class for all aggregators
"""


import os
import functools
import math
from typing import Any, Dict, Optional, Tuple, List

from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.common.secagg import SecaggCrypter


class Aggregator:
    """
    Defines methods for aggregating strategy
    (eg FedAvg, FedProx, SCAFFOLD, ...).
    """

    def __init__(self):
        self._aggregator_args: dict = None
        self._fds: FederatedDataSet = None
        self._training_plan_type: TrainingPlans = None
        self._secagg_crypter = SecaggCrypter()

    def secure_aggregation(
            self,
            params: List[List[int]],
            encryption_factors: List[Dict[str, List[int]]],
            secagg_random: float,
            aggregation_round: int,
            total_sample_size: int,
            training_plan: 'BaseTrainingPlan'
    ):
        """ Apply aggregation for encrypted model parameters

        Args:
            params: List containing list of encrypted parameters of each node
            encryption_factors: List of encrypted integers to validate encryption
            secagg_random: Randomly generated float value to validate secure aggregation correctness
            aggregation_round: The round of the aggregation.
            total_sample_size: Sum of sample sizes used for training
            training_plan: Training plan instance used for the training.

        Returns:
            aggregated model parameters
        """

        # TODO: verify with secagg context number of parties
        num_nodes = len(params)

        # TODO: Use server key here
        key = -(len(params) * 10)

        # IMPORTANT = Keep this key for testing purposes
        key = -4521514305280526329525552501850970498079782904248225896786295610941010325354834129826500373412436986239012584207113747347251251180530850751209537684586944643780840182990869969844131477709433555348941386442841023261287875379985666260596635843322044109172782411303407030194453287409138194338286254652273563418119335656859169132074431378389356392955315045979603414700450628308979043208779867835835935403213000649039155952076869962677675951924910959437120608553858253906942559260892494214955907017206115207769238347962438107202114814163305602442458693305475834199715587932463252324681290310458316249381037969151400784780
        logger.info("Securely aggregating model parameters...")

        aggregate = functools.partial(self._secagg_crypter.aggregate,
                                      current_round=aggregation_round,
                                      num_nodes=num_nodes,
                                      key=key,
                                      total_sample_size=total_sample_size
                                      )
        # Validation
        encryption_factors = [f for k, f in encryption_factors.items()]
        validation: List[int] = aggregate(params=encryption_factors)

        if len(validation) != 1 or not math.isclose(validation[0], secagg_random, abs_tol=0.01):
            raise FedbiomedAggregatorError("Aggregation is failed due to incorrect decryption.")

        aggregated_params = aggregate(params=params)

        # Convert model params
        model = training_plan._model

        model_params = model.unflatten(aggregated_params)

        return model_params

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

    def save_state(
        self,
        breakpoint_path: Optional[str] = None,
        **aggregator_args_create: Any,
    ) -> Dict[str, Any]:
        """
        use for breakpoints. save the aggregator state
        """
        aggregator_args_thr_msg, aggregator_args_thr_files = self.create_aggregator_args(**aggregator_args_create)
        if aggregator_args_thr_msg:
            if self._aggregator_args is None:
                self._aggregator_args = {}
            self._aggregator_args.update(aggregator_args_thr_msg)
            # aggregator_args = copy.deepcopy(self._aggregator_args)
            if breakpoint_path is not None and aggregator_args_thr_files:
                for node_id, node_arg in aggregator_args_thr_files.items():
                    if isinstance(node_arg, dict):

                        for arg_name, aggregator_arg in node_arg.items():
                            if arg_name != 'aggregator_name': # do not save `aggregator_name` as a file
                                filename = self._save_arg_to_file(breakpoint_path, arg_name, node_id, aggregator_arg)
                                self._aggregator_args.setdefault(arg_name, {})


                                self._aggregator_args[arg_name][node_id] = filename  # replacing value by a path towards a file
                    else:
                        filename = self._save_arg_to_file(breakpoint_path, arg_name, node_id, node_arg)
                        self._aggregator_args[arg_name] = filename
        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "parameters": self._aggregator_args
        }
        return state

    def _save_arg_to_file(self, breakpoint_path: str, arg_name: str, node_id: str, arg: Any) -> str:

        filename = os.path.join(breakpoint_path, f"{arg_name}_{node_id}.mpk")
        Serializer.dump(arg, filename)
        return filename

    def load_state(self, state: Dict[str, Any], **kwargs) -> None:
        """
        use for breakpoints. load the aggregator state
        """
        self._aggregator_args = state['parameters']

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Top class for strategy implementation
"""


from typing import Dict, Any

from fedbiomed.common.constants  import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedStrategyError
from fedbiomed.common.logger     import logger

from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.datasets import FederatedDataSet


class Strategy:
    """
    Default Strategy as Parent class. Custom strategy classes must inherit from this parent class.

    """

    def __init__(self, data: FederatedDataSet):
        """

        Args:
            data: Object that includes all active nodes and the meta-data of the dataset that is going to be
                used for federated training.
        """
        self._fds = data
        self._sampling_node_history = {}
        self._success_node_history = {}
        self._parameters = None

    def sample_nodes(self, round_i: int):
        """
        Abstract method that must be implemented by child class

        Args:
            round_i: Current round of experiment
        """
        msg = ErrorNumbers.FB402.value + \
            ": sample nodes method should be overloaded by the provided strategy"
        logger.critical(msg)
        raise FedbiomedStrategyError(msg)

    def refine(self, training_replies: Responses, round_i: int) -> tuple[list, list]:
        """
        Abstract method that must be implemented by child class

        Args:
            training_replies: is a list of elements of type
                 Response( { 'success': m['success'],
                             'msg': m['msg'],
                             'dataset_id': m['dataset_id'],
                             'node_id': m['node_id'],
                             'params_path': params_path,
                             'params': params } )
            round_i: Current round of experiment

        Raises:
            FedbiomedStrategyError: If method is not implemented by child class
        """
        msg = ErrorNumbers.FB402.value + \
            ": refine method should be overloaded by the provided strategy"
        logger.critical(msg)
        raise FedbiomedStrategyError(msg)

    def save_state(self) -> Dict[str, Any]:
        """
        Method for saving strategy state for saving breakpoints

        Returns:
            The state of the strategy
        """

        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "parameters": self._parameters,
            "fds": self._fds.data()
        }
        return state

    def load_state(self, state: Dict[str, Any] = None, **kwargs):
        """
        Method for loading strategy state from breakpoint state

        Args:
            state: The state that will be loaded
        """
        # fds may be modified and diverge from Experiment
        self._fds = FederatedDataSet(state.get('fds'))
        self._parameters = state['parameters']

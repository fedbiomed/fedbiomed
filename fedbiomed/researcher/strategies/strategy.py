# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Top class for strategy implementation
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedStrategyError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import TrainReply


class Strategy:
    """
    Default Strategy as Parent class. Custom strategy classes must inherit from this parent class.

    !!! warning "Inconsistent history"
        The Strategy class keeps a history of sampled and successful nodes. No attempt is made to keep this history
        consistent when the `_fds` member is modified.

    """

    def __init__(self):
        """

        Args:
            data: Object that includes all active nodes and the meta-data of the dataset that is going to be
                used for federated training.
        """
        self._sampling_node_history = {}
        self._success_node_history = {}
        self._parameters = None

    def sample_nodes(self, from_nodes: List[str], round_i: int):
        """
        Abstract method that must be implemented by child class

        Args:
            from_nodes: the node ids which may be sampled
            round_i: Current round of experiment
        """
        msg = (
            ErrorNumbers.FB402.value
            + ": sample nodes method should be overloaded by the provided strategy"
        )
        logger.critical(msg)
        raise FedbiomedStrategyError(msg)

    def refine(
        self, training_replies: Dict[str, TrainReply], round_i: int
    ) -> Tuple[
        Dict[str, Dict[str, Union["torch.Tensor", "np.ndarray"]]],
        Dict[str, float],
        int,
        Dict[str, List[int]],
    ]:
        """
        Abstract method that must be implemented by child class

        Args:
            training_replies: Dictionary of replies from nodes
            round_i: Current round of experiment

        Raises:
            FedbiomedStrategyError: If method is not implemented by child class
        """
        msg = (
            ErrorNumbers.FB402.value
            + ": refine method should be overloaded by the provided strategy"
        )
        logger.critical(msg)
        raise FedbiomedStrategyError(msg)

    def save_state_breakpoint(self) -> Dict[str, Any]:
        """
        Method for saving strategy state for saving breakpoints

        Returns:
            The state of the strategy
        """

        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "parameters": self._parameters,
        }
        return state

    def load_state_breakpoint(self, state: Dict[str, Any] = None):
        """
        Method for loading strategy state from breakpoint state

        Args:
            state: The state that will be loaded
        """
        # fds may be modified and diverge from Experiment
        self._parameters = state["parameters"]

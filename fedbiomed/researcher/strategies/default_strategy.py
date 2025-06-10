# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of the default strategy

This strategy is used then user does not provide its own
"""

import copy
from typing import Dict, List, Tuple, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedStrategyError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import TrainReply
from fedbiomed.researcher.strategies import Strategy


class DefaultStrategy(Strategy):
    """
    Default strategy to be used when sampling/selecting nodes
    and to check whether nodes have responded or not

    Strategy is:
    - select all nodes for each round
    - raise an error if one node does not answer
    - raise an error is one node returns an error
    """

    def sample_nodes(self, from_nodes: List[str], round_i: int) -> List[str]:
        """Samples and selects nodes on which to train local model. In this strategy we will consider all existing
        nodes

        Args:
            from_nodes: the node ids which may be sampled
            round_i: number of round.

        Returns:
            node_ids: list of all node ids considered for training during
                this round `round_i`.
        """
        self._sampling_node_history[round_i] = copy.copy(from_nodes)

        return from_nodes

    def refine(
        self, training_replies: Dict[str, TrainReply], round_i: int
    ) -> Tuple[
        Dict[str, Dict[str, Union["torch.Tensor", "numpy.ndarray"]]],
        Dict[str, float],
        int,
        Dict[str, List[int]],
    ]:
        """
        The method where node selection is completed by extracting parameters and length from the training replies

        Args:
            training_replies: Dictionary of replies from nodes.
            round_i: Current round of experiment.

        Returns:
            model_params: list with each element representing a dictionary. Its only key represents
                the node_id and the corresponding value is a dictionary containing list of weight
                matrices of every node : [{"n1":{"layer1":m1,"layer2":m2},{"layer3":"m3"}},{"n2":
                ...}] Including the node_id is useful for the proper functioning of some strategies
                like Scaffold: At each round, local model params are linked to a certain correction.
                The correction is updated every round. The computation of correction states at round
                i is dependant to client states and correction states of round i-1. Since
                training_replies    can potentially order the node replies differently from round to
                round, the bridge between all these parameters is represented by the node_id
            weights: Proportions list, each element of this list represents a dictionary with
                its only key as the node_id and its value the proportion of lines the node has
                with respect to the whole
            total_rows: sum of number of samples used by all nodes
            encryption_factors: encryption factors from the participating nodes

        Raises:
            FedbiomedStrategyError: If any of the following occur:
                - A node in the sampling list does not reply
                - Not all nodes successfully complete training
                - A successful reply does not include `sample_size` value.
        """
        missing_node_replies = []
        for node in self._sampling_node_history.get(round_i):
            if node not in training_replies:
                missing_node_replies.append(node)
                logger.error(ErrorNumbers.FB409.value + " (node = " + node + ")")
        if missing_node_replies:
            raise FedbiomedStrategyError(
                ErrorNumbers.FB408.value
                + f": {len(missing_node_replies)} missing training replies for round {round_i}"
            )

        # check that all nodes that answer could successfully train
        self._success_node_history[round_i] = []
        all_success = True
        model_params = {}
        sample_sizes = {}
        encryption_factors = {}
        total_rows = 0
        for tr in training_replies.values():
            if tr["success"] is True:
                model_params[tr["node_id"]] = tr["params"]
                encryption_factors[tr["node_id"]] = tr.get("encryption_factor", None)

                if tr.get("sample_size") is None:
                    # if a Node `sample_size` is None, we cannot compute the weights: in this case
                    # return an error
                    raise FedbiomedStrategyError(
                        ErrorNumbers.FB402.value
                        + f" : Node {tr['node_id']} did not return "
                        "any `sample_size` value (number of samples seen during one Round),"
                        " can not compute weights for the aggregation. Aborting"
                    )
                sample_sizes[tr["node_id"]] = tr["sample_size"]

                total_rows += tr["sample_size"]
                self._success_node_history[round_i].append(tr["node_id"])
            else:
                all_success = False
                msg = tr.get("msg") or ""
                logger.error(
                    "Unsuccessful training reply{} (node = {} )".format(
                        f": {msg}" if msg else "", tr["node_id"]
                    )
                )

        if not all_success:
            raise FedbiomedStrategyError(ErrorNumbers.FB402.value)

        weights = {
            node_id: (
                sample_size / total_rows if total_rows != 0 else 1 / len(sample_sizes)
            )
            for node_id, sample_size in sample_sizes.items()
        }

        logger.info(
            f"Nodes that successfully reply in round {round_i} "
            f"{self._success_node_history[round_i]}"
        )

        return model_params, weights, total_rows, encryption_factors

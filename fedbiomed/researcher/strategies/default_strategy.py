"""
Implementation of thedefault strategy

This strategy is used then user does not provide its own
"""

import uuid
from typing import List, Tuple

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedStrategyError
from fedbiomed.common.logger import logger
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.responses import Responses
from fedbiomed.researcher.strategies.strategy import Strategy


class DefaultStrategy(Strategy):
    """
    Default strategy to be used when sampling/selecting nodes
    and checking whether nodes have responded or not

    Strategy is:
    - select all node for each round
    - raise an error if one node does not answer
    - raise an error is one node returns an error
    """

    def __init__(self, data: FederatedDataSet):
        """Constructor of Default Strategy

        Args:
            data: Object that includes all active nodes and the meta-data of the dataset that is going to be
                used for federated training. Should be passed to `super().__init__` to initialize parent class
        """

        super().__init__(data)

    def sample_nodes(self, round_i: int) -> List[uuid.UUID]:
        """Samples and selects nodes on which to train local model. In this strategy we will consider all existing
        nodes

        Args:
            round_i: number of round.

        Returns:
          node_ids: list of all node ids considered for training during
            this round `round_i`.
        """
        self._sampling_node_history[round_i] = self._fds.node_ids()

        return self._fds.node_ids()

    def refine(self, training_replies: Responses, round_i: int) -> Tuple[List, List]:
        """
        The method where node selection is completed by extracting parameters and length from the training replies

        Args:
            training_replies: is a list of elements of type
                 Response( { 'success': m['success'],
                             'msg': m['msg'],
                             'dataset_id': m['dataset_id'],
                             'node_id': m['node_id'],
                             'params_path': params_path,
                             'params': params } )
            round_i: Current round of experiment

        Returns:
            A tuple that contains:
                weights: Proportions list, each element of this list represent the proportion of lines the node has with
                    respect to the whole,
                model_params:  list containing dictionaries with list of weight matrices of
                    every node : [{"n1":{"layer1":m1,"layer2":m2},{"layer3":"m3"}},{"n2": ...}]

        Raises:
            FedbiomedStrategyError: - Miss-matched in answered nodes and existing nodes
                - If not all nodes successfully completes training
        """
        models_params = []
        weights = []

        # check that all nodes answered
        cl_answered = [val["node_id"] for val in training_replies.data()]

        answers_count = 0
        for cl in self.sample_nodes(round_i):
            if cl in cl_answered:
                answers_count += 1
            else:
                # this node did not answer
                logger.error(ErrorNumbers.FB408.value + " (node = " + cl + ")")

        if len(self.sample_nodes(round_i)) != answers_count:
            if answers_count == 0:
                # none of the nodes answered
                msg = ErrorNumbers.FB407.value

            else:
                msg = ErrorNumbers.FB408.value

            logger.critical(msg)
            raise FedbiomedStrategyError(msg)

        # check that all nodes that answer could successfully train
        self._success_node_history[round_i] = []
        all_success = True
        for tr in training_replies:
            if tr["success"] is True:
                model_params = tr["params"]
                models_params.append(model_params)
                self._success_node_history[round_i].append(tr["node_id"])
            else:
                # node did not succeed
                all_success = False
                logger.error(
                    ErrorNumbers.FB409.value + " (node = " + tr["node_id"] + ")"
                )

        if not all_success:
            raise FedbiomedStrategyError(ErrorNumbers.FB402.value)

        # so far, everything is OK
        totalrows = sum([val[0]["shape"][0] for (key, val) in self._fds.data().items()])
        weights = [
            val[0]["shape"][0] / totalrows for (key, val) in self._fds.data().items()
        ]
        logger.info(
            "Nodes that successfully reply in round "
            + str(round_i)
            + " "
            + str(self._success_node_history[round_i])
        )
        return models_params, weights

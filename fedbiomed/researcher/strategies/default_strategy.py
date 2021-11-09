from fedbiomed.common.logger import logger

import uuid
from typing import List, Tuple, Dict, Any

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.exceptions import DefaultStrategyException
from fedbiomed.researcher.strategies.strategy import Strategy


class DefaultStrategy(Strategy):
        """Default strategy to be used when sampling/selecting nodes
        and checking whether nodes have responded or not
        """
        def __init__(self, data: FederatedDataSet):
                super().__init__(data)
                self.parameters = None

        def sample_nodes(self, round_i: int) -> List[uuid.UUID]:
                """Samples and selects nodes on which to train local model.
                        In this strategy we will consider all existing nodes

                Args:
                        round_i (int): number of round.

                Returns:
                        node_ids: list of all node ids considered for training during
                        this round `round_i.
                """
                self._sampling_node_history[round_i] = self._fds.node_ids
                return self._fds.node_ids

        def refine(self, training_replies, round_i) -> Tuple[List, List]:
                models_params = []
                weights = []
                try:
                        # check that all nodes answered
                        cl_answered = [val['node_id'] for val in training_replies.data]
                        for cl in self.sample_nodes(round_i):
                                if cl not in cl_answered:
                                        raise DefaultStrategyException("At least one node didn't answer " + cl)

                        # check that all nodes that answer could train
                        self._success_node_history[round_i] = []
                        for tr in training_replies:
                                if tr['success'] == True:
                                        model_params = tr['params']
                                        models_params.append(model_params)
                                        self._success_node_history[round_i].append(tr['node_id'])
                                else:
                                        raise DefaultStrategyException("At least one node couldn't train successfully " + tr['node_id'])
                except DefaultStrategyException as ex:
                        logger.error("default strategy exception: " + str(ex))


                totalrows = sum([val[0]["shape"][0] for (key,val) in self._fds.data().items()])
                weights = [val[0]["shape"][0] / totalrows for (key,val) in self._fds.data().items()]
                logger.info('Nodes that successfully reply in round ' + str(round_i) + ' ' + str(self._success_node_history[round_i] ))
                return models_params, weights


        def save_state(self) -> Dict[str, Any]:
                state = {
                        "class": type(self).__name__,
                        "module": self.__module__,
                        "parameters": self.parameters
                }
                return state

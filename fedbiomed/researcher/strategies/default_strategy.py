from fedbiomed.common.logger import logger
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.exceptions import DefaultStrategyException
from fedbiomed.researcher.strategies.strategy import Strategy


class DefaultStrategy(Strategy):
        def __init__(self, data: FederatedDataSet):
                super().__init__(data)

        def sample_clients(self, round_i):
                self._sampling_client_history[round_i] = self._fds.client_ids
                return self._fds.client_ids

        def refine(self, training_replies, round_i) :
                models_params = []
                weights = []
                try:
                        # check that all nodes answered
                        cl_answered = [ val['client_id'] for val in training_replies.data ]
                        for cl in self.sample_clients(round_i):
                                if cl not in cl_answered:
                                        raise DefaultStrategyException("At least one node didn't answer " + cl)

                        # check that all nodes that answer could train
                        self._success_client_history[round_i] = []
                        for tr in training_replies:
                                if tr['success'] == True:
                                        model_params = tr['params']
                                        models_params.append(model_params)
                                        self._success_client_history[round_i].append(tr['client_id'])
                                else:
                                        raise DefaultStrategyException("At least one node couldn't train successfully " + tr['client_id'])
                except DefaultStrategyException as ex:
                        logger.error("default strategy exception: " + str(ex))

                totalrows = sum([ val[0]["shape"][0] for (key,val) in self._fds.data().items() ] )
                weights = [ val[0]["shape"][0] / totalrows for (key,val) in self._fds.data().items() ]
                logger.info('Clients that successfully reply in round ' + str(round_i) + ' ' + str(self._success_client_history[round_i] ))
                return models_params, weights

"""
Top class for strategy implementation
"""


from typing import Dict, Any

from fedbiomed.common.constants  import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedStrategyError
from fedbiomed.common.logger     import logger

from fedbiomed.researcher.datasets import FederatedDataSet


class Strategy:
    """
    Default Strategy parent class

    provide node management helpers
    """

    def __init__(self, data: FederatedDataSet):
        self._fds = data
        self._sampling_node_history = {}
        self._success_node_history = {}
        self._parameters = None

    def sample_nodes(self, round_i: int):
        """
        1) returns the id of the sampled nodes
        2) TODO: creates the nodes weights in case they can all perform their work
        """
        return

    def refine(self, training_replies, round_i):
        """
        @param training_replies is a list of elements of type
         Response( { 'success': m['success'],
                     'msg': m['msg'],
                     'dataset_id': m['dataset_id'],
                     'node_id': m['node_id'],
                     'params_path': params_path,
                     'params': params } )

        @return weights : proportions list, each element of this list represent the proportion of
        lines the node has with respect to the whole.
        @return model_params : list containing dictionnaries with list of weight matrices of
        every node : [{"n1":{"layer1":m1,"layer2":m2},{"layer3":"m3"}},{"n2": ...}]
        """
        msg = ErrorNumbers.FB402.value + \
            ": refine method should be overloaded by the provided strategy"
        logger.critical(msg)
        raise FedbiomedStrategyError(msg)


    def save_state(self) -> Dict[str, Any]:
        """
        used in breakpoint saving
        """
        state = {
            "class": type(self).__name__,
            "module": self.__module__,
            "parameters": self._parameters,
            "fds": self._fds.data()
        }
        return state

    def load_state(self, state: Dict[str, Any] = None):
        """
        used in breakpoint loading
        """
        # fds may be modified and diverge from Experiment
        self._fds = FederatedDataSet(state.get('fds'))
        self._parameters = state['parameters']

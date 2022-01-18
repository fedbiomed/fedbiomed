import uuid
from typing import List, Dict, Any


from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.logger import logger
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.exceptions import StrategyException

# TODO: do we consider it as an abstract class ?
class Strategy:
    """
    Default Strategy parent class

    provide node management helpers
    """

    def __init__(self, dataset: FederatedDataSet):
        self._fds = dataset
        self._sampling_node_history = {} # list of nodes at each round
        self._success_node_history = {}  # status of nodes at each round
        self._parameters = None

        # used to verify that the current round is OK (private)
        self.__current_round = -1

    def sample_nodes(self, round_i: int):
        """
        1) returns the id of the sampled nodes
        2) TODO: creates the nodes weights in case they can all perform their work
        """
        logger.critical("your strategy must implement sample_nodes()")
        raise StrategyException(ErrorNumbers.FB402.value)
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

        @return weights : proportions list, each element of this list represent the proportion of lines the node has with respect to the whole.
        @return model_params : list containing dictionnaries with list of weight matrices of every node : [{"n1":{"layer1":m1,"layer2":m2},{"layer3":"m3"}},{"n2": ...}]
        """
        return

    def save_state(self) -> Dict[str, Any]:
        state = {
                "class": type(self).__name__,
                "module": self.__module__,
                "parameters": self._parameters,
                "fds": self._fds.data()
        }
        return state

    def load_state(self,  state: Dict[str, Any]=None):
        # fds may be modified and diverge from Experiment
        self._fds = FederatedDataSet(state.get('fds'))
        self._parameters = state['parameters']


    def store_node_history(self, round_i: int, node_list: List[uuid.UUID]):
        """
        Help function: stores the sampled nodes at each round

        It **must** be called by sample_nodes() of the child class
        """

        # verify that the rounds are only incremented by one
        _error = False

        if ( round_i - self.__current_round > 1 ):
            _error = True

        if round_i < self.__current_round :
            _error = True

        if _error :
            logger.critical("strategy called with wrong round_i")
            raise StrategyException(ErrorNumbers.FB402.value)

        self.__current_round = round_i

        self._sampling_node_history[round_i] = node_list
        return

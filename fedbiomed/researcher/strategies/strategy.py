from fedbiomed.researcher.datasets import FederatedDataSet
from typing import Dict, Any

# TODO: do we consider it as an abstract class ?
class Strategy:
        """Default Strategy parent class"""
        def __init__(self, data: FederatedDataSet):
                self._fds = data
                self._sampling_node_history = {}
                self._success_node_history = {}

        """
                1) returns the id of the sampled nodes
                2) TODO: creates the nodes weights in case they can all perform their work
        """
        def sample_nodes(self, round_i: int):
                return

        """
            @param training_replies is a list of elemets of type Response( { 'success': m['success'], 'msg': m['msg'], 'dataset_id': m['dataset_id'],
                                    'node_id': m['node_id'],
                             'params_path': params_path, 'params': params } )
            @return weights : proportions list, each element of this list represent the proportion of lines the node has with respect to the whole.
            @return model_params : list containing dictionnaries with list of weight matrices of every node : [{"n1":{"layer1":m1,"layer2":m2},{"layer3":"m3"}},{"n2": ...}]
        """
        def refine(self, training_replies, round_i):
                return

        def save_state(self) -> Dict[str, Any]:
                state = {
                        "class": type(self).__name__,
                        "module": self.__module__,
                        "parameters": self.parameters
                }
                return state

        def load_state(self,  state: Dict[str, Any]=None):
                pass

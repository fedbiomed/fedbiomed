
from fedbiomed.researcher.datasets import FederatedDataSet
from typing import Dict, Any

# TODO: do we consider it as an abstract class ?
class Strategy:
	"""Default Strategy parent class"""
	def __init__(self, data: FederatedDataSet):
		self._fds = data
		self._sampling_client_history = {}
		self._success_client_history = {}

	"""
		1) returns the id of the sampled clients
		2) TODO: creates the clients weights in case they can all perform their work
	"""
	def sample_clients(self, round_i: int):
		return

	"""
	    @param training_replies is a list of elemets of type Response( { 'success': m['success'], 'msg': m['msg'], 'dataset_id': m['dataset_id'],
	                            'client_id': m['client_id'],
                             'params_path': params_path, 'params': params } )
	    @return weights : proportions list, each element of this list represent the proportion of lines the node has with respect to the whole.
	    @return model_params : list containing dictionnaries with list of weight matrices of every node : [{"n1":{"layer1":m1,"layer2":m2},{"layer3":"m3"}},{"n2": ...}]
	"""
	def refine(self, training_replies, round_i):
		return

	def save_state(self) -> Dict[str, Any]:
		return None
	
	def load_state(self):
		pass

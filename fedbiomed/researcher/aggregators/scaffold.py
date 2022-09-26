"""
"""

from copy import deepcopy
from typing import Dict, Iterator, OrderedDict

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging
from fedbiomed.researcher.aggregators.functional import initialize

class Scaffold(Aggregator):
    """
    Defines the Scaffold strategy
    """

    def __init__(self, server_lr: float, scaffold_option: int = 2):
        """Construct `Scaffold` object as an instance of [`Aggregator`]
        [fedbiomed.researcher.aggregators.Aggregator].
        
        References:
        [Scaffold: Stochastic Controlled Averaging for Federated Learning][https://arxiv.org/abs/1910.06378]
        
        Args:
            server_lr (float): server's (or Researcher's) learning rate
            scaffold_option (int). Scaffold option. If equals to 1, client correction equals gradient update.
            If equals to 2, client correction equals `grad(y_i)/(lr) - c` (according to 
            [Scaffold][https://arxiv.org/abs/1910.06378] paper). Defaults to 2.
        """
        super(Scaffold, self).__init__()
        self.aggregator_name = "Scaffold"
        self.server_lr = server_lr
        self.nodes_correction_states = {}
        self.aggr_correction = .0
        self.scaffold_option = scaffold_option

    def aggregate(self, model_params: list, weights: list, global_model: OrderedDict, lr: float, n_updates: int=1, *args, **kwargs) -> Dict:
        """ Aggregates local models sent by participating nodes into a global model, using Federated Averaging
        also present in Scaffold for the aggregation step.

        Args:
            model_params: contains each model layers
            weights: contains all weights of a given layer.

        Returns:
            Aggregated parameters
        """
        
        weights_processed = [list(weight.values())[0] for weight in weights] # same retrieving
        
        model_params_processed = self.scaling(model_params, global_model)
        model_params_processed = [list(model_param.values())[0] for model_param in model_params] # model params are contained in a dictionary with node_id as key, we just retrieve the params

        #model_params_processed = list(model_params_processed.values())

        weights_processed = self.normalize_weights(weights_processed)
        aggregated_parameters = federated_averaging(model_params_processed, weights_processed)
        
        self.update_correction_states(model_params_processed, global_model, lr, n_updates)
        return aggregated_parameters
    def init_correction_states(self, global_model: OrderedDict, node_ids: Iterator):
        # initialize nodes states
        init_params = {key:initialize(tensor)[1] for key, tensor in global_model.items()}
        self.nodes_correction_states = {node_id: deepcopy(init_params) for node_id in node_ids}


    
    def scaling(self, model_params: list, global_model: OrderedDict, *args, **kwargs) -> list:
        # should scale regading option
        for idx, model_param in enumerate(model_params):
            node_id = list(model_param.keys())[0]
            for layer in model_param[node_id]:
                model_params[idx][node_id][layer] = model_param[node_id][layer] * self.server_lr + (1 - self.server_lr) * global_model.state_dict()[layer]
        return model_params
        
    def update_correction_states(self, model_params: list, global_model: OrderedDict, lr: float, n_updates: int=0):
        for node_id, client_state in model_params.items(): # iterate params of each client
            for key in client_state:
                self.nodes_correction_states[node_id][key] += (global_model[key] - client_state[key]) / (self.server_lr * lr * n_updates)

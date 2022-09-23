"""
"""

from typing import Dict

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging


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
            If equals to 2, client correction equals `grad(y_i)/(K*lr) - c` (according to 
            [Scaffold][https://arxiv.org/abs/1910.06378] paper). Defaults to 2.
        """
        super(Scaffold, self).__init__()
        self.aggregator_name = "Scaffold"
        self.server_lr = server_lr
        self.client_correction_states = {}
        self.aggr_correction = .0
        self.scaffold_option = scaffold_option

    def aggregate(self, model_params: list, weights: list, n_updates: int=0) -> Dict:
        """ Aggregates local models sent by participating nodes into a global model, using Federated Averaging
        also present in Scaffold for the aggregation step.

        Args:
            model_params: contains each model layers
            weights: contains all weights of a given layer.

        Returns:
            Aggregated parameters
        """
        weights = self.normalize_weights(weights)
        return federated_averaging(model_params, weights)

    def update_client_states(self):
        """Updates client states (or client)
        """
    def update_correction_states(self, model_params: list, global_model: dict, lr: float, n_updates: int=0):
        for node_id, client_state in model_params.items(): # iterate params of each client
            for key in client_state:
                self.client_correction_states[node_id][key] += (global_model[key] - client_state[key]) / (self.server_lr * lr * n_updates)

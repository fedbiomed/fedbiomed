"""
"""

from typing import Dict

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging


class FedCos(Aggregator):
    """
    Defines the Federated learning with Cosine-similarity penalty strategy
    """

    def __init__(self):
        """
        constructor
        """
        super(FedCos, self).__init__()
        self.aggregator_name = "FedCos"

    def aggregate(self, model_params: list, weights: list) -> Dict:
        """
        Aggregates  local models sent by participating nodes into
        a global model, following Federated Averaging strategy, and 
        evaluate the current displacement of the global model
        with respect to the previous iteration.

        Args:
            model_params (list): contains each model layers
            weights (list): contains all weigths of a given
            layer.

        Returns:
            Dict: [description]
        """
        weights = self.normalize_weights(weights)

        # Recover global model at previous iteration
        global_params_list=[]
        weights_gl = [1 for _ in range(len(weights))]
        for cl in model_params:
            global_params_list.append(cl['global_r_1'])
            del cl['global_r_1']
        global_r_1 = federated_averaging(global_params_list, weights_gl)

        # Evaluate Updated gobal model through FedAvg
        global_update = federated_averaging(model_params, weights)

        # Evaluate global displacement
        disp_global = {}
        for name, param in global_update.items():
            disp_global[name]=param-global_r_1[name]
        global_update.update(disp_global=disp_global)
        
        return global_update

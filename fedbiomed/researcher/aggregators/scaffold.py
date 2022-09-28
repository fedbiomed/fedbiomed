"""
"""

from copy import deepcopy
from typing import Dict, Iterator, Mapping, OrderedDict, Union
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.common.training_args import TrainingArgs

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging
from fedbiomed.researcher.aggregators.functional import initialize

from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.researcher.datasets import FederatedDataSet

import torch
import numpy as np

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
        if server_lr == 0.:
            raise FedbiomedAggregatorError("Server learning rate cannot be equal to 0")
        self.server_lr = server_lr
        self.nodes_correction_states = None
        self.scaffold_option = scaffold_option
        self.aggregated_correction = None

    def aggregate(self, model_params: list,
                  weights: list,
                  global_model: Mapping[str, Union[torch.tensor, np.ndarray]],
                  training_args: TrainingArgs,
                  node_ids: Iterator[str],
                  n_updates: int = 1,
                  n_round: int = 0,
                  *args, **kwargs) -> Dict:
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
        
        lr = self.extract_learning_rate(training_args)
        if n_round == 0:
            self.init_correction_states(global_model, node_ids)
        self.update_correction_states(aggregated_parameters, global_model, lr, node_ids, n_updates)
        return aggregated_parameters

    def get_aggregator_args(self, global_model, node_ids: Iterator[str]) -> Dict:
        
        if self.nodes_correction_states is None:
            self.init_correction_states(global_model, node_ids) # making parameters JSON serializable
        aggregator_args = {}
        for node_id in node_ids:
            # serializing correction parameters
            serialized_aggregator_correction = {key: tensor.tolist() for key, tensor in self.nodes_correction_states[node_id].items()}
            aggregator_args.update({node_id: {'aggregator_name': self.aggregator_name,
                                              'aggregator_correction': serialized_aggregator_correction}})
        
        return aggregator_args

    def check_values(self, lr: float):
        """
        This method checks if all values are correct and have been set before using aggregator.
        Raises error otherwise
        This can prove usefull, so that user will have errors before performing first round of training

        Args:
            lr (float): _description_

        Raises:
            FedbiomedAggregatorError: _description_
        """
        # check if values are non zero
        
        if self.total_nb_nodes is None:
            raise FedbiomedAggregatorError(" Attribute `total_nb_nodes` is not set, cannot use SCAFFOLD aggregator")

    def extract_learning_rate(self, training_args: TrainingArgs) -> float:
        # to be implemented in a utils module
        try:
            lr = training_args['optimizer_args']['lr']
        except KeyError:
            raise  FedbiomedAggregatorError("Missing learning rate in the training argument. Cannot perform SCAFFOLD")
        return lr

    def init_correction_states(self, global_model: Mapping[str, Union[torch.tensor, np.ndarray]], node_ids: Iterator[str], to_list: bool = False):
        # initialize nodes states
        if to_list:
            # TODO: remove that part: we better send a file instead of a seralized JSON through MQTT
            init_params = {key:initialize(tensor)[1].tolist() for key, tensor in global_model.items()}
        else:
            init_params = {key:initialize(tensor)[1]for key, tensor in global_model.items()}
        self.nodes_correction_states = {node_id: deepcopy(init_params) for node_id in node_ids}

    
    def scaling(self, model_params: list, global_model: Mapping[str, Union[torch.tensor, np.ndarray]]) -> list:
        """
        
        Proof: 
            x <- x + eta_g * grad(x)
            x <- x + eta_g / S * sum_i(y_i - x)
            x <- x (1 - eta_g) + eta_g / S * sum_i(y_i)
            x <- sum_i(x (1 - eta_g) + eta_g * y_i) / S
            x <- avg(x (1 - eta_g) + eta_g * y_i)

        Args:
            model_params (list): _description_
            global_model (OrderedDict): _description_

        Returns:
            list: _description_
        """
        # refers as line 13 and 17 in pseudo code
        # should scale regading option
        for idx, model_param in enumerate(model_params):
            node_id = list(model_param.keys())[0]
            for layer in model_param[node_id]:
                model_params[idx][node_id][layer] = model_param[node_id][layer] * self.server_lr + (1 - self.server_lr) * global_model[layer]
        return model_params
        
    def update_correction_states(self, updated_model_params: dict,
                                 global_model: Mapping[str, Union[torch.tensor, np.ndarray]],
                                 lr: float, node_ids: Iterator[str], n_updates: int=1,):
        """_summary_
        
        Proof:
        
        c <- c + S/N grad(c)
        c <- c + 1/N sum_i(c_i(+) - c_i)
        c <- c + 1/N * sum_i( 1/ (K * eta_l)(x - y_i) - c)

        Args:
            updated_model_params (dict): _description_
            global_model (OrderedDict): _description_
            lr (float): _description_
            node_ids (Iterator[str]): Iterable of all nodes taking part in the round
            n_updates (int, optional): _description_. Defaults to 1.
        """
        # refers as line 12, 13 and 17 in pseudo code
        total_nb_nodes = len(self._fds.node_ids())  # get the total number of nodes
        
        weights = [1/total_nb_nodes] * len(node_ids)

        # get weights for weighted summation
        _tmp_correction_update = []
        for idx, node_id in enumerate(node_ids):
            _tmp_correction_update.append({})
            #_tmp_correction_update[idx][node_id] = {}
            for layer_name, node_layer in updated_model_params.items(): # iterate params of each client

                # `_tmp_correction_update`` is an intermediate variable equals to 1/ (K * eta_l)(x - y_i) - c
                _tmp_correction_update[idx][layer_name] = (global_model[layer_name] - node_layer) / (self.server_lr * lr * n_updates)
                _tmp_correction_update[idx][layer_name] -= self.nodes_correction_states[node_id][layer_name]

        _aggregated_tmp_correction_update = federated_averaging(_tmp_correction_update, weights)
        for node_id in node_ids:
            for layer_name, node_layer in updated_model_params.items(): 
                self.nodes_correction_states[node_id][layer_name] += _aggregated_tmp_correction_update[layer_name]

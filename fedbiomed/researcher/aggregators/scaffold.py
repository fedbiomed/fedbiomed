"""
"""

from copy import deepcopy
import copy
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, OrderedDict, Tuple, Union

from fedbiomed.common.logger import logger
from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.exceptions import FedbiomedAggregatorError

from fedbiomed.common.training_plans import BaseTrainingPlan

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import federated_averaging, weitghted_sum
from fedbiomed.researcher.aggregators.functional import initialize

from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.responses import Responses

import torch
import numpy as np


class Scaffold(Aggregator):
    """
    Defines the Scaffold strategy
    
    Attributes:
     - aggregator_name(str): name of the aggregator 
     - server_lr (float): value of the server learning rate
     - nodes_correction_states(Dict[str, Mapping[str, Union[torch.tensor, np.ndarray]]]): corrections
     parameters obtained for each client
    """

    def __init__(self, server_lr: float = .01, fds: Optional[FederatedDataSet] = None):
        """Constructs `Scaffold` object as an instance of [`Aggregator`]
        [fedbiomed.researcher.aggregators.Aggregator].
        
        Despite being an algorithm of choice for federated learning, it is observed that FedAvg
        suffers from `client-drift` when the data is heterogeneous (non-iid), resulting in unstable and slow
        convergence. SCAFFOLD uses control variates (variance reduction) to correct for the `client-drift` in its local
        updates.
        Intuitively, SCAFFOLD estimates the update direction for the server model (c) and the update direction for each
        client (c_i).
        The difference (c - c_i) is then an estimate of the client-drift which is used to correct the local update.
        
        References:
        [Scaffold: Stochastic Controlled Averaging for Federated Learning][https://arxiv.org/abs/1910.06378]
        [TCT: Convexifying Federated Learning using Bootstrapped Neural
        Tangent Kernels][https://arxiv.org/pdf/2207.06343.pdf]

        Args:
            server_lr (float): server's (or Researcher's) learning rate. Defaults to .01.
            fds (FederatedDataset, optional): FederatedDataset obtained after a `search` request. Defaults to None.
            
        """
        super(Scaffold, self).__init__()
        self.aggregator_name: str = "Scaffold"
        if server_lr == 0.:
            raise FedbiomedAggregatorError("SCAFFOLD Error: Server learning rate cannot be equal to 0")
        self.server_lr: float = server_lr
        self.nodes_correction_states: Dict[str, Mapping[str, Union[torch.tensor, np.ndarray]]] = None

        self.nodes_lr: Dict[str, List[float]] = {}
        if fds is not None:
            self.set_fds(fds)
        if self._aggregator_args is None:
            self._aggregator_args = {}
        #self.update_aggregator_params()

    def aggregate(self, model_params: list,
                  weights: List[Dict[str, float]],
                  global_model: Mapping[str, Union[torch.tensor, np.ndarray]],
                  training_plan: BaseTrainingPlan,
                  training_replies: Responses,
                  node_ids: Iterable[str],
                  n_updates: int = 1,
                  n_round: int = 0,
                  *args, **kwargs) -> Dict:
        """
        Aggregates local models coming from nodes into a global model, using SCAFFOLD algorithm (2nd option)
        [Scaffold: Stochastic Controlled Averaging for Federated Learning][https://arxiv.org/abs/1910.06378]
        
        Performed computations:
        -----------------------
        
        c_i(+) <- c_i - c + 1/(K*eta_l*eta_g)(x - y_i)
        c <- c + 1/N * avg_S(c_i(+) - c_i)
        
        x <- x + eta_g/S * avg_S(y_i - x)
        
        where, accroding to paper notations
            c_i: correction state for node `i`;
            c: correction state at the beginning of round
            eta_g: server's learning rate
            eta_l: nodes learning rate (may be different from one node to another)
            N: total number of node participating to federated learning
            S: number of nodes considered during current round (S<=N)
            K: number of updates done during the round (ie number of data batches).
            x: global model parameters
            y_i: node i 's local model parameters  

        Args:
            model_params (list): list of models parameters recieved from nodes
            weights (List[Dict[str, float]]): weitghs depciting sample proportions available
                on each node
            global_model (Mapping[str, Union[torch.tensor, np.ndarray]]): global model,
                ie aggregated model
            training_plan (BaseTrainingPlan): _description_
            node_ids (Iterable[str]): iterable containing node_id participating to the current round
            n_updates (int, optional): number of updates (number of batch performed). Defaults to 1.
            n_round (int, optional): current round. Defaults to 0.

        Returns:
            Dict: aggregated parameters
        """

    
        weights_processed = [list(weight.values())[0] for weight in weights] # same retrieving

        
        model_params_processed = self.scaling(model_params, global_model)
        model_params_processed = [list(model_param.values())[0] for model_param in model_params_processed] # model params are contained in a dictionary with node_id as key, we just retrieve the params
 
        #model_params_processed = list(model_params_processed.values())

        weights_processed = self.normalize_weights(weights_processed)

        aggregated_parameters = federated_averaging(model_params_processed, weights_processed)

        self.set_nodes_learning_rate_after_training(training_plan, training_replies, n_round)
        if n_round == 0:
            self.init_correction_states(global_model, node_ids)

        self.update_correction_states(aggregated_parameters, global_model,  node_ids, n_updates)
        
        self.update_aggregator_args(global_model)  # update aggregator_params (for breakpoints)
        return aggregated_parameters

    def create_aggregator_args(self,
                               global_model: Mapping[str, Union[torch.tensor, np.ndarray]],
                               node_ids: Iterator[str]) -> Tuple[Dict, Dict]:
        """Sends additional arguments for aggregator. For scaffold, it is mainly correction states

        Args:
            global_model (Mapping[str, Union[torch.tensor, np.ndarray]]): _description_
            node_ids (Iterator[str]): _description_

        Returns:
            Dict: _description_
        """
        if self.nodes_correction_states is None:
            self.init_correction_states(global_model, node_ids) 
        aggregator_args_thr_msg, aggregator_args_thr_file = {}, {}
        for node_id in node_ids:
            # serializing correction parameters
            # logger.critical("CORRECTION", self.nodes_correction_states)

            # print(self.nodes_correction_states)
            #serialized_aggregator_correction = {key: tensor.tolist() for key, tensor in self.nodes_correction_states[node_id].items()}
            aggregator_args_thr_file.update({node_id: {'aggregator_name': self.aggregator_name,
                                                       'aggregator_correction': self.nodes_correction_states[node_id]}})
            aggregator_args_thr_msg.update({node_id: {'aggregator_name': self.aggregator_name,
                                                      }})
            # if self._aggregator_args.get(node_id) is None:
            #     self._aggregator_args[node_id] = {}
        self._aggregator_args['aggregator_correction']= self.nodes_correction_states
        return aggregator_args_thr_msg, aggregator_args_thr_file

    def check_values(self, node_lrs: List[float], n_updates: int):
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
        if not node_lrs.any():
            raise FedbiomedAggregatorError(f"Learning rate(s) should be non-zero, but got {node_lrs} ")
        if n_updates == 0 or int(n_updates) != float(n_updates):
            raise FedbiomedAggregatorError(f"n_updates should be a non zero integer, but got n_updates: {n_updates} in SCAFFOLD aggregator")
        if self._fds is None:
            raise FedbiomedAggregatorError(" Federated Dataset not provided, but needed for Scaffold. Please use `set_fds()")

    def set_nodes_learning_rate_after_training(self, training_plan: BaseTrainingPlan, training_replies: List[Responses], n_round: int) -> Dict[str, List[float]]:
        # to be implemented in a utils module (for pytorch optimizers)

        n_model_layers = len(training_plan.get_model_params())
        for node_id in self._fds.node_ids():
            lrs: List[float] = []

            if training_replies[n_round].get_index_from_node_id(node_id) is not None:
                # get updated learning rate if provided...
                node_idx: int = training_replies[n_round].get_index_from_node_id(node_id)
                lrs += training_replies[n_round][node_idx]['optimizer_args'].get('lr')

            else:
                # ...otherwise retrieve default learning rate 
                lrs += training_plan.get_learning_rate()

            if len(lrs) == 1:
                # case where there is one learning rate
                lr = lrs * n_model_layers
                
            elif len(lrs) == n_model_layers:
                # case where there are several learning rates value
                lr = lrs
            else:

                raise FedbiomedAggregatorError("Error when setting node learning rate for SCAFFOLD: cannot extract node learning rate.")
            
            self.nodes_lr[node_id] = lr
        return self.nodes_lr

    def init_correction_states(self,
                               global_model: Mapping[str, Union[torch.tensor, np.ndarray]],
                               node_ids: Iterable[str],
                               ):
        # initialize nodes states

        init_params = {key:initialize(tensor)[1]for key, tensor in global_model.items()}
        self.nodes_correction_states = {node_id: deepcopy(init_params) for node_id in node_ids}
    
    def scaling(self,
                model_params: List[Dict[str, Mapping[str, Union[torch.Tensor, np.ndarray]]]],
                global_model: Mapping[str, Union[torch.tensor, np.ndarray]]) -> List[Dict[str, Mapping[str, Union[torch.Tensor, np.ndarray]]]]:
        """
        Computes quantity `x (1 - eta_g) + eta_g * y_i`
        Proof: 
            x <- x + eta_g * grad(x)
            x <- x + eta_g / S * sum_i(y_i - x)
            x <- x (1 - eta_g) + eta_g / S * sum_i(y_i)
            x <- sum_i(x (1 - eta_g) + eta_g * y_i) / S
            x <- avg(x (1 - eta_g) + eta_g * y_i) ... averaging is done afterwards, in aggregate method

        Args:
            model_params (list): _description_
            global_model (OrderedDict): _description_

        Returns:
            list: _description_
        """
        # refers as line 13 and 17 in pseudo code
        # should scale regading option
        for idx, model_param in enumerate(model_params):
            node_id = list(model_param.keys())[0] # retrieve node_id
            for layer in model_param[node_id]:
                model_params[idx][node_id][layer] = model_param[node_id][layer] * self.server_lr + (1 - self.server_lr) * global_model[layer]

        return copy.deepcopy(model_params)
        
    def update_correction_states(self, updated_model_params: Mapping[str, Union[torch.tensor, np.ndarray]],
                                 global_model: Mapping[str, Union[torch.tensor, np.ndarray]],
                                 node_ids: Iterable[str], n_updates: int = 1,):
        """_summary_
        
        Proof:
        
        c <- c + S/N grad(c)
        c <- c + 1/N sum_i(c_i(+) - c_i)
        c <- c + 1/N * sum_i( 1/ (K * eta_l)(x - y_i) - c)

        Args:
            updated_model_params (dict): _description_
            global_model (OrderedDict): _description_
            lr (float): _description_
            node_ids (Iterator[str]): Iterable of all node ids taking part in the round
            n_updates (int, optional): number of batches (or updates) performed during one round. Refers to `K` in
            Scaffold paper. Defaults to 1.
        """
        # refers as line 12, 13 and 17 in pseudo code
        if self._fds is None:
            raise FedbiomedAggregatorError("Cannot run SCAFFOLD aggregator: No Federated Dataset set")
        total_nb_nodes = len(self._fds.node_ids())  # get the total number of nodes
        
        weights = [1/total_nb_nodes] * len(node_ids)

        present_nodes_idx = list(range(len(self._fds.node_ids())))
        
        for idx, node_id in enumerate(self._fds.node_ids()):
            if node_id not in node_ids:
                present_nodes_idx.remove(idx)
        
        assert len(present_nodes_idx) == len(node_ids)
        # get weights for weighted summation
        _tmp_correction_update = []
        
        for idx, node_id in enumerate( node_ids):
            
            _tmp_correction_update.append({})
            #_tmp_correction_update[idx][node_id] = {}
            lrs: List[float] = self.nodes_lr[node_id]
            for idx_layer, (layer_name, node_layer) in enumerate(updated_model_params.items()): # iterate params of each client

                # `_tmp_correction_update`` is an intermediate variable equals to 1/ (K * eta_l)(x - y_i) - c

                _tmp_correction_update[idx][layer_name] = (global_model[layer_name] - node_layer) / (self.server_lr * lrs[idx_layer] * n_updates)
                # FIXME: check why we need server learning_rate in above formulae
                _tmp_correction_update[idx][layer_name] = _tmp_correction_update[idx][layer_name] - self.nodes_correction_states[node_id][layer_name]

        _aggregated_tmp_correction_update = weitghted_sum(_tmp_correction_update, weights)
        
        # finally, perform `c <- c + S/N \Delta{c}`
        for node_id in self._fds.node_ids():
            for layer_name, node_layer in updated_model_params.items(): 

                self.nodes_correction_states[node_id][layer_name] += _aggregated_tmp_correction_update[layer_name]


    def set_training_plan_type(self, training_plan_type: TrainingPlans) -> TrainingPlans:
        """
        Overrides `set_training_plan_type` from parent class. 
        Checks if trainning plan type, and if it is SKlearnTrainingPlan,
        raises an error. Otherwise, calls parent method.

        Args:
            training_plan_type (TrainingPlans): _description_

        Raises:
            FedbiomedAggregatorError: _description_

        Returns:
            TrainingPlans: _description_
        """
        if training_plan_type == TrainingPlans.SkLearnTrainingPlan:
            raise FedbiomedAggregatorError("Aggregator SCAFFOLD not implemented for SKlearn")
        training_plan_type = super().set_training_plan_type(training_plan_type)
        
        # TODO: trigger a warning if user is trying to use scaffold with something else than SGD
        return training_plan_type

    def update_aggregator_args(self,
                                 global_model: Mapping[str, Union[torch.tensor, np.ndarray]],
                                 ):
        aggregator_args_msg, aggregator_args_file = self.create_aggregator_args(global_model, self._fds.node_ids())
        self._aggregator_args.update({'name': self.aggregator_name,
                                        'server_lr': self.server_lr})
    
    def save_state(self, training_plan: BaseTrainingPlan, breakpoint_path: str, global_model: Mapping[str, Union[torch.tensor, np.ndarray]]) -> Dict[str, Any]:
        self.update_aggregator_args(global_model)
        
        return super().save_state(training_plan, breakpoint_path, ['aggregator_correction'])       
        
    def load_state(self, state: Dict[str, Any] = None, training_plan: BaseTrainingPlan = None):
        super().load_state(state)
        self.server_lr = self._aggregator_args['server_lr']
        
        self.nodes_correction_states = {}
        for node_id in self._aggregator_args['aggregator_correction'].keys():
            arg_filename = self._aggregator_args['aggregator_correction'][node_id]
             
            self.nodes_correction_states[node_id] = training_plan.load(arg_filename)
            #self.nodes_correction_states[node_id].pop('aggregator_name')
            

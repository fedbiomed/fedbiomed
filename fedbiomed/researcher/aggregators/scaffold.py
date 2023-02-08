# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Scaffold Aggregator."""

import copy
import os
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union
import uuid

import numpy as np
import torch
from fedbiomed.common.logger import logger
from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.common.training_plans import BaseTrainingPlan

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import initialize
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.responses import Responses


class Scaffold(Aggregator):
    """
    Defines the Scaffold strategy

    Despite being an algorithm of choice for federated learning, it is observed that FedAvg
    suffers from `client-drift` when the data is heterogeneous (non-iid), resulting in unstable and slow
    convergence. SCAFFOLD uses control variates (variance reduction) to correct for the `client-drift` in its local
    updates.
    Intuitively, SCAFFOLD estimates the update direction for the server model (c) and the update direction for each
    client (c_i).
    The difference (c - c_i) is then an estimate of the client-drift which is used to correct the local update.

    Fed-BioMed implementation details
    ---
    Our implementation is heavily influenced by our design choice to prevent storing any state on the nodes between
    FL rounds. In particular, this means that the computation of the control variates (i.e. the correction states)
    needs to be performed centrally by the aggregator.
    Roughly, our implementation follows these steps (following the notation of the original Scaffold
    [paper](https://arxiv.org/abs/1910.06378)):

    0. let \(\delta_i = \mathbf{c} - \mathbf{c}_i \)
    1. foreach(round):
    2. sample \( S \) nodes participating in this round out of \( N \) total
    3. the server communicates the global model \( \mathbf{x} \) and the correction states \( \delta_i \) to all clients
    4. parallel on each client
    5. initialize local model \( \mathbf{y}_i = \mathbf{x} \)
    6. foreach(update) until K updates have been performed
    7. obtain a data batch
    8. compute the gradients for this batch \( g(\mathbf{y}_i) \)
    9. add correction term to gradients \( g(\mathbf{y}_i) += \delta_i \)
    10. update model with one optimizer step \( \mathbf{y}_i += \eta_i g(\mathbf{y}_i) \)
    11. end foreach(update)
    12. communicate updated model \( \mathbf{y}_i \) and learning rate \( \eta_i \)
    13. end parallel section on each client
    14. the server computes the node-wise average of corrected gradients \( \mathbf{ACG}_i = (\mathbf{x} - \mathbf{y}_i)/(\eta_iK) \)
    15. the server updates the global correction term \( \mathbf{c} = (1 - S/N)\mathbf{c} + 1/N\sum_{i \in S}\mathbf{ACG}_i \)
    16. the server updates the correction states of each client \(\delta_i = \mathbf{ACG}_i - \mathbf{c} - \delta_i \)
    17. the server updates the global model by average \( \mathbf{x} = (1-\eta)\mathbf{x} + \eta/S\sum_{i \in S} \mathbf{y}_i \)
    18. end foreach(round)

    References:

    - [Scaffold: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378)
    - [TCT: Convexifying Federated Learning using Bootstrapped Neural
    Tangent Kernels](https://arxiv.org/pdf/2207.06343.pdf)

    Attributes:
     aggregator_name: name of the aggregator
     server_lr: value of the server learning rate
     nodes_correction_states: a nested dictionary
        of correction parameters obtained for each client, in the format {node id: node-wise corrections}. The
        node-wise corrections are a dictionary in the format {parameter name: correction value} where the
        model parameters are those contained in each node's model.named_parameters().
    """

    def __init__(self, server_lr: float = 1., fds: Optional[FederatedDataSet] = None):
        """Constructs `Scaffold` object as an instance of [`Aggregator`]
        [fedbiomed.researcher.aggregators.Aggregator].

        Args:
            server_lr (float): server's (or Researcher's) learning rate. Defaults to 1..
            fds (FederatedDataset, optional): FederatedDataset obtained after a `search` request. Defaults to None.

        """
        super().__init__()
        self.aggregator_name: str = "Scaffold"
        if server_lr == 0.:
            raise FedbiomedAggregatorError("SCAFFOLD Error: Server learning rate cannot be equal to 0")
        self.server_lr: float = server_lr
        self.nodes_correction_states: Dict[str, Mapping[str, Union[torch.Tensor, np.ndarray]]] = {}
        self.global_state: Mapping[str, Union[torch.Tensor, np.ndarray]] = {}

        self.nodes_lr: Dict[str, List[float]] = {}
        if fds is not None:
            self.set_fds(fds)
        
        self._aggregator_args = {}  # we need `_aggregator_args` to be not None
        #self.update_aggregator_params()FedbiomedAggregatorError:

    def aggregate(self,
                  model_params: Dict,
                  weights: Dict[str, float],
                  global_model: Mapping[str, Union[torch.Tensor, np.ndarray]],
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

        c_i(+) <- c_i - c + 1/(K*eta_l)(x - y_i)
        c <- c + 1/N * sum_S(c_i(+) - c_i)

        x <- x + eta_g/S * sum_S(y_i - x)

        where, according to paper notations
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
            model_params: list of models parameters received from nodes
            weights: weights depicting sample proportions available
                on each node. Unused for Scaffold.
            global_model: global model, ie aggregated model
            training_plan (BaseTrainingPlan): instance of TrainingPlan
            training_replies: Training replies from each node that participates in the current round
            node_ids: iterable containing node_id (string) participating in the current round.
            n_updates: number of updates (number of batch performed). Defaults to 1.
            n_round: current round. Defaults to 0.

        Returns:
            Dict: aggregated parameters, ie mapping of layer names and layer values.
        """

        # Gather the learning rates used by nodes, updating `self.nodes_lr`.
        self.set_nodes_learning_rate_after_training(training_plan, training_replies, n_round)

        # Compute the new aggregated model parameters.
        aggregated_parameters = self.scaling(model_params, global_model)
        
        # At round 0, initialize zero-valued correction states.
        if n_round == 0:
            self.init_correction_states(global_model, node_ids)
        # Update correction states.
        self.update_correction_states(model_params, global_model, n_updates)
        # Return aggregated parameters.
        return aggregated_parameters

    def create_aggregator_args(self,
                               global_model: Mapping[str, Union[torch.Tensor, np.ndarray]],
                               node_ids: Iterator[str]) -> Tuple[Dict, Dict]:
        """Sends additional arguments for aggregator. For scaffold, it is mainly correction states

        Args:
            global_model (Mapping[str, Union[torch.Tensor, np.ndarray]]): aggregated model
            node_ids (Iterator[str]): iterable that contains strings of nodes id that have participated in
                the round

        Returns:
            Tuple[Dict, Dict]: first dictionary contains parameters that will be sent through MQTT message
                service, second dictionary parameters that will be sent through file exchange message.
                Aggregators args are dictionary mapping node_id to SCAFFOLD parameters specific to 
                each `Nodes`.
        """

        if not self.nodes_correction_states:
            self.init_correction_states(global_model, node_ids)
        aggregator_args_thr_msg, aggregator_args_thr_file = {}, {}
        for node_id in node_ids:
            # in case of a new node, initialize its correction state
            if node_id not in self.nodes_correction_states:
                self.nodes_correction_states[node_id] = {
                    key: copy.deepcopy(initialize(tensor))[1] for key, tensor in global_model.items()
                }
            # pack information and parameters to send
            aggregator_args_thr_file[node_id] = {
                'aggregator_name': self.aggregator_name,
                'aggregator_correction': self.nodes_correction_states[node_id]
            }
            aggregator_args_thr_msg[node_id] = {
                'aggregator_name': self.aggregator_name
            }
        return aggregator_args_thr_msg, aggregator_args_thr_file

    def check_values(self, n_updates: int, training_plan: BaseTrainingPlan) -> True:
        """
        This method checks if all values/parameters are correct and have been set before using aggregator.
        Raises error otherwise
        This can prove useful if user has set wrong hyperparameter values, so that user will
        have errors before performing first round of training

        Args:

            n_updates (int): number of updates. Must be non-zero and an integer.
            training_plan (BaseTrainingPlan): training plan. used for checking if optimizer is SGD, otherwise, 
                triggers warning.

        Raises:
            FedbiomedAggregatorError: triggered if `num_updates` entry is missing (needed for Scaffold aggregator)
            FedbiomedAggregatorError: triggered if any of the learning rate(s) equals 0
            FedbiomedAggregatorError: triggered if number of updates equals 0 or is not an integer
            FedbiomedAggregatorError: triggered if [FederatedDataset][fedbiomed.researcher.datasets.FederatedDataset]
                has not been set.
             
        """
        if n_updates is None:
            raise FedbiomedAggregatorError("Cannot perform Scaffold: missing 'num_updates' entry in the training_args")
        elif n_updates <= 0 or int(n_updates) != float(n_updates):
            raise FedbiomedAggregatorError(f"n_updates should be a positive non zero integer, but got n_updates: {n_updates} in SCAFFOLD aggregator")
        if self._fds is None:
            raise FedbiomedAggregatorError(" Federated Dataset not provided, but needed for Scaffold. Please use setter `set_fds()`")
        if hasattr(training_plan, "_optimizer") and training_plan.type() is TrainingPlans.TorchTrainingPlan:
            if not isinstance(training_plan._optimizer, torch.optim.SGD):
                logger.warning(f"Found optimizer {training_plan._optimizer}, but SCAFFOLD requieres SGD optimizer. Results may be inconsistants")

        return True

    def set_nodes_learning_rate_after_training(self, training_plan: BaseTrainingPlan,
                                               training_replies: List[Responses],
                                               n_round: int) -> Dict[str, List[float]]:
        """Gets back learning rate of optimizer from Node (if learning rate scheduler is used)

        Args:
            training_plan (BaseTrainingPlan): training plan instance
            training_replies (List[Responses]): training replies that must contain am `optimizer_args`
                entry and a learning rate
            n_round (int): number of rounds already performed

        Raises:
            FedbiomedAggregatorError: raised when setting learning rate has been unsuccessful

        Returns:
            Dict[str, List[float]]: dictionary mapping node_id and a list of float, as many as
                the number of layers contained in the model (in Pytroch, each layer can have a specific learning rate).
        """
        # to be implemented in a utMapping[str, Union[np.ndarray, torch.Tensor]]ils module (for pytorch optimizers)

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
                               global_model: Mapping[str, Union[torch.Tensor, np.ndarray]],
                               node_ids: Iterable[str],
                               ):
        """Initialises correction_states variable for Scaffold

        Args:
            global_model (Mapping[str, Union[torch.Tensor, np.ndarray]]): global model mapping layer name to model
            parameters
            node_ids (Iterable[str]): iterable containing node_ids
        """
        # initialize nodes states with zeros tensors
        init_params = {key: initialize(tensor)[1] for key, tensor in global_model.items()}
        self.nodes_correction_states = {node_id: copy.deepcopy(init_params) for node_id in node_ids}
        self.global_state = init_params

    def scaling(self,
                model_params: Dict[str, Mapping[str, Union[np.ndarray, torch.Tensor]]],
                global_model: Mapping[str, Union[np.ndarray, torch.Tensor]]
                ) -> Mapping[str, Union[np.ndarray, torch.Tensor]]:
        """Computes the aggregated model.

        Let

            - x = the global model from the previous aggregation round
            - y_i = the local model after training for the i^th node
            - eta_g = the global learning rate

        Then this function computes the quantity `x (1 - eta_g) + eta_g / S * sum_i(y_i))`
        Proof:
            x <- x + eta_g * grad(x)
            x <- x + eta_g / S * sum_i(y_i - x)
            x <- x (1 - eta_g) + eta_g / S * sum_i(y_i)

        Args:
            model_params: dictionary of model parameters obtained after one round of federated training,
                in the format {node id: {parameter name: parameter value}}.
            global_model: dictionary representing the previous iteration of the global model,
                in the format {parameter name: parameter value}. This corresponds to $\mathbf{x}$ in the notation
                of the scaffold paper.

        Returns:
            A dictionary of aggregated parameters, in the format {parameter name: parameter value}, where the
                parameter names are the same as those of the input global models
        """
        aggregated_parameters = {}
        for key, val in global_model.items():
            update = sum(params[key] for params in model_params.values()) / len(model_params)
            newval = (1 - self.server_lr) * val + self.server_lr * update

            aggregated_parameters[key] = newval

        return aggregated_parameters

    def update_correction_states(self,
                                 local_models: Dict[str, Mapping[str, Union[torch.Tensor, np.ndarray]]],
                                 global_model: Mapping[str, Union[torch.Tensor, np.ndarray]],
                                 n_updates: int = 1,) -> None:
        """Updates correction states

        Proof:

        c <- c + S/N grad(c)
        c <- c + 1/N sum_i(c_i(+) - c_i)
        c <- c + 1/N * sum_i( 1/ (K * eta_l)(x - y_i) - c)
        c <- (1 - S/N) c + ACG_i , where ACG_i = sum_i( 1/ (K * eta_l)(x - y_i))

        where (according to Scaffold paper):
        c: is the correction term
        S: the number of nodes participating in the current round
        N: the total number of node participating in the experiment
        K: number of updates
        eta_l: nodes' learning rate
        x: global model before updates
        y_i: local model updates
        
        Remark: 
        c^{t=0} = 0

        Args:
            local_models: Node-wise local model parameters after updates, as
                as {name: value} parameters mappings indexed by node id.
            global_model: Global model parameters (before updates), as a single
                {name: value} parameters mapping.
            n_updates: number of batches (or updates) performed during one round
                Referred to as `K` in the Scaffold paper. Defaults to 1.

        Raises:
            FedbiomedAggregatorError: if no FederatedDataset has been found.
        """
        # Gather the total number of nodes (not just participating ones).
        if self._fds is None:
            raise FedbiomedAggregatorError("Cannot run SCAFFOLD aggregator: No Federated Dataset set")
        total_nb_nodes = len(self._fds.node_ids())
        # Compute the node-wise average of corrected gradients (ACG_i).
        # i.e. (x^t - y_i^t}) / (K * eta_l)
        local_state_updates: Dict[str, Mapping[str, Union[torch.Tensor, np.ndarray]]] = {} 
        for node_id, params in local_models.items():
            local_state_updates[node_id] = {
                key: (global_model[key] - val) / (self.nodes_lr[node_id][idx] * n_updates)
                for idx, (key, val) in enumerate(params.items())
            }
        # Compute the shared state variable's update by averaging the former.
        global_state_update = {
            key: sum(state[key] for state in local_state_updates.values()) / total_nb_nodes
            for key in global_model
        }
        # Compute the updated shared state variable.
        # c^{t+1} = (1 - S/N)c^t + (1/N) sum_{i=1}^S ACG_i
        share = 1 - len(local_models) / total_nb_nodes
        global_state_new = {
            key: share * self.global_state[key] + val
            for key, val in global_state_update.items()
        }
        # Compute the difference between past and new shared state variables
        # (ie c^t−c^{t+1} ).
        global_state_diff = {
            key: self.global_state[key] - val
            for key, val in global_state_new.items()
        }
        # Compute the updated node-wise correction terms.
        for node_id in self._fds.node_ids():
            acg = local_state_updates.get(node_id, None)
            # Case when the node did not participate in the round.
            # d_i^{t+1} = d_i^t + c^t - c^{t+1}
            if acg is None:
                for key, val in self.nodes_correction_states[node_id].items():
                    self.nodes_correction_states[node_id][key] += global_state_diff[key]
            # Case when the node participated in the round
            # d_i^{t+1} = c_i^{t+1} - c^{t+1} = ACG_i - d_i^{t} - c^{t+1}
            else:
                for key, val in self.nodes_correction_states[node_id].items():
                    self.nodes_correction_states[node_id][key] = (
                        local_state_updates[node_id][key] - val - global_state_new[key]
                    )
        # Assign the updated shared state.
        self.global_state = global_state_new

    def set_training_plan_type(self, training_plan_type: TrainingPlans) -> TrainingPlans:
        """
        Overrides `set_training_plan_type` from parent class.
        Checks the training plan type, and if it is SKlearnTrainingPlan,
        raises an error. Otherwise, calls parent method.

        Args:
            training_plan_type (TrainingPlans): training_plan type

        Raises:
            FedbiomedAggregatorError: raised if training_plan type has been set to SKLearn training plan

        Returns:
            TrainingPlans: training plan type
        """
        if training_plan_type == TrainingPlans.SkLearnTrainingPlan:
            raise FedbiomedAggregatorError("Aggregator SCAFFOLD not implemented for SKlearn")
        training_plan_type = super().set_training_plan_type(training_plan_type)

        # TODO: trigger a warning if user is trying to use scaffold with something else than SGD
        return training_plan_type

    def save_state(self, training_plan: BaseTrainingPlan,
                   breakpoint_path: str,
                   global_model: Mapping[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
        # adding aggregator parameters to the breakpoint that wont be sent to nodes
        self._aggregator_args['server_lr'] = self.server_lr
        
        # saving global state variable into a file
        filename = os.path.join(breakpoint_path, 'global_state_' + str(uuid.uuid4()) + '.pt')
        training_plan.save(filename, self.global_state)
        self._aggregator_args['global_state_filename'] = filename
        # adding aggregator parameters that will be sent to nodes afterwards
        return super().save_state(training_plan,
                                  breakpoint_path,
                                  global_model=global_model,
                                  node_ids=self._fds.node_ids())

    def load_state(self, state: Dict[str, Any] = None, training_plan: BaseTrainingPlan = None):
        super().load_state(state)

        self.server_lr = self._aggregator_args['server_lr']

        # loading global state
        global_state_filename = self._aggregator_args['global_state_filename']
        self.global_state = training_plan.load(global_state_filename, to_params=True)

        for node_id in self._aggregator_args['aggregator_correction'].keys():
            arg_filename = self._aggregator_args['aggregator_correction'][node_id]

            self.nodes_correction_states[node_id] = training_plan.load(arg_filename)

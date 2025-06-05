# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Scaffold Aggregator."""

import copy
import os
import uuid
from typing import Any, Dict, Collection, List, Mapping, Optional, Tuple, Union

import numpy as np
from fedbiomed.common.optimizers.generic_optimizers import NativeTorchOptimizer
import torch

from fedbiomed.common.logger import logger
from fedbiomed.common.constants import TrainingPlans
from fedbiomed.common.exceptions import FedbiomedAggregatorError
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.training_plans import BaseTrainingPlan

from fedbiomed.researcher.aggregators.aggregator import Aggregator
from fedbiomed.researcher.aggregators.functional import initialize
from fedbiomed.researcher.datasets import FederatedDataSet


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

    0. let \(\delta_i = \mathbf{c}_i - \mathbf{c} \)
    1. foreach(round):
    2. sample \( S \) nodes participating in this round out of \( N \) total
    3. the server communicates the global model \( \mathbf{x} \) and the correction states \( \delta_i \) to all clients
    4. parallel on each client
    5. initialize local model \( \mathbf{y}_i = \mathbf{x} \)
    6. foreach(update) until K updates have been performed
    7. obtain a data batch
    8. compute the gradients for this batch \( g(\mathbf{y}_i) \)
    9. apply correction term to gradients \( g(\mathbf{y}_i) -= \delta_i \)
    10. update model with one optimizer step e.g. for SGD \( \mathbf{y}_i -= \eta_i g(\mathbf{y}_i) \)
    11. end foreach(update)
    12. communicate updated model \( \mathbf{y}_i \) and learning rate \( \eta_i \)
    13. end parallel section on each client
    14. the server computes the node-wise model update \( \mathbf{\Delta y}_i =  \mathbf{x} - \mathbf{y}_i \)
    15. the server updates the node-wise states \( \mathbf{c}_i = \delta_i + (\mathbf{\Delta y}_i) / (\eta_i K) \)
    16. the server updates the global state \( \mathbf{c} = (1/N) \sum_{i \in N} \mathbf{c}_i \)
    17. the server updates the node-wise correction state \(\delta_i = \mathbf{c}_i - \mathbf{c} \)
    18. the server updates the global model by averaging \( \mathbf{x} = \mathbf{x} - (\eta/|S|) \sum_{i \in S} \mathbf{\Delta y}_i \)
    19. end foreach(round)

    This [diagram](http://www.plantuml.com/plantuml/dsvg/xLRDJjmm4BxxAUR82fPbWOe2guYsKYyhSQ6SgghosXDYuTYMFIbjdxxE3r7MIac3UkH4i6Vy_SpCsZU1kAUgrApvOEofK1hX8BSUkIZ0syf88506riV7NnQCNGLUkXXojmcLosYpgl-0YybAACT9cGSmLc80Mn7O7BZMSDikNSTqOSkoCafmGdZGTiSrb75F0pUoYLe6XqBbIe2mtgCWPGqG-f9jTjdc_l3axEFxRBEAtmC2Hz3kdDUhkqpLg_iH4JlNzfaV8MZCwMeo3IJcog047Y3YYmvuF7RPXmoN8x3rZr6wCef0Mz5B7WXwyTmOTBg-FCcIX4HVMhlAoThanwvusqNhlgjgvpsN2Wr130OgL80T9r4qIASd5zaaiwF77lQAEwT_fTK2iZrAO7FEJJNFJbr27tl-eh4r-SwbjY1FYWgm1i4wKgNwZHu2eGFs3-27wvJv7CPjuCLUq6kAWKPsRS1pGW_RhWt28fczN9czqTF8lQc7myVTQRslKRljKYBSgDxhTbA0Ft1btkPbwjotUNcRbqY_krm-TPrA1RRNw9CA-2o6DUcNvzd_u9bUU9C7zhrpNxCPq1lCGAWj5BCuJVSh7C9iuQk3CQjXknW8eA9_koHJF50nplnWlRfTD0WVpZg4vh_FxxBR5ch_X57pGA8c7jY43MFuKoudhvYqWdL3fI-tfFbVsKYzxQkxl_XprxATLz69br_40nMQWWRqFz1_rvunjlnQA2dHV5jc340YSL54zMXa-o8U_72y58i_7NfLeg5h5iWwTXDNgrB_0G00)
    provides a visual representation of the algorithm.

    References:

    - [Scaffold: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378)
    - [TCT: Convexifying Federated Learning using Bootstrapped Neural
    Tangent Kernels](https://arxiv.org/pdf/2207.06343.pdf)

    Attributes:
     aggregator_name: name of the aggregator
     server_lr: value of the server learning rate
     global_state: a dictionary representing the global correction state \( \mathbf{c} \) in the format
        {parameter name: correction value}
     nodes_states: a nested dictionary
        of correction parameters obtained for each client, in the format {node id: node-wise corrections}. The
        node-wise corrections are a dictionary in the format {parameter name: correction value} where the
        model parameters are those contained in each node's model.named_parameters().
     nodes_deltas: a nested dictionary of deltas for each client, in the same format as nodes_states. The deltas
        are defined as \(\delta_i = \mathbf{c}_i - \mathbf{c} \)
     nodes_lr: dictionary of learning rates observed at end of the latest round, in the format
        {node id: learning rate}
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
        self.global_state: Dict[str, Union[torch.Tensor, np.ndarray]] = {}
        self.nodes_states: Dict[str, Dict[str, Union[torch.Tensor, np.ndarray]]] = {}
        # FIXME: `nodes_states` is mis-named, because can conflict with `node_state`s that are saved 
        # whitin 2 Rounds
        self.nodes_deltas: Dict[str, Dict[str, Union[torch.Tensor, np.ndarray]]] = {}
        self.nodes_lr: Dict[str, Dict[str, float]] = {}
        if fds is not None:
            self.set_fds(fds)
        self._aggregator_args = {}  # we need `_aggregator_args` to be not None

    def aggregate(self,
                  model_params: Dict,
                  weights: Dict[str, float],
                  global_model: Dict[str, Union[torch.Tensor, np.ndarray]],
                  training_plan: BaseTrainingPlan,
                  training_replies: Dict,
                  n_updates: int = 1,
                  n_round: int = 0,
                  *args, **kwargs) -> Dict:
        """
        Aggregates local models coming from nodes into a global model, using SCAFFOLD algorithm (2nd option)
        [Scaffold: Stochastic Controlled Averaging for Federated Learning][https://arxiv.org/abs/1910.06378]

        Performed computations:
        -----------------------

        - Compute participating nodes' model update:
            * update_i = y_i - x
        - Compute aggregated model parameters:
            * x(+) = x - eta_g sum_S(update_i)
        - Update participating nodes' state:
            * c_i = delta_i + 1/(K*eta_i) * update_i
        - Update the global state and all nodes' correction state:
            * c = 1/N sum_{i=1}^n c_i
            * delta_i = (c_i - c)

        where, according to paper notations
            c_i: local state variable for node `i`
            c: global state variable
            delta_i: (c_i - c), correction state for node `i`
            eta_g: server's learning rate
            eta_i: node i's learning rate
            N: total number of node participating to federated learning
            S: number of nodes considered during current round (S<=N)
            K: number of updates done during the round (ie number of data batches).
            x: global model parameters
            y_i: node i 's local model parameters at the end of the round

        Args:
            model_params: list of models parameters received from nodes
            weights: weights depicting sample proportions available
                on each node. Unused for Scaffold.
            global_model: global model, ie aggregated model
            training_plan (BaseTrainingPlan): instance of TrainingPlan
            training_replies: Training replies from each node that participates in the current round
            n_updates: number of updates (number of batch performed). Defaults to 1.
            n_round: current round. Defaults to 0.

        Returns:
            Aggregated parameters, as a dict mapping weight names and values.

        Raises:
            FedbiomedAggregatorError: If no FederatedDataset is attached to this
                Scaffold instance, or if `node_ids` do not belong to the dataset
                attached to it.
        """
        # Gather the learning rates used by nodes, updating `self.nodes_lr`.
        self.set_nodes_learning_rate_after_training(training_plan, training_replies)
        # At round 0, initialize zero-valued correction states.
        if n_round == 0:
            self.init_correction_states(global_model)
        # Check that the input node_ids match known ones.
        if not set(model_params).issubset(self._fds.node_ids()):
            raise FedbiomedAggregatorError(
                "Received updates from nodes that are unknown to this aggregator."
            )
        # Compute the node-wise model update: (x^t - y_i^t).
        model_updates = {
            node_id: {
                key: (global_model[key] - local_value)
                for key, local_value in params.items()
            }
            for node_id, params in model_params.items()
        }
        # Update all Scaffold state variables.
        self.update_correction_states(model_updates, n_updates)
        # Compute and return the aggregated model parameters.
        global_new = {}  # type: Dict[str, Union[torch.Tensor, np.ndarray]]
        for key, val in global_model.items():
            upd = sum(model_updates[node_id][key] for node_id in model_params)
            global_new[key] = val - upd * (self.server_lr / len(model_params))
        return global_new

    def init_correction_states(
        self,
        global_model: Dict[str, Union[torch.Tensor, np.ndarray]],
    ) -> None:
        """Initialize Scaffold state variables.

        Args:
            global_model: parameters of the global model, formatted as a dict
                mapping weight tensors to their names.

        Raises:
            FedbiomedAggregatorError: if no FederatedDataset is attached to
                this aggregator.
        """
        # Gather node ids from the attached FederatedDataset.
        if self._fds is None:
            raise FedbiomedAggregatorError(
                "Cannot initialize correction states: Scaffold aggregator does "
                "not have a FederatedDataset attached."
            )
        node_ids = self._fds.node_ids()
        # Initialize nodes states with zero scalars, that will be summed into actual tensors.
        init_params = {key: initialize(tensor)[1] for key, tensor in global_model.items()}
        self.nodes_deltas = {node_id: copy.deepcopy(init_params) for node_id in node_ids}
        self.nodes_states = copy.deepcopy(self.nodes_deltas)
        self.global_state = init_params

    def update_correction_states(
        self,
        model_updates: Dict[str, Dict[str, Union[np.ndarray, torch.Tensor]]],
        n_updates: int,
    ) -> None:
        """Update all Scaffold state variables based on node-wise model updates.

        Performed computations:
        ----------------------

        - Update participating nodes' state:
            * c_i = delta_i + 1/(K*eta_i) * update_i
        - Update the global state and all nodes' correction state:
            * c = 1/N sum_{i=1}^n c_i
            * delta_i = (c_i - c)

        Args:
            model_updates: node-wise model weight updates.
            n_updates: number of local optimization steps.
        """
        # Update the node-wise states for participating nodes:
        # c_i^{t+1} = delta_i^t + (x^t - y_i^t) / (M * eta)
        for node_id, updates in model_updates.items():
            d_i = self.nodes_deltas[node_id]
            for (key, val) in updates.items():
                if self.nodes_lr[node_id].get(key) is not None:
                    self.nodes_states[node_id].update(
                        {
                        key: d_i[key] + val / (self.nodes_lr[node_id][key] * n_updates)
                         }
                    )
        # Update the global state: c^{t+1} = average(c_i^{t+1})
        for key in self.global_state:
            self.global_state[key] = 0
            for state in self.nodes_states.values():
                if state.get(key) is not None:
                    self.global_state[key] = (
                        sum(state[key] for state in self.nodes_states.values())
                            / len(self.nodes_states)
                        )

        # Compute the new node-wise correction states:
        # delta_i^{t+1} = c_i^{t+1} - c^{t+1}
        self.nodes_deltas = {
            node_id: {
                key: val - self.global_state[key] for key, val in state.items()
            }
            for node_id, state in self.nodes_states.items()
        }

    def create_aggregator_args(
        self,
        global_model: Dict[str, Union[torch.Tensor, np.ndarray]],
        node_ids: Collection[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Return correction states that are to be sent to the nodes.

        Args:
            global_model: parameters of the global model, formatted as a dict
                mapping weight tensors to their names.
            node_ids: identifiers of the nodes that are to receive messages.

        Returns:
            Aggregator arguments to share with the nodes for the next round
        """
        # Optionally initialize states, and verify that nodes are known.
        if not self.nodes_deltas:
            self.init_correction_states(global_model)
        if not set(node_ids).issubset(self._fds.node_ids()):
            raise FedbiomedAggregatorError(
                "Scaffold cannot create aggregator args for nodes that are not"
                "covered by its attached FederatedDataset."
            )

        aggregator_dat = {}
        for node_id in node_ids:
            # If a node was late-added to the FederatedDataset, create states.
            if node_id not in self.nodes_deltas:
                zeros = {key: initialize(val)[1] for key, val in self.global_state.items()}
                self.nodes_deltas[node_id] = zeros
                self.nodes_states[node_id] = copy.deepcopy(zeros)
            # Add information for the current node to the message dicts.
            aggregator_dat[node_id] = {
                'aggregator_name': self.aggregator_name,
                'aggregator_correction': self.nodes_deltas[node_id]
            }
 
        return aggregator_dat

    def check_values(self, n_updates: int, training_plan: BaseTrainingPlan, *args, **kwargs) -> True:
        """Check if all values/parameters are correct and have been set before using aggregator.

        Raise an error otherwise.

        This can prove useful if user has set wrong hyperparameter values, so that user will
        have errors before performing first round of training

        Args:
            n_updates: number of updates. Must be non-zero and an integer.
            training_plan: training plan. used for checking if optimizer is SGD, otherwise,
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
            raise FedbiomedAggregatorError(
                "n_updates should be a positive non zero integer, but got "
                f"n_updates: {n_updates} in SCAFFOLD aggregator"
            )
        if self._fds is None:
            raise FedbiomedAggregatorError(
                "Federated Dataset not provided, but needed for Scaffold. Please use setter `set_fds()`."
            )

        # raise warnings
        if kwargs.get('secagg'):
            logger.warning("Warning: secagg setting detected. Nodes correction states involved in Scaffold algorithm"
                           " are not encrypted and can be read by the `Researcher`.\n"
                           "Please consider using `declearn`'s Scaffold to encrypt both model parameters and correction"
                           " terms")
        return True

    def set_nodes_learning_rate_after_training(
        self,
        training_plan: BaseTrainingPlan,
        training_replies: Dict,
    ) -> Dict[str, List[float]]:
        """Gets back learning rate of optimizer from Node (if learning rate scheduler is used)

        Args:
            training_plan: training plan instance
            training_replies: training replies that must contain am `optimizer_args`
                entry and a learning rate

        Raises:
            FedbiomedAggregatorError: raised when setting learning rate has been unsuccessful

        Returns:
            Dict[str, List[float]]: dictionary mapping node_id and a list of float, as many as
                the number of layers contained in the model (in Pytroch, each layer can have a specific learning rate).
        """

        n_model_layers = len(training_plan.get_model_params(
            only_trainable=False,
            exclude_buffers=True)
        )
        for node_id in self._fds.node_ids():
            lrs: Dict[str, float] = {}

            node = training_replies.get(node_id, None)
            if node is not None:
                lrs = training_replies[node_id]["optimizer_args"].get('lr')

            if node is None or lrs is None:
                # fall back to default value if no lr information was provided
                lrs = training_plan.optimizer().get_learning_rate()

            if len(lrs) != n_model_layers:
                raise FedbiomedAggregatorError(
                    "Error when setting node learning rate for SCAFFOLD: cannot extract node learning rate."
                )

            self.nodes_lr[node_id] = lrs
        return self.nodes_lr

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

    def save_state_breakpoint(
        self,
        breakpoint_path: str,
        global_model: Mapping[str, Union[torch.Tensor, np.ndarray]]
    ) -> Dict[str, Any]:
        # adding aggregator parameters to the breakpoint that wont be sent to nodes
        self._aggregator_args['server_lr'] = self.server_lr

        # saving global state variable into a file
        filename = os.path.join(breakpoint_path, f"global_state_{uuid.uuid4()}.mpk")
        Serializer.dump(self.global_state, filename)
        self._aggregator_args['global_state_filename'] = filename

        self._aggregator_args["nodes"] = self._fds.node_ids()
        # adding aggregator parameters that will be sent to nodes afterwards
        return super().save_state_breakpoint(
            breakpoint_path, global_model=global_model, node_ids=self._fds.node_ids()
        )

    def load_state_breakpoint(self, state: Dict[str, Any] = None):
        super().load_state_breakpoint(state)

        self.server_lr = self._aggregator_args['server_lr']

        # loading global state
        global_state_filename = self._aggregator_args['global_state_filename']
        self.global_state = Serializer.load(global_state_filename)

        for node_id in self._aggregator_args['nodes']:
            self.nodes_deltas[node_id] = self._aggregator_args[node_id]['aggregator_correction']

        self.nodes_states = copy.deepcopy(self.nodes_deltas)

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from fedbiomed.common.message import ErrorMessage, TrainReply, TrainRequest
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.common.logger import logger
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.requests import MessagesByNode

from ._job import Job


class TrainingJob(Job):
    """
    TrainingJob is a task for training an ML model on the nodes by executing a
    [TrainingPlan][fedbiomed.common.training_plans.BaseTrainingPlan].
    """

    def __init__(
        self,
        experiment_id: str,
        round_: int,
        training_plan: BaseTrainingPlan,
        training_args: Union[dict, TrainingArgs],
        model_args: Optional[dict],
        data: FederatedDataSet,
        nodes_state_ids: Dict[str, str],
        aggregator_args: Dict[str, Dict[str, Any]],
        secagg_arguments: Union[Dict, None] = None,
        do_training: bool = True,
        optim_aux_var: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ):

        """ Constructor of the class

        Args:
            experiment_id: unique ID of this experiment
            round_: current number of round the algorithm is performing (a round is considered to be all the
                training steps of a federated model between 2 aggregations).
            training_plan: TrainingPlan with properly initialized model and optimizer
            training_args: arguments for training
            model_args: arguments for the model
            data: metadata of the federated data set
            nodes_state_ids: unique IDs of the node states saved remotely
            aggregator_args: aggregator arguments required for remote execution
            secagg_arguments: Secure aggregation ServerKey context id
            do_training: if False, skip training in this round (do only validation). Defaults to True.
            optim_aux_var: Auxiliary variables of the researcher-side Optimizer, if any.
                Note that such variables may only be used if both the Experiment and node-side training plan
                hold a declearn-based [Optimizer][fedbiomed.common.optimizers.Optimizer], and their plug-ins
                are coherent with each other as to expected information exchange.
            *args: Positional argument of parent class
                [`Job`][fedbiomed.researcher.federated_workflows.jobs.Job]
            **kwargs: Named arguments of parent class. Please see
                [`Job`][fedbiomed.researcher.federated_workflows.jobs.Job]
        """
        super().__init__(**kwargs)
        # to be used for `execute()`
        self._experiment_id = experiment_id
        self._round_ = round_
        self._training_plan = training_plan
        self._training_args = training_args
        self._model_args = model_args
        self._data = data
        self._nodes_state_ids = nodes_state_ids
        self._aggregator_args = aggregator_args
        self._secagg_arguments = secagg_arguments or {}  # Assign empty dict to secagg arguments if it is None
        self._do_training = do_training
        self._optim_aux_var = optim_aux_var


    def _get_training_results(self,
                              replies: Dict[str, TrainReply],
                              errors: Dict[str, ErrorMessage],
                              ) -> Dict:
        """"Waits for training replies, and updates `_training_replies` wrt replies from Node(s) participating
         in the training

        Args:
            replies: replies from the request sent to Nodes
            errors: errors collected (if any) while sending requests and rertieving replies
        """
        training_replies = {}
        # Loops over errors
        for node_id, error in errors.items():
            logger.info(f"Error message received during training: {error.errnum}. {error.extra_msg}")
            self._nodes.remove(node_id)

        # Loops over replies
        for node_id, reply in replies.items():

            reply: TrainReply
            if not reply.success:
                logger.error(f"Training failed for node {reply.node_id}: {reply.msg}")
                self._nodes.remove(reply.node_id)  # remove the faulty node from the list
                continue

            params_path = os.path.join(self._keep_files_dir, f"params_{str(node_id)[0:11]}_{uuid.uuid4()}.mpk")
            Serializer.dump(reply.params, params_path)

            training_replies.update({
                node_id: {
                    **reply.get_dict(),
                    'params_path': params_path,
                }
            })
        return training_replies

    def _get_timing_results(self, replies: Dict[str, TrainReply], timer: Dict):
        """Retrieves timing results and updates it to the `_training_replies`"""
        # Loops over replies
        timings = {}
        for node_id, reply in replies.items():
            timing = reply.timing
            timing['rtime_total'] = timer[node_id]
            timings[node_id] = timing

        return timings

    def execute(self) -> Tuple[Dict, Optional[Dict]]:
        """ Sends training request to nodes and waits for the responses

        Returns:
            A tuple of
              * training replies for this round
              * Dict of auxiliary variables, collating node-wise information, with
                format `{mod_name: {node_id: node_dict}}`.
        """

        # Populate request message
        msg = {
            'researcher_id': self._researcher_id,
            'experiment_id': self._experiment_id,
            'training_args': self._training_args.dict(),
            'training': self._do_training,
            'model_args': self._model_args if self._model_args is not None else {},
            'round': self._round_,
            'training_plan': self._training_plan.source(),
            'training_plan_class': self._training_plan.__class__.__name__,
            'params': self._training_plan.get_model_params(
                exclude_buffers=not self._training_args.dict()['share_persistent_buffers']),
            'secagg_servkey_id': self._secagg_arguments.get('secagg_servkey_id'),
            'secagg_biprime_id': self._secagg_arguments.get('secagg_biprime_id'),
            'secagg_random': self._secagg_arguments.get('secagg_random'),
            'secagg_clipping_range': self._secagg_arguments.get('secagg_clipping_range'),
            'command': 'train',
            'aggregator_args': {},
        }

        # Prepare optimizer auxiliary variables, if any.
        if self._do_training and self._optim_aux_var:
            aux_shared, aux_bynode = (
                self._prepare_agg_optimizer_aux_var(self._optim_aux_var, nodes=list(self._nodes))
            )
        else:
            aux_shared = {}
            aux_bynode = {}

        # Loop over nodes, add node specific data and send train request
        messages = MessagesByNode()

        for node in self._nodes:
            msg['dataset_id'] = self._data.data()[node]['dataset_id']
            msg['aux_vars'] = [aux_shared, aux_bynode.get(node, None)]
            msg['state_id'] = self._nodes_state_ids.get(node)

            # add aggregator parameters to message header
            msg['aggregator_args'] = self._aggregator_args.get(node, {}) if self._aggregator_args else {}

            self._log_round_info(node=node, training=self._do_training)

            messages.update({node: TrainRequest(**msg)})  # send request to node

        with self.RequestTimer(self._nodes) as timer:  # compute request time
            # Send training request
            with self._reqs.send(messages, self._nodes, self._policies) as federated_req:

                errors = federated_req.errors()
                replies = federated_req.replies()

        training_replies = self._get_training_results(replies=replies,
                                                      errors=errors)

        timing_results = self._get_timing_results(replies, timer)
        # `training_replies` can be empty if there wasnot any replies
        for node_id in replies:
            if training_replies.get(node_id):
                training_replies[node_id].update({'timing': timing_results[node_id]})

        # Extract aux variables from training replies
        aux_vars = None
        if self._do_training:
            aux_vars = self._extract_received_optimizer_aux_var_from_round(training_replies)

        return training_replies, aux_vars

    def _log_round_info(self, node: str, training: True) -> None:
        """Logs round details

        Args:
            node: Node id
            training: If True round will do training, otherwise it is the last validation round
        """

        if training:
            logger.info(f'\033[1mSending request\033[0m \n'
                        f'\t\t\t\t\t\033[1m To\033[0m: {str(node)} \n'
                        f'\t\t\t\t\t\033[1m Request: \033[0m: TRAIN'
                        f'\n {5 * "-------------"}')
        else:
            logger.info(f'\033[1mSending request\033[0m \n'
                        f'\t\t\t\t\t\033[1m To\033[0m: {str(node)} \n'
                        f'\t\t\t\t\t\033[1m Request: \033[0m:Perform final validation on '
                        f'aggregated parameters \n {5 * "-------------"}')

    # FIXME: are aux_var supposed to be dealt with in the TrainingJob
    @staticmethod
    def _prepare_agg_optimizer_aux_var(
        aux_var: Dict[str, Dict[str, Any]],
        nodes: List[str],
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Dict[str, Any]]],
    ]:
        """Collect and structure researcher-side Optimizer auxiliary variables.

        Args:
            aux_var: Auxiliary variables with to structure into multiple dicts,
                from `{mod_name: (shared_dict | {node_id: node_dict})}` to
                `{mod_name: shared_dict}` & `{node_id: {mod_name: node_dict}}`.
            nodes: Ids of the nodes to whom auxiliary variables should be
                sent. This is used to drop information of non-participating
                nodes.

        Returns:
            aux_shared: Dict containing auxiliary variables that are shared
                across all nodes, with `{mod_name: shared_dict}` format.
            aux_bynode: Dict containing node-wise dicts of node-specific
                auxiliary variables, with `{node_id: {mod_name: node_dict}}`
                format.
        """
        aux_shared = {}  # type: Dict[str, Dict[str, Any]]
        aux_bynode = {}  # type: Dict[str, Dict[str, Dict[str, Any]]]
        # Iterate over nodes and plug-in-module-wise auxiliary variables.
        for node_id in nodes:
            aux_bynode[node_id] = {}
            for mod_name, mod_info in aux_var.items():
                # Case of node-specfic information.
                if node_aux := mod_info.get(str(node_id)):
                    aux_bynode[node_id][mod_name] = node_aux
                # Case of global information shared with all nodes.
                elif mod_name not in aux_shared:
                    aux_shared[mod_name] = mod_info
        # Return the restructured auxiliary variables dicts.
        return aux_shared, aux_bynode

    @staticmethod
    def _extract_received_optimizer_aux_var_from_round(
        training_replies: Dict
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Restructure the received auxiliary variables (if any) from a round.

        Args:
            training_replies: training replies received for this job

        Returns:
            Dict of auxiliary variables, collating node-wise information, with
            format `{mod_name: {node_id: node_dict}}`.
        """
        aux_var = {}  # type: Dict[str, Dict[str, Dict[str, Any]]]

        for node_id, reply in training_replies.items():
            node_av = reply.get("optim_aux_var", {})
            for module, params in node_av.items():
                aux_var.setdefault(module, {})[node_id] = params
        return aux_var

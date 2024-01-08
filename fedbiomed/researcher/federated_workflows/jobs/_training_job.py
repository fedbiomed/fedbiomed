# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manage the training part of the experiment."""

import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from fedbiomed.common.logger import logger
from fedbiomed.common.message import TrainReply, TrainRequest
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.training_args import TrainingArgs

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.requests import Requests, MessagesByNode

from fedbiomed.researcher.federated_workflows.jobs._job import Job


class TrainingJob(Job):
    """
    Represents the entity that manage the training part at  the nodes level

    Starts a message queue, loads python model file created by researcher (through
    [`training_plans`][fedbiomed.common.training_plans]) and saves the loaded model in a temporary file
    (under the filename '<TEMP_DIR>/my_model_<random_id>.py').

    """

    def __init__(self,
                 reqs: Requests = None,
                 nodes: Optional[dict] = None,
                 keep_files_dir: str = None):

        """ Constructor of the class

        Args:
            reqs: Researcher's requests assigned to nodes. Defaults to None.
            nodes: A dict of node_id containing the nodes used for training
            training_plan_class: instance or class of the TrainingPlan.
            training_plan_path: Path to file containing model class code
            training_args: Contains training parameters; lr, epochs, batch_size.
            model_args: Contains output and input feature dimension
            data: Federated datasets
            keep_files_dir: Directory for storing files created by the job that we want to keep beyond the execution
                of the job. Defaults to None, files are not kept after the end of the job.

        Raises:
            NameError: If model is not defined or if the class can not to be inspected
        """

        super().__init__(
            reqs,
            nodes,
            keep_files_dir
        )
        self._model_file = None  # path to local file containing model code
        self._model_params_file = ""  # path to local file containing current version of aggregated params
        self._aggregator_args = None

    @property
    def aggregator_args(self):
        return self._aggregator_args

    def get_initialized_tp_instance(self,
                                    training_plan_class: Type[Callable],
                                    training_args: TrainingArgs,
                                    model_args: Optional[dict],
                                    ):
        skeleton_training_plan = self.get_default_constructed_tp_instance(training_plan_class)
        skeleton_training_plan.post_init(model_args={} if model_args is None else model_args,
                                         training_args=training_args)
        return skeleton_training_plan

    def _get_training_testing_results(self, replies, errors, timer: Dict) -> Dict:
        """"Waits for training replies

        Args:
            timer: Stores time elapsed on the researcher side
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

            params_path = os.path.join(self._keep_files_dir, f"params_{node_id}.mpk")
            Serializer.dump(reply.params, params_path)

            rtime_total = time.perf_counter() - timer[node_id]

            # TODO: could choose completely different name/structure for
            timing = reply.timing
            timing['rtime_total'] = rtime_total

            training_replies.update({
                node_id: {
                    **reply.get_dict(),
                    'params_path': params_path,
                    'timing': timing,
                }
            })

        return training_replies

    def start_nodes_training_round(
        self,
        job_id: str, 
        round_: int,
        training_plan,
        training_plan_class,
        training_args: TrainingArgs,
        model_args: Optional[dict],
        data: FederatedDataSet,
        nodes_state_ids: Dict[str, str],
        aggregator_args: Dict[str, Dict[str, Any]],
        secagg_arguments: Union[Dict, None] = None,
        do_training: bool = True,
        optim_aux_var: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict:
        """ Sends training request to nodes and waits for the responses

        Args:
            round_: current number of round the algorithm is performing (a round is considered to be all the
                training steps of a federated model between 2 aggregations).
            aggregator_args_thr_msg: dictionary containing some metadata about the aggregation
                strategy, useful to transfer some data when it's required by am aggregator. First key should be the
                node_id, and sub-dictionary sould be parameters to be sent through MQTT messaging system
            aggregator_args_thr_files: dictionary containing metadata about aggregation strategy, to be transferred
                via the Repository's HTTP API, as opposed to the mqtt system. Format is the same as
                aggregator_args_thr_msg .
            secagg_arguments: Secure aggregation ServerKey context id
            do_training: if False, skip training in this round (do only validation). Defaults to True.
            optim_aux_var: Auxiliary variables of the researcher-side Optimizer, if any.
                Note that such variables may only be used if both the Experiment and node-side training plan
                hold a declearn-based [Optimizer][fedbiomed.common.optimizers.Optimizer], and their plug-ins
                are coherent with each other as to expected information exchange.
        """

        # Assign empty dict to secagg arguments if it is None
        if secagg_arguments is None:
            secagg_arguments = {}

        msg = {
            'researcher_id': self._researcher_id,
            'job_id': job_id,
            'training_args': training_args.dict(),
            'training': do_training,
            'model_args': model_args if model_args is not None else {},
            'round': round_,
            'training_plan': training_plan.source(),
            'training_plan_class': training_plan_class.__name__,
            'params': training_plan.get_model_params(),
            'secagg_servkey_id': secagg_arguments.get('secagg_servkey_id'),
            'secagg_biprime_id': secagg_arguments.get('secagg_biprime_id'),
            'secagg_random': secagg_arguments.get('secagg_random'),
            'secagg_clipping_range': secagg_arguments.get('secagg_clipping_range'),
            'command': 'train',
            'aggregator_args': {},
        }
        
        timer = {}

        # Upload optimizer auxiliary variables, when there are.
        if do_training and optim_aux_var:
            aux_shared, aux_bynode = (
                self._prepare_agg_optimizer_aux_var(optim_aux_var, nodes=list(self._nodes))
            )
        else:
            aux_shared = {}
            aux_bynode = {}
            
        # Loop over nodes, add node specific data and send train request
        messages = MessagesByNode()

        for node in self._nodes:
            msg['dataset_id'] = data.data()[node]['dataset_id']
            msg['aux_vars'] = [aux_shared, aux_bynode.get(node, None)]
            msg['state_id'] = nodes_state_ids.get(node)

            # add aggregator parameters to message header
            # FIXME: There might be another node join recently
            msg['aggregator_args'] = aggregator_args.get(node, {}) if aggregator_args else {}

            self._log_round_info(node=node, training=do_training)

            timer[node] = time.perf_counter()
            messages.update({node: TrainRequest(**msg)})  # send request to node

        # Sends training request    
        with self._reqs.send(messages, self._nodes) as federated_req:
            errors = federated_req.errors()
            replies = federated_req.replies()
            formatted_training_replies = self._get_training_testing_results(replies=replies,
                                                                            errors=errors,
                                                                            timer=timer)

        # return the list of nodes which answered because nodes in error have been removed
        return formatted_training_replies

    def _log_round_info(self, node: str, training: True) -> None:
        """Logs round details

        Args:
            node: Node id
            training: If True round will do training, otherwise it is the last validation round
        """

        if not training:
            logger.info(f'\033[1mSending request\033[0m \n'
                        f'\t\t\t\t\t\033[1m To\033[0m: {str(node)} \n'
                        f'\t\t\t\t\t\033[1m Request: \033[0m:Perform final validation on '
                        f'aggregated parameters \n {5 * "-------------"}')
        else:
            logger.info(f'\033[1mSending request\033[0m \n'
                        f'\t\t\t\t\t\033[1m To\033[0m: {str(node)} \n'
                        f'\t\t\t\t\t\033[1m Request: \033[0m: TRAIN'
                        f'\n {5 * "-------------"}')

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

    def extract_received_optimizer_aux_var_from_round(
        self,
        round_id: int,
        training_replies: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Restructure the received auxiliary variables (if any) from a round.

        Args:
            round_id: Index of the round, replies from which to parse through.

        Returns:
            Dict of auxiliary variables, collating node-wise information, with
            format `{mod_name: {node_id: node_dict}}`.
        """
        aux_var = {}  # type: Dict[str, Dict[str, Dict[str, Any]]]

        for node_id, reply in training_replies[round_id].items():
            node_av = reply.get("optim_aux_var", {})
            for module, params in node_av.items():
                aux_var.setdefault(module, {})[node_id] = params
        return aux_var

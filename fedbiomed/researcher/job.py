# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manage the training part of the experiment."""

import atexit
import copy
import inspect
import os
import shutil
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Type

from fedbiomed.common.constants import TrainingPlanApprovalStatus, JOB_PREFIX, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedJobError, FedbiomedNodeStateAgentError
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.message import  TrainRequest, TrainReply, TrainingPlanStatusRequest
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import TorchTrainingPlan, SKLearnTrainingPlan
from fedbiomed.common import utils

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.filetools import create_unique_link, create_unique_file_link
from fedbiomed.researcher.node_state_agent import NodeStateAgent
from fedbiomed.researcher.requests import Requests, MessagesByNode, DiscardOnTimeout

# for checking class passed to job (same definitions as experiment ...)
# TODO : should we move this to common/constants.py ? No because it means import training plans in it ...
training_plans_types = (TorchTrainingPlan, SKLearnTrainingPlan)
# for typing only
Typevar_TrainingPlanClass = TypeVar('Typevar_TrainingPlanClass', Type[TorchTrainingPlan], Type[SKLearnTrainingPlan])


class Job:
    """
    Represents the entity that manage the training part at  the nodes level

    Starts a message queue, loads python model file created by researcher (through
    [`training_plans`][fedbiomed.common.training_plans]) and saves the loaded model in a file/

    """

    def __init__(self,
                 reqs: Optional[Requests] = None,
                 training_plan_class: Optional[Typevar_TrainingPlanClass] = None,
                 training_args: TrainingArgs = None,
                 model_args: dict = None,
                 data: FederatedDataSet = None,
                 keep_files_dir: str = None):

        """ Constructor of the class

        Args:
            reqs: Researcher's requests assigned to nodes. Defaults to None.
            training_plan_class: Class containing the code of the TrainingPlan.
            training_args: Contains training parameters; lr, epochs, batch_size.
            model_args: Contains output and input feature dimension
            data: Federated datasets
            keep_files_dir: Directory for storing files created by the job that we want to keep beyond the execution
                of the job. Defaults to None, files are not kept after the end of the job.

        Raises:
            FedbiomedJobError: bad argument type or value
            FedbiomedJobError: cannot save training plan to file
        """
        # Check arguments
        if not inspect.isclass(training_plan_class):
            msg = f"{ErrorNumbers.FB418.value}: bad type for argument `training_plan_class` {type(training_plan_class)}"
            raise FedbiomedJobError(msg)

        if not issubclass(training_plan_class, training_plans_types):
            msg = f"{ErrorNumbers.FB418.value}: bad type for argument `training_plan_class`. It is not subclass of " + \
                  f" supported training plans {training_plans_types}"
            raise FedbiomedJobError(msg)

        # List of node ID of the nodes used in the current round
        # - initially None (no current round yet)
        # - then updated during the round with the list of nodes to be used in the round, then the nodes
        #   that actually replied during the round
        self._nodes : Optional[List[str]] = None

        self._id = JOB_PREFIX + str(uuid.uuid4())  # creating a unique job id
        self._researcher_id = environ['RESEARCHER_ID']
        self._training_args = training_args
        self._model_args = model_args
        self._training_replies = {}  # will contain all node replies for every round
        self._model_file = None  # path to local file containing model code
        self._model_params_file = ""  # path to local file containing current version of aggregated params
        self._training_plan_class = training_plan_class
        self._aggregator_args = None

        if keep_files_dir:
            self._keep_files_dir = keep_files_dir
        else:
            self._keep_files_dir = tempfile.mkdtemp(prefix=environ['TMP_DIR'])
            atexit.register(lambda: shutil.rmtree(self._keep_files_dir))  # remove directory
            # when script ends running (replace
            # `with tempfile.TemporaryDirectory(dir=environ['TMP_DIR']) as self._keep_files_dir: `)

        if reqs is None:
            self._reqs = Requests()
        else:
            self._reqs = reqs

        self.last_msg = None
        self._data = data
        self._node_state_agent = NodeStateAgent(list(self._data.data().keys())
                                                if self._data and self._data.data() else [])

        # create TrainingPlan instance
        self._training_plan = self._training_plan_class()  # contains TrainingPlan

        # save and load training plan to a file to be sure
        # 1. a file is associated to training plan so we can read its source, etc.
        # 2. all dependencies are applied
        training_plan_module = 'model_' + str(uuid.uuid4())
        self._training_plan_file = os.path.join(self._keep_files_dir, training_plan_module + '.py')
        try:
            self._training_plan.save_code(self._training_plan_file)
        except Exception as e:
            msg = f"{ErrorNumbers.FB418}: cannot save training plan to file: {e}"
            logger.critical(msg)
            raise FedbiomedJobError(msg)
        del self._training_plan

        _, self._training_plan = utils.import_class_object_from_file(
            self._training_plan_file, self._training_plan_class.__name__)

        self._training_plan.post_init(model_args={} if self._model_args is None else self._model_args,
                                      training_args=self._training_args)

    @property
    def id(self):
        return self._id

    @property
    def training_plan(self):
        return self._training_plan

    @property
    def training_plan_file(self):
        return self._training_plan_file

    @property
    def requests(self):
        return self._reqs

    @property
    def aggregator_args(self):
        return self._aggregator_args

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: dict):
        self._nodes = nodes

    @property
    def training_replies(self):
        return self._training_replies

    @property
    def training_args(self):
        return self._training_args.dict()

    @training_args.setter
    def training_args(self, training_args: TrainingArgs):
        self._training_args = training_args

    def check_training_plan_is_approved_by_nodes(self) -> Dict:
        """ Checks whether model is approved or not.

        This method sends `training-plan-status` request to the nodes. It should be run before running experiment.
        So, researchers can find out if their model has been approved

        Returns:
            A dict of `Message` objects indexed by node ID, one for each job's nodes
        """

        message = TrainingPlanStatusRequest(**{
            'researcher_id': self._researcher_id,
            'job_id': self._id,
            'training_plan': self._training_plan.source(),
            'command': 'training-plan-status'
        })

        node_ids = self._data.node_ids()

        # Send message to each node that has been found after dataset search request
        with self._reqs.send(message, node_ids, policies=[DiscardOnTimeout(5)]) as federated_req:
            replies = federated_req.replies()

            for node_id, reply in replies.items():
                if reply.success is True:
                    if reply.approval_obligation is True:
                        if reply.status == TrainingPlanApprovalStatus.APPROVED.value:
                            logger.info(f'Training plan has been approved by the node: {node_id}')
                        else:
                            logger.warning(f'Training plan has NOT been approved by the node: {node_id}.' +
                                           f'Training plan status : {node_id}')
                    else:
                        logger.info(f'Training plan approval is not required by the node: {node_id}')
                else:
                    logger.warning(f"Node : {node_id} : {reply.msg}")

        # Get the nodes that haven't replied training-plan-status request
        non_replied_nodes = list(set(node_ids) - set(replies.keys()))
        if non_replied_nodes:
            logger.warning(f"Request for checking training plan status hasn't been replied \
                             by the nodes: {non_replied_nodes}. You might get error \
                                 while running your experiment. ")

        return replies

    def _get_training_testing_results(self, replies, errors, round_, timer: Dict) -> None:
        """"Waits for training replies

        Args:
            round_: Training round
            timer: Stores time elapsed on the researcher side
        """

        self._training_replies[round_] = {}

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

            self._training_replies[round_].update({
                node_id: {
                    **reply.get_dict(),
                    'params_path': params_path,
                    'timing': timing,
                }
            })

    def start_nodes_training_round(
        self,
        round_: int,
        aggregator_args: Dict[str, Dict[str, Any]],
        secagg_arguments: Optional[Dict] = None,
        do_training: bool = True,
        optim_aux_var: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """ Sends training request to nodes and waits for the replies

        Args:
            round_: current number of round the algorithm is performing (a round is considered to be all the
                training steps of a federated model between 2 aggregations).
            aggregator_args: dictionary containing some metadata about the aggregation
                strategy, useful to transfer some data when it's required by am aggregator.
            secagg_arguments: Secure aggregation ServerKey context id
            do_training: if False, skip training in this round (do only validation). Defaults to True.
            optim_aux_var: Auxiliary variables of the researcher-side Optimizer, if any.
                Note that such variables may only be used if both the Experiment and node-side training plan
                hold a declearn-based [Optimizer][fedbiomed.common.optimizers.Optimizer], and their plug-ins
                are coherent with each other as to expected information exchange.
        """

        # Assign empty dict to secagg arguments if it is None
        secagg_arguments = {} if secagg_arguments is None else secagg_arguments

        msg = {
            'researcher_id': self._researcher_id,
            'job_id': self._id,
            'training_args': self._training_args.dict(),
            'training': do_training,
            'model_args': self._model_args,
            'round': round_,
            'training_plan': self._training_plan.source(),
            'training_plan_class': self._training_plan_class.__name__,
            'params': self._get_model_params(),
            'secagg_servkey_id': secagg_arguments.get('secagg_servkey_id'),
            'secagg_biprime_id': secagg_arguments.get('secagg_biprime_id'),
            'secagg_random': secagg_arguments.get('secagg_random'),
            'secagg_clipping_range': secagg_arguments.get('secagg_clipping_range'),
            'command': 'train',
            'aggregator_args': {},
            'aux_vars': [],
        }

        timer = {}

        if do_training:
            # update node states when used node list has changed from one round to another
            self._update_nodes_states_agent()

        # FIXME: should be part of a method called from Experiment
        # (behaviour can be defined by user / changed by strategy)
        nodes_state_ids = self._node_state_agent.get_last_node_states()

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

        #MANI
        #for node in self._nodes:
        for iter, node in enumerate(self._nodes):
            msg['training_args'] = {**msg['training_args'], "gpu_num": iter}

            msg['dataset_id'] = self._data.data()[node]['dataset_id']
            msg['aux_vars'] = [aux_shared, aux_bynode.get(node, None)]

            msg['state_id'] = nodes_state_ids.get(node)

            # FIXME: There might be another node join recently
            msg['aggregator_args'] = aggregator_args.get(node, {}) if aggregator_args else {}
            self._log_round_info(node=node, training=do_training)

            timer[node] = time.perf_counter()

            messages.update({node: TrainRequest(**msg)})

            # Sends training request

        with self._reqs.send(messages, self._nodes) as federated_req:
            errors = federated_req.errors()
            replies = federated_req.replies()
            self._get_training_testing_results(replies=replies, errors=errors, round_=round_, timer=timer)

        if do_training:
            # update node states with node answers + when used node list has changed during the round
            self._update_nodes_states_agent(before_training=False)

        # return the list of nodes which answered because nodes in error have been removed
        return self._nodes

    @staticmethod
    def _prepare_agg_optimizer_aux_var(
        aux_var: Dict[str, Dict[str, Any]],
        nodes: List[uuid.UUID],
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[uuid.UUID, Dict[str, Dict[str, Any]]],
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
        aux_bynode = {}  # type: Dict[uuid.UUID, Dict[str, Dict[str, Any]]]
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
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Restructure the received auxiliary variables (if any) from a round.

        Args:
            round_id: Index of the round, replies from which to parse through.

        Returns:
            Dict of auxiliary variables, collating node-wise information, with
            format `{mod_name: {node_id: node_dict}}`.
        """
        aux_var = {}  # type: Dict[str, Dict[str, Dict[str, Any]]]
        for reply in self.training_replies[round_id].values():
            node_id = reply["node_id"]
            node_av = reply.get("optim_aux_var", {})
            for module, params in node_av.items():
                aux_var.setdefault(module, {})[node_id] = params
        return aux_var

    def _get_model_params(self) -> Dict[str, Any]:
        """Gets model parameters form the training plan.

        Returns:
            Model weights, as a dictionary mapping parameters' names to their value.
        """
        return self._training_plan.get_model_params()

    def _load_and_set_model_params_from_file(self, path: str) -> None:
        """Loads model parameters from given path

        Args:
            path: The path where model parameters are saved
        """
        params = Serializer.load(path)
        self._training_plan.set_model_params(params)

    def _update_model_params(self, params: Dict[str, Any]) -> None:
        """"Updates the parameters of the model by given params

        Args:
            params: Parameters that are going to be loaded into model
        """
        self._training_plan.set_model_params(params)

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

    def update_parameters(
        self,
        params: Optional[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Update model parameters

        Args:
            params: Aggregated model parameters

        Returns:
            Path of filename where parameters are saved
        """
        self._update_model_params(params)
        filename = os.path.join(self._keep_files_dir, f"aggregated_params_{uuid.uuid4()}.mpk")
        Serializer.dump(params, filename)
        self._model_params_file = filename

        return filename

    def _update_nodes_states_agent(self, before_training: bool = True):
        """Updates [`NodeStateAgent`][fedbiomed.researcher.node_state_agent.NodeStateAgent], with the latest
        state_id coming from `Nodes` contained among all `Nodes` within
        [`FederatedDataset`][fedbiomed.researcher.datasets.FederatedDataset].

        Args:
            before_training: whether to update `NodeStateAgent` at the begining or at the end of a `Round`:
                - if before, only updates `NodeStateAgent` wrt `FederatedDataset`, otherwise
                - if after, updates `NodeStateAgent` wrt latest reply

        Raises:
            FedBiomedNodeStateAgenError: failing to update `NodeStateAgent`.

        """
        node_ids = list(self._data.data().keys()) if self._data and self._data.data() else []
        if before_training:
            self._node_state_agent.update_node_states(node_ids)
        else:
            # extract last node state
            # FIXME: for now we are only considering the case where we need last Round update,
            # but we may want to generalize to other use cases (for some aggregators, we may want to retrieve even more
            # previous Node replies)
            try:
                last_tr_entry = list(self._training_replies.keys())[-1]
            except IndexError as ie:
                raise FedbiomedNodeStateAgentError(f"{ErrorNumbers.FB323.value}: Cannot update NodeStateAgent if No "
                                                   "replies form Node(s) has(ve) been recieved!") from ie

            self._node_state_agent.update_node_states(node_ids, self._training_replies[last_tr_entry])

    def save_state_breakpoint(self, breakpoint_path: str) -> dict:
        """Creates current state of the job to be included in a breakpoint.

        Includes creating links to files included in the job state.

        Args:
            breakpoint_path: path to the existing breakpoint directory

        Returns:
            Job's current state for breakpoint
        """

        # Note: some state is passed to __init__() thus is not managed
        # as job state but as experiment state in current version
        state = {
            'researcher_id': self._researcher_id,
            'job_id': self._id,
            'model_params_path': self._model_params_file,
            'training_replies': self._save_training_replies(self._training_replies),
            'node_state': self._node_state_agent.save_state_breakpoint()
        }

        state['model_params_path'] = create_unique_link(
            breakpoint_path, 'aggregated_params_current', '.mpk',
            os.path.join('..', os.path.basename(state["model_params_path"]))
        )

        for round_replies in state['training_replies']:
            for response in round_replies.values():
                node_params_path = create_unique_file_link(
                    breakpoint_path, response['params_path']
                )
                response['params_path'] = node_params_path

        return state

    def load_state_breakpoint(self, saved_state: Dict[str, Any]) -> None:
        """Load breakpoints state for a Job from a saved state

        Args:
            saved_state: breakpoint content
        """
        # Reload the job and researched ids.
        self._id = saved_state.get('job_id')
        self._researcher_id = saved_state.get('researcher_id')
        self._node_state_agent.load_state_breakpoint(saved_state.get('node_state'))
        # Upload the latest model parameters. This records the filename and url.
        params = Serializer.load(saved_state.get("model_params_path"))
        self.update_parameters(params)

        self._load_and_set_model_params_from_file(saved_state.get("model_params_path"))
        # Reload the latest training replies.
        self._training_replies = self._load_training_replies(
            saved_state.get('training_replies', {})
        )

    @staticmethod
    def _save_training_replies(training_replies: Dict[int, Any]) -> List[List[Dict[str, Any]]]:
        """Extracts a copy of `training_replies` and prepares it for saving in breakpoint

        - strip unwanted fields
        - structure as list/dict, so it can be saved with JSON

        Args:
            training_replies: training replies of already executed rounds of the job

        Returns:
            Extract from `training_replies` formatted for breakpoint
        """
        converted_training_replies = []

        for round_ in training_replies.keys():
            training_reply = copy.deepcopy(training_replies[round_])
            # we want to strip some fields for the breakpoint
            for reply in training_reply.values():
                reply.pop('params', None)

            converted_training_replies.append(training_reply)

        return converted_training_replies

    @staticmethod
    def _load_training_replies(bkpt_training_replies: List[List[dict]]) -> Dict[int, Dict]:
        """Reads training replies from a formatted breakpoint file, and build a job training replies data structure .

        Args:
            bkpt_training_replies: Extract from training replies saved in breakpoint

        Returns:
            Training replies of already executed rounds of the job
        """

        training_replies = {}
        if not bkpt_training_replies:
            logger.warning("No Replies has been found in this breakpoint")

        for round_ in range(len(bkpt_training_replies)-1, len(bkpt_training_replies)):
            loaded_training_reply = bkpt_training_replies[round_]
            # reload parameters from file params_path
            for node in loaded_training_reply.values():
                node["params"] = Serializer.load(node["params_path"])

            training_replies[round_] = loaded_training_reply

        return training_replies


class localJob:
    """Represents the entity that manage the training part. LocalJob is the version of Job but applied locally on a
    local dataset (thus not involving any network). It is only used to compare results to a Federated approach, using
    networks.
    """

    def __init__(self,
                 dataset_path: Optional[str] = None,
                 training_plan_class: Optional[Typevar_TrainingPlanClass] = None,
                 training_args: Optional[TrainingArgs] = None,
                 model_args: Optional[dict] = None):

        """
        Constructor of the class

        Args:
            dataset_path : The path where data is stored on local disk.
            training_plan_class: Class containing the code of the TrainingPlan.
            training_args: Contains training parameters: lr, epochs, batch_size...
            model_args: Contains output and input feature dimension.

        Raises:
            FedbiomedJobError: bad argument type or value
        """
        # Check arguments
        if not inspect.isclass(training_plan_class):
            raise FedbiomedJobError(
                f"{ErrorNumbers.FB418}: bad type for argument `training_plan_class` {type(training_plan_class)}"
            )
        if not issubclass(training_plan_class, training_plans_types):
            raise FedbiomedJobError(
                f"{ErrorNumbers.FB418}: bad type for argument "
                "`training_plan_class` {training_plan_class} is not subclass of training plans")

        # Initialize values
        self._training_args = training_args
        self._model_args = model_args
        self.dataset_path = dataset_path

        if training_args is not None:
            if training_args.get('test_on_local_updates', False) \
                    or training_args.get('test_on_global_updates', False):
                # if user wants to perform validation, display this message
                logger.warning("Cannot perform validation, not supported for LocalJob")

        if not isinstance(training_args, TrainingArgs):
            self._training_args = TrainingArgs(training_args, only_required=False)
        else:
            self._training_args = training_args

        # create/save model instance
        self._training_plan = training_plan_class()

        self._training_plan.post_init(model_args=self._model_args,
                                      training_args=self._training_args)

    @property
    def training_plan(self):
        return self._training_plan

    @property
    def training_args(self):
        return self._training_args.dict()

    @training_args.setter
    def training_args(self, training_args: TrainingArgs):
        self._training_args = training_args

    def start_training(self):
        """Run the local training"""
        # Run import statements (very unsafely).
        for i in self._training_plan.dependencies:
            exec(i, globals())

        # Run the training routine.
        try:
            self._training_plan.set_dataset_path(self.dataset_path)
            data_manager = self._training_plan.training_data()
            tp_type = self._training_plan.type()
            data_manager.load(tp_type=tp_type)
            train_loader, test_loader = data_manager.split(test_ratio=0)
            self._training_plan.training_data_loader = train_loader
            self._training_plan.testing_data_loader = test_loader
            self._training_plan.training_routine()
        except Exception as exc:
            logger.error("Cannot train model in job: %s", repr(exc))
        # Save the current parameters.
        else:
            try:
                # TODO: should test status code but not yet returned by upload_file
                path = os.path.join(
                    environ["TMP_DIR"], f"local_params_{uuid.uuid4()}.mpk"
                )
                Serializer.dump(self._training_plan.get_model_params(), path)
            except Exception as exc:
                logger.error("Cannot write results: %s", repr(exc))

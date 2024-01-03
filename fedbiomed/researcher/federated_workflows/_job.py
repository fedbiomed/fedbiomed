# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manage the training part of the experiment."""

from abc import abstractmethod
import atexit
import importlib
import inspect
import os
import re
import sys
import shutil
import tempfile
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, Union
from fedbiomed.common import utils
from fedbiomed.common.message import TrainingPlanStatusRequest
from fedbiomed.researcher.node_state_agent import NodeStateAgent

import validators

from fedbiomed.common.constants import ErrorNumbers, TrainingPlanApprovalStatus, JOB_PREFIX
from fedbiomed.common.exceptions import FedbiomedJobError, FedbiomedNodeStateAgentError, FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.filetools import create_unique_link, create_unique_file_link
from fedbiomed.researcher.requests import Requests, DiscardOnTimeout



class Job:
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

        """

        #self._id = JOB_PREFIX + str(uuid.uuid4())  # creating a unique job id
        self._researcher_id = environ['RESEARCHER_ID']
        #self._repository_args = {}


        # List of node ID of the nodes used in the current round
        # - initially None (no current round yet)
        # - then updated during the round with the list of nodes to be used in the round, then the nodes
        #   that actually replied during the round
        self._nodes : Optional[List[str]] = nodes

        if keep_files_dir:
            self._keep_files_dir = keep_files_dir
        else:
            self._keep_files_dir = tempfile.mkdtemp(prefix=environ['TMP_DIR'])
            atexit.register(lambda: shutil.rmtree(self._keep_files_dir))  # remove directory
            # executed when script ends running (replace
            # `with tempfile.TemporaryDirectory(dir=environ['TMP_DIR']) as self._keep_files_dir: `)

        if reqs is None:
            self._reqs = Requests()
        else:
            self._reqs = reqs

        self.last_msg = None

        self._training_plan_module = 'my_model_' + str(uuid.uuid4())
        self._training_plan_file = os.path.join(self._keep_files_dir, self._training_plan_module + '.py')

    def get_default_constructed_workflow_instance(self,
                                                  training_plan_class: Type[Callable],
                                                  ) -> 'BaseTrainingPlan':

        # create TrainingPlan instance
        training_plan = training_plan_class()  # contains TrainingPlan

        # save and load training plan to a file to be sure
        # 1. a file is associated to training plan so we can read its source, etc.
        # 2. all dependencies are applied
        training_plan_module = 'model_' + str(uuid.uuid4())
        training_plan_file = os.path.join(self._keep_files_dir, training_plan_module + '.py')
        try:
            training_plan.save_code(training_plan_file)
        except Exception as e:
            msg = f"{ErrorNumbers.FB418}: cannot save training plan to file: {e}"
            logger.critical(msg)
            raise FedbiomedJobError(msg)
        del training_plan

        _, training_plan = utils.import_class_object_from_file(
            training_plan_file, training_plan_class.__name__)

        # # handle case when model is in a file
        # if training_plan_path is not None:
        #     try:
        #         # import model from python file

        #         model_module = os.path.basename(training_plan_path)
        #         model_module = re.search("(.*)\.py$", model_module).group(1)
        #         sys.path.insert(0, os.path.dirname(training_plan_path))

        #         module = importlib.import_module(model_module)
        #         tr_class = getattr(module, training_plan_class)
        #         training_plan_class = tr_class
        #         sys.path.pop(0)

        #     except Exception as e:
        #         e = sys.exc_info()
        #         logger.critical(f"Cannot import class {training_plan_class} from "
        #                         f"path {training_plan_path} - Error: {str(e)}")
        #         sys.exit(-1)

        # # check class is defined
        # try:
        #     _ = inspect.isclass(training_plan_class)
        # except NameError:
        #     mess = f"Cannot find training plan for Job, training plan class {training_plan_class} is not defined"
        #     logger.critical(mess)
        #     raise NameError(mess)

        # # create/save TrainingPlan instance
        # if inspect.isclass(training_plan_class):
        #     training_plan = training_plan_class()  # contains TrainingPlan

        # else:
        #     training_plan = training_plan_class
        # training_plan.configure_dependencies()

        return training_plan

    @property
    def training_plan_file(self):
        return self._training_plan_file

    @property
    def requests(self):
        return self._reqs

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: dict):
        self._nodes = nodes

    def _save_workflow_code_to_file(self,
                                    training_plan: 'fedbiomed.researcher.federated_workflows.FederatedWorkflow'
    ) -> None:
        try:
           training_plan.save_code(self._training_plan_file)
        except Exception as e:
            msg = f"Cannot save the training plan to a local tmp dir : {str(e)}"
            raise FedbiomedTrainingPlanError(msg)

    def check_training_plan_is_approved_by_nodes(self,
                                                 training_plan, 
                                                 job_id: str,
                                                 data: FederatedDataSet = None,
                                                 ) -> List:

        """ Checks whether model is approved or not.

        This method sends `training-plan-status` request to the nodes. It should be run before running experiment.
        So, researchers can find out if their model has been approved

        """

        message = TrainingPlanStatusRequest(**{
            'researcher_id': self._researcher_id,
            'job_id': job_id,
            'training_plan': training_plan.source(),
            'command': 'training-plan-status'
        })


        replied_nodes = []
        node_ids = data.node_ids()

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
        non_replied_nodes = list(set(node_ids) - set(replied_nodes))
        if non_replied_nodes:
            logger.warning(f"Request for checking training plan status hasn't been replied \
                             by the nodes: {non_replied_nodes}. You might get error \
                                 while running your experiment. ")

        return replies

    # TODO: This method should change in the future or as soon as we implement other of strategies different
    #   than DefaultStrategy

    # def waiting_for_nodes(self, responses: Responses) -> bool:
    #     """ Verifies if all nodes involved in the job are present and Responding

    #     Args:
    #         responses: contains message answers

    #     Returns:
    #         False if all nodes are present in the Responses object. True if waiting for at least one node.
    #     """
    #     try:
    #         nodes_done = set(responses.dataframe()['node_id'])
    #     except KeyError:
    #         nodes_done = set()

    #     return not nodes_done == set(self._nodes)

    def save_state_breakpoint(self, job_id: str, breakpoint_path: str) -> dict:
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
            'job_id': job_id,
            'training_replies': self._save_training_replies(self._training_replies)
        }

        state['model_params_path'] = create_unique_link(
            breakpoint_path, 'aggregated_params_current', '.mpk',
            os.path.join('../..', os.path.basename(state["model_params_path"]))
        )

        return state

    def load_state_breakpoint(self, saved_state: Dict[str, Any]) -> None:
        """Load breakpoints state for a Job from a saved state

        Args:
            saved_state: breakpoint content
        """
        # Reload the job and researched ids.
        #self._id = saved_state.get('job_id')
        self._researcher_id = saved_state.get('researcher_id')
        # Reload the latest training replies.
        self._training_replies = self._load_training_replies(
            saved_state.get('training_replies', [])
        )

    # def _update_nodes_states_agent(self, before_training: bool = True):
    #     """Updates [`NodeStateAgent`][fedbiomed.researcher.node_state_agent.NodeStateAgent], with the latest
    #     state_id coming from `Nodes` contained among all `Nodes` within
    #     [`FederatedDataset`][fedbiomed.researcher.datasets.FederatedDataset].

    #     Args:
    #         before_training: whether to update `NodeStateAgent` at the begining or at the end of a `Round`:
    #             - if before, only updates `NodeStateAgent` wrt `FederatedDataset`, otherwise
    #             - if after, updates `NodeStateAgent` wrt latest [`Responses`][fedbiomed.researcher.responses.Responses]

    #     Raises:
    #         FedBiomedNodeStateAgenError: failing to update `NodeStateAgent`.

    #     """
    #     # TODO: find a way to access to data
        
    #     if before_training:
    #         self._node_state_agent.update_node_states(self._nodes)
    #     else:
    #         # extract last node state
    #         # FIXME: for now we are only considering the case where we need last Round update,
    #         # but we may want to generalize to other use cases (for some aggregators, we may want to retrieve even more
    #         # previous Node replies)
    #         try:
    #             last_tr_entry = list(self.training_replies.keys())[-1]
    #         except IndexError as ie:
    #             raise FedbiomedNodeStateAgentError(f"{ErrorNumbers.FB323.value}: Cannot update NodeStateAgent if No "
    #                                                "replies form Node(s) has(ve) been recieved!") from ie

    #         self._node_state_agent.update_node_states(node_ids, self.training_replies[last_tr_entry])
            
    @abstractmethod
    def _save_training_replies(training_replies: Dict[int, Any]) -> List[List[Dict[str, Any]]]:
        """Extracts a copy of `training_replies` and prepares it for saving in breakpoint

        - strip unwanted fields
        - structure as list/dict, so it can be saved with JSON

        Args:
            training_replies: training replies of already executed rounds of the job

        Returns:
            Extract from `training_replies` formatted for breakpoint
        """

    @abstractmethod
    def _load_training_replies(bkpt_training_replies: List[List[dict]]) -> Dict[int, Any]:
        """Reads training replies from a formatted breakpoint file, and build a job training replies data structure .

        Args:
            bkpt_training_replies: Extract from training replies saved in breakpoint

        Returns:
            Training replies of already executed rounds of the job
        """

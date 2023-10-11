# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manage the training part of the experiment."""

import atexit
import copy
import importlib
import inspect
import os
import re
import sys
import shutil
import tempfile
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import validators

from fedbiomed.common.constants import TrainingPlanApprovalStatus
from fedbiomed.common.exceptions import FedbiomedRepositoryError, FedbiomedDataQualityCheckError
from fedbiomed.common.logger import logger
from fedbiomed.common.repository import Repository
from fedbiomed.common.serializer import Serializer
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan

from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.filetools import create_unique_link, create_unique_file_link
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.responses import Responses


class Job:
    """
    Represents the entity that manage the training part at  the nodes level

    Starts a message queue, loads python model file created by researcher (through
    [`training_plans`][fedbiomed.common.training_plans]) and saves the loaded model in a temporary file
    (under the filename '<TEMP_DIR>/my_model_<random_id>.py').

    """

    def __init__(self,
                 reqs: Requests = None,
                 nodes: dict = None,
                 training_args: TrainingArgs = None,
                 data: FederatedDataSet = None,
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

        self._id = str(uuid.uuid4())  # creating a unique job id
        self._researcher_id = environ['RESEARCHER_ID']
        self._repository_args = {}
        self._training_args = training_args
        self._nodes = nodes
        self._training_replies = {}  # will contain all node replies for every round

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
        self.repo = Repository(environ['UPLOADS_URL'], self._keep_files_dir, environ['CACHE_DIR'])
        self._training_plan_module = 'my_model_' + str(uuid.uuid4())
        self._training_plan_file = os.path.join(self._keep_files_dir, self._training_plan_module + '.py')

    def create_workflow_instance_from_path(self,
                                           training_plan_path: str = None,
                                           training_plan_class: Union[Type[Callable], str] = None,
                                          ) -> 'FederatedWorkflow':
        # handle case when model is in a file
        if training_plan_path is not None:
            try:
                # import model from python file

                model_module = os.path.basename(training_plan_path)
                model_module = re.search("(.*)\.py$", model_module).group(1)
                sys.path.insert(0, os.path.dirname(training_plan_path))

                module = importlib.import_module(model_module)
                tr_class = getattr(module, training_plan_class)
                training_plan_class = tr_class
                sys.path.pop(0)

            except Exception as e:
                e = sys.exc_info()
                logger.critical(f"Cannot import class {training_plan_class} from "
                                f"path {training_plan_path} - Error: {str(e)}")
                sys.exit(-1)

        # check class is defined
        try:
            _ = inspect.isclass(training_plan_class)
        except NameError:
            mess = f"Cannot find training plan for Job, training plan class {training_plan_class} is not defined"
            logger.critical(mess)
            raise NameError(mess)

        # create/save TrainingPlan instance
        if inspect.isclass(training_plan_class):
            training_plan = training_plan_class()  # contains TrainingPlan

        else:
            training_plan = training_plan_class
        training_plan.configure_dependencies()

        # find the name of the class in any case
        # (it is `model` only in the case where `model` is not an instance)

        return training_plan

    def upload_workflow_code(self, training_plan: 'FederatedWorkflow') -> None:

        try:
            training_plan.save_code(self._training_plan_file)
        except Exception as e:
            logger.error("Cannot save the training plan to a local tmp dir : " + str(e))
            return

        # upload my_model_xxx.py on repository server (contains model definition)
        repo_response = self.repo.upload_file(self._training_plan_file)

        self._repository_args['training_plan_url'] = repo_response['file']

        training_plan_name = training_plan.__class__.__name__
        # (below) regex: matches a character not present among "^", "\", "."
        # characters at the end of string.
        self._repository_args['training_plan_class'] = training_plan_name

        # Validate fields in each argument
        self.validate_minimal_arguments(self._repository_args,
                                        ['training_plan_url', 'training_plan_class'])

    @staticmethod
    def validate_minimal_arguments(obj: dict, fields: Union[tuple, list]):
        """ Validates a given dictionary by given mandatory fields.

        Args:
            obj: Object to be validated
            fields: List of fields that should be present on the obj
        """
        for f in fields:
            assert f in obj.keys(), f'Field {f} is required in object {obj}. Was not found.'
            if 'url' in f:
                assert validators.url(obj[f]), f'Url not valid: {f}'

    @property
    def id(self):
        return self._id

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

    @property
    def training_replies(self):
        return self._training_replies

    @property
    def training_args(self):
        return self._training_args.dict()

    @training_args.setter
    def training_args(self, training_args: TrainingArgs):
        self._training_args = training_args

    def check_training_plan_is_approved_by_nodes(self) -> List:

        """ Checks whether model is approved or not.

        This method sends `training-plan-status` request to the nodes. It should be run before running experiment.
        So, researchers can find out if their model has been approved

        """

        message = {
            'researcher_id': self._researcher_id,
            'job_id': self._id,
            'training_plan_url': self._repository_args['training_plan_url'],
            'command': 'training-plan-status'
        }

        responses = Responses([])
        replied_nodes = []
        node_ids = self._data.node_ids()

        # Send message to each node that has been found after dataset search request
        for cli in node_ids:
            logger.info('Sending request to node ' +
                        str(cli) + " to check model is approved or not")
            self._reqs.send_message(
                message,
                cli)

        # Wait for responses
        for resp in self._reqs.get_responses(look_for_commands=['training-plan-status'], only_successful=False):
            responses.append(resp)
            replied_nodes.append(resp.get('node_id'))

            if resp.get('success') is True:
                if resp.get('approval_obligation') is True:
                    if resp.get('status') == TrainingPlanApprovalStatus.APPROVED.value:
                        logger.info(f'Training plan has been approved by the node: {resp.get("node_id")}')
                    else:
                        logger.warning(f'Training plan has NOT been approved by the node: {resp.get("node_id")}.' +
                                       f'Training plan status : {resp.get("status")}')
                else:
                    logger.info(f'Training plan approval is not required by the node: {resp.get("node_id")}')
            else:
                logger.warning(f"Node : {resp.get('node_id')} : {resp.get('msg')}")

        # Get the nodes that haven't replied training-plan-status request
        non_replied_nodes = list(set(node_ids) - set(replied_nodes))
        if non_replied_nodes:
            logger.warning(f"Request for checking training plan status hasn't been replied \
                             by the nodes: {non_replied_nodes}. You might get error \
                                 while running your experiment. ")

        return responses

    # TODO: This method should change in the future or as soon as we implement other of strategies different
    #   than DefaultStrategy

    def waiting_for_nodes(self, responses: Responses) -> bool:
        """ Verifies if all nodes involved in the job are present and Responding

        Args:
            responses: contains message answers

        Returns:
            False if all nodes are present in the Responses object. True if waiting for at least one node.
        """
        try:
            nodes_done = set(responses.dataframe()['node_id'])
        except KeyError:
            nodes_done = set()

        return not nodes_done == set(self._nodes)

    def save_state(self, breakpoint_path: str) -> dict:
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
            'training_replies': self._save_training_replies(self._training_replies)
        }

        state['model_params_path'] = create_unique_link(
            breakpoint_path, 'aggregated_params_current', '.mpk',
            os.path.join('../..', os.path.basename(state["model_params_path"]))
        )

        return state

    def load_state(self, saved_state: Dict[str, Any]) -> None:
        """Load breakpoints state for a Job from a saved state

        Args:
            saved_state: breakpoint content
        """
        # Reload the job and researched ids.
        self._id = saved_state.get('job_id')
        self._researcher_id = saved_state.get('researcher_id')
        # Reload the latest training replies.
        self._training_replies = self._load_training_replies(
            saved_state.get('training_replies', [])
        )

    @staticmethod
    def _save_training_replies(training_replies: Dict[int, Responses]) -> List[List[Dict[str, Any]]]:
        """Extracts a copy of `training_replies` and prepares it for saving in breakpoint

        - strip unwanted fields
        - structure as list/dict, so it can be saved with JSON

        Args:
            training_replies: training replies of already executed rounds of the job

        Returns:
            Extract from `training_replies` formatted for breakpoint
        """
        converted_training_replies = []
        return converted_training_replies

    @staticmethod
    def _load_training_replies(bkpt_training_replies: List[List[dict]]) -> Dict[int, Responses]:
        """Reads training replies from a formatted breakpoint file, and build a job training replies data structure .

        Args:
            bkpt_training_replies: Extract from training replies saved in breakpoint

        Returns:
            Training replies of already executed rounds of the job
        """

        training_replies = {}
        if not bkpt_training_replies:
            logger.warning("No Replies has been found in this breakpoint")
        return training_replies


class localJob:
    """Represents the entity that manage the training part. LocalJob is the version of Job but applied locally on a
    local dataset (thus not involving any network). It is only used to compare results to a Federated approach, using
    networks.
    """

    def __init__(self, dataset_path: str = None,
                 training_plan_class: str = 'MyTrainingPlan',
                 training_plan_path: str = None,
                 training_args: TrainingArgs = None,
                 model_args: dict = None):

        """
        Constructor of the class

        Args:
            dataset_path : The path where data is stored on local disk.
            training_plan_class: Name of the model class to use for training or model class.
            training_plan_path: path to file containing model code. Defaults to None.
            training_args: contains training parameters: lr, epochs, batch_size...
            model_args: contains output and input feature dimension.
        """

        self._id = str(uuid.uuid4())
        self._repository_args = {}
        self._localjob_training_args = training_args
        self._model_args = model_args
        self._training_args = TrainingArgs(training_args, only_required=False)
        self.dataset_path = dataset_path

        if training_args is not None:
            if training_args.get('test_on_local_updates', False) \
                    or training_args.get('test_on_global_updates', False):
                # if user wants to perform validation, display this message
                logger.warning("Cannot perform validation, not supported for LocalJob")

        # handle case when model is in a file
        if training_plan_path is not None:
            try:
                model_module = os.path.basename(training_plan_path)
                model_module = re.search("(.*)\.py$", model_module).group(1)
                sys.path.insert(0, os.path.dirname(training_plan_path))

                module = importlib.import_module(model_module)
                tr_class = getattr(module, training_plan_class)
                self._training_plan = tr_class()
                sys.path.pop(0)

            except Exception as e:
                e = sys.exc_info()
                logger.critical("Cannot import class " + training_plan_class + " from path " +
                                training_plan_path + " - Error: " + str(e))
                sys.exit(-1)
        else:

            # create/save model instance
            if inspect.isclass(training_plan_class):
                self._training_plan = training_plan_class()
            else:
                self._training_plan = training_plan_class

        self._training_plan.post_init(model_args=self._model_args,
                                      training_args=self._training_args)

    @property
    def training_plan(self):
        return self._training_plan

    @property
    def model(self):
        return self._training_plan.model()

    @property
    def training_args(self):
        return self._localjob_training_args

    @training_args.setter
    def training_args(self, training_args: dict):
        self._localjob_training_args = training_args

    def start_training(self):
        """Sends training task to nodes and waits for the responses"""
        # Run import statements (very unsafely).
        for i in self._training_plan._dependencies:
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

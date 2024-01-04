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

        self._researcher_id = environ['RESEARCHER_ID']

        # List of node ID of the nodes used in the current round
        # - initially None (no current round yet)
        # - then updated during the round with the list of nodes to be used in the round, then the nodes
        #   that actually replied during the round
        self._nodes: Optional[List[str]] = nodes

        if keep_files_dir:
            self._keep_files_dir = keep_files_dir
        else:
            self._keep_files_dir = tempfile.mkdtemp(prefix=environ['TMP_DIR'])
            atexit.register(lambda: shutil.rmtree(self._keep_files_dir))  # remove directory

        if reqs is None:
            self._reqs = Requests()
        else:
            self._reqs = reqs

        self.last_msg = None

        self._training_plan_module = 'my_model_' + str(uuid.uuid4())
        self._training_plan_file = os.path.join(self._keep_files_dir, self._training_plan_module + '.py')

    def get_default_constructed_tp_instance(self,
                                            training_plan_class: Type[Callable],
                                            ) -> 'fedbiomed.common.training_plans.BaseTrainingPlan':

        # create TrainingPlan instance
        training_plan = training_plan_class()  # contains TrainingPlan

        # save and load training plan to a file to be sure
        # 1. a file is associated to training plan, so we can read its source, etc.
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

    def _save_tp_code_to_file(self,
                              training_plan: 'fedbiomed.common.training_plans.BaseTrainingPlan'
                              ) -> None:
        try:
            training_plan.save_code(self._training_plan_file)
        except Exception as e:
            msg = f"Cannot save the training plan to a local tmp dir : {str(e)}"
            raise FedbiomedTrainingPlanError(msg)

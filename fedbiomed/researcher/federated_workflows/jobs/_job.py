# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manage the training part of the experiment."""

import atexit
import os
import shutil
import tempfile
import uuid
from typing import Callable, List, Optional, Type

from fedbiomed.common import utils
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedJobError
from fedbiomed.common.logger import logger

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests


class Job:
    """
    Job represents a task to be executed on the node.
    """

    def __init__(self,
                 reqs: Requests = None,
                 nodes: Optional[List[str]] = None,
                 keep_files_dir: str = None):

        """ Constructor of the class

        Args:
            reqs: Researcher's requests assigned to nodes. Defaults to None.
            nodes: A dict of node_id containing the nodes used for training
            keep_files_dir: Directory for storing files created by the job that we want to keep beyond the execution
                of the job. Defaults to None, files are not kept after the end of the job.

        """

        self._researcher_id = environ['RESEARCHER_ID']

        # List of node ids participating in this task
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
    def requests(self):
        return self._reqs

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: dict):
        self._nodes = nodes

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manage the training part of the experiment."""

import os
import uuid
from typing import Callable, List, Optional, Type

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.common import utils
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedJobError
from fedbiomed.common.logger import logger


class Job:
    """
    Job represents a task to be executed on the node.

    This is a base class that provides the basic functionality necessary to establish communication with the remote
    nodes. Actual tasks should inherit from `Job` to implement their own domain logic.

    !!! info "Functional life-cycle"
        Jobs must follow a "functional" life-cycle, meaning that they should be created just before the execution of
        the task, and destroyed shortly after. Jobs should not persist outside the scope of the function that requested
        the execution of the task.

    Attributes:
        requests: read-only [`Requests`][fedbiomed.researcher.requests.Requests] object handling communication with remote nodes
        nodes: node IDs participating in the task
    """

    def __init__(self,
                 nodes: List[str] | None,
                 keep_files_dir: str):

        """ Constructor of the class

        Args:
            nodes: A dict of node_id containing the nodes used for training
            keep_files_dir: Directory for storing files created by the job that we want to keep beyond the execution
                of the job. 

        """

        self._researcher_id = environ['RESEARCHER_ID']
        self._reqs = Requests()
        self._nodes: List[str] = nodes or []  # List of node ids participating in this task
        self._keep_files_dir = keep_files_dir
        self._policies = None

    @property
    def requests(self):
        return self._reqs


    def _get_default_constructed_tp_instance(self,
                                             training_plan_class: Type[Callable],
                                             ) -> BaseTrainingPlan:
        """Returns a default-constructed instance of the training plan class.

        !!! note "Saves to temporary file"
            This function saves the code of the training plan to a temporary file.

        Assumptions:

        - the `training_plan_class` is a class, inheriting from
            [`BaseTrainingPlan`][fedbiomed.common.training_plan.BaseTrainingPlan] that can be default-constructed

        Returns:
            Default-constructed object of type `training_plan_class`
        """

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
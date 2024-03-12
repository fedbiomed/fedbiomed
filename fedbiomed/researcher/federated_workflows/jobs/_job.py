# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manage the training part of the experiment."""

import atexit
import shutil
import tempfile
from typing import List, Optional, Any

from abc import ABC, abstractmethod

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests


class Job(ABC):
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
                 *,
                 nodes: Optional[List[str]] = None,
                 keep_files_dir: str = None,
                 ):

        """ Constructor of the class

        Args:
            nodes: A dict of node_id containing the nodes used for training
            keep_files_dir: Directory for storing files created by the job that we want to keep beyond the execution
                of the job. Defaults to None, files are not kept after the end of the job.

        """

        self._researcher_id = environ['RESEARCHER_ID']
        self._reqs = Requests()
        self.last_msg = None
        self._nodes: List[str] = nodes or []  # List of node ids participating in this task

        if keep_files_dir:
            self._keep_files_dir = keep_files_dir
        else:
            self._keep_files_dir = tempfile.mkdtemp(prefix=environ['TMP_DIR'])

    @property
    def requests(self):
        return self._reqs

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: List[str]):
        self._nodes = nodes

    @abstractmethod
    def execute(self) -> Any:
        """Payload of the job.

        Completes a request to the job's nodes and collects replies.

        Returns:
            values specific to the type of job
        """

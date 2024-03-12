# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manage the training part of the experiment."""

import time
from typing import Callable, Dict, List, Optional, Type

from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.requests import Requests


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
        self._timer = Job.NodeTimer(self._nodes)

    @property
    def requests(self):
        return self._reqs
    

    class NodeTimer:
        """Context manager that computes the processing time while entering and exiting for each node

        Usage:
        ```
        nodes = ['node_1', 'node_2']
        job = Job(nodes, file)
        with job._timer():
            # ... send some request
        
        job._timer.get_timer()
        # {node_1: 2.22, node_2: 2.21}
        ```
        
        """
        def __init__(self, nodes: List[str]):
            """
            Constructor of NodeTimer
            
            Args:
                nodes: existing nodes that will be requested for the Job
            """
            self._timer = {node_id: 0.  for node_id in nodes}


        def __enter__(self):
            self._timer.update({node_id: time.perf_counter() for node_id in self._timer.keys()})
        
        def __exit__(self, type, value, traceback):
            self._timer.update({node_id: time.perf_counter() - self._timer[node_id] for node_id in self._timer.keys()})

        def get_timer(self) -> Dict:
            return self._timer

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manage the training part of the experiment."""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List

from fedbiomed.researcher.requests import RequestPolicy, Requests


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
        requests: read-only [`Requests`][fedbiomed.researcher.requests.Requests] object handling communication with
                 remote nodes
        nodes: node IDs participating in the task
    """

    def __init__(
        self,
        *,
        researcher_id: str,
        requests: Requests,
        nodes: List[str] | None,
        keep_files_dir: str
    ):
        """Constructor of the class

        Args:
            researcher_id: Unique ID of the researcher
            requests: Object for handling communications
            nodes: A dict of node_id containing the nodes used for training
            keep_files_dir: Directory for storing files created by the job that we want to keep beyond the execution
                of the job.

        """

        self._researcher_id = researcher_id
        self._reqs = requests
        self._nodes: List[str] = nodes or []  # List of node ids participating in this task
        self._keep_files_dir = keep_files_dir
        self._policies: List[RequestPolicy] | None = None

    @property
    def requests(self) -> List[RequestPolicy] | None:
        return self._reqs

    @property
    def nodes(self) -> List[str]:
        return self._nodes

    # FIXME: this method is very basic, and doesnot compute the total time of request since it waits for all requests
    # before computing elapsed time
    class RequestTimer:
        """Context manager that computes the processing time elapsed for the request and the reply

        Usage:
        ```
        nodes = ['node_1', 'node_2']
        job = Job(nodes, file)
        with job._timer() as my_timer:
            # ... send some request

        my_timer
        # {node_1: 2.22, node_2: 2.21} # request time for each Node in second
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
            return self._timer

        def __exit__(self, type, value, traceback):
            self._timer.update({node_id: time.perf_counter() - self._timer[node_id] for node_id in self._timer.keys()})
            return self._timer


    @abstractmethod
    def execute(self) -> Any:
        """Payload of the job.

        Completes a request to the job's nodes and collects replies.

        Returns:
            values specific to the type of job
        """

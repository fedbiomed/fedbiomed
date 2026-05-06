# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
implementation of the base Job class of the node component
"""

from abc import ABC, abstractmethod

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.message import ErrorMessage, RequestReply
from fedbiomed.node.dataset_manager import DatasetManager


class _InternalJobError(FedbiomedError):
    """Internal error raised during job execution on node."""


class _BaseJob(ABC):
    """
    This class represents the payload executed by a node in a job.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_manager: DatasetManager,
        node_id: str,
        node_name: str,
        request: RequestReply,
    ) -> None:
        """Constructor of the class

        Args:
            root_dir: Root fedbiomed directory where node instance files will be stored.
            dataset_manager: DatasetManager instance to retrieve datasets
            node_id: Node id
            node_name: Node name (Hospital name)
            request: Message object containing all information about the task
        """
        self._dir = root_dir
        self._dataset_manager = dataset_manager
        self._node_id = node_id
        self._node_name = node_name
        self._dataset_id = request.dataset_id
        self._researcher_id = request.researcher_id  # we can assume this exists
        self._request_id = request.request_id
        self._message = request.__class__.__name__

    def _build_error_msg(
        self, msg: str, errnum: str = ErrorNumbers.FB313.value
    ) -> ErrorMessage:
        """Build error message for job failure."""
        return ErrorMessage(
            request_id=self._request_id,
            researcher_id=self._researcher_id,
            node_id=self._node_id,
            node_name=self._node_name,
            extra_msg=msg,
            errnum=errnum,
        )

    @abstractmethod
    def run(self) -> RequestReply | ErrorMessage:
        """Run job and return reply message or ErrorMessage in case of failure."""

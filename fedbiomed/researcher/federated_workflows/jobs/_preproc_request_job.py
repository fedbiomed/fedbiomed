# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Preprocess Job class of the researcher component
"""

from typing import Dict

from fedbiomed.common.constants import PreprocType
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ErrorMessage, PreprocReply, PreprocRequest
from fedbiomed.researcher.requests import MessagesByNode

from ._job import Job


class PreprocRequestJob(Job):
    """Preprocessing Request Job class.

    This class represents a preprocessing request job in the Fed-BioMed framework.
    It inherits from the base Job class and is used to handle preprocessing requests.
    Attributes:
        preproc_request: The preprocessing request associated with this job.
    """

    def __init__(
        self,
        preproc_id: str,
        experiment_id: str,
        federated_dataset: dict,
        preproc_type: PreprocType,
        preproc_step: int,
        preproc_args: dict,
        **kwargs,
    ) -> None:
        """Initialize the FARequestJob with the given FA request.

        Args:
            fa_request: The FA request to be associated with this job.
        """
        super().__init__(**kwargs)

        self._experiment_id = experiment_id
        self._federated_dataset = federated_dataset

        self._preproc_id = preproc_id
        self._preproc_type = preproc_type
        self._preproc_step = preproc_step
        self._preproc_args = preproc_args

    def execute(self) -> Dict[str, PreprocReply]:
        """Executes preprocessing request

        Returns:
            A dictionary mapping node IDs to their respective preprocessing replies.
        """
        preproc_request = dict(
            researcher_id=self._researcher_id,
            experiment_id=self._experiment_id,
            preproc_id=self._preproc_id,
            preproc_type=self._preproc_type,
            preproc_step=self._preproc_step,
            preproc_args=self._preproc_args,
            state_id="",  # To be updated if state management is implemented
        )

        requests = MessagesByNode()
        for node in self._nodes:
            requests[node] = PreprocRequest(
                **{
                    **preproc_request,
                }
            )

        with self._reqs.send(requests, self._nodes, self._policies) as responses:
            errors: Dict[str, ErrorMessage] = responses.errors()
            replies: Dict[str, PreprocReply] = responses.replies()

        if errors:
            # Handle errors appropriately (logging, raising exceptions, etc.)
            for node_id, error in errors.items():
                logger.error(
                    "Error message received during federated analytics "
                    f"in node_id={node_id}: {error.errnum}. {error.extra_msg}"
                )

        logger.debug(f"Replies are: {replies} for step {self._preproc_step}")

        return replies

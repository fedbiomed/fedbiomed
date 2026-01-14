# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Preprocess Job class of the researcher component
"""

from copy import deepcopy
from typing import Dict

from fedbiomed.common.constants import PreprocStep, PreprocType
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ErrorMessage, PreprocReply, PreprocRequest
from fedbiomed.researcher.requests import MessagesByNode

from ._job import Job


class PreprocRequestJob(Job):
    """Preprocessing Request Job class.

    This class represents a preprocessing request job in the Fed-BioMed framework.
    It inherits from the base Job class and is used to handle preprocessing requests.
    """

    def __init__(
        self,
        experiment_id: str,
        preproc_type: PreprocType,
        preproc_step: PreprocStep,
        preproc_id: str,
        federated_dataset: dict,
        preproc_args: dict,
        state_id: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the FARequestJob with the given FA request.
        Args:
            experiment_id: The experiment ID associated with this preprocessing job.
            preproc_type: The type of preprocessing to be performed.
            preproc_step: The step of preprocessing to be executed.
            preproc_id: The unique identifier for the preprocessing task.
            federated_dataset: The federated dataset on which preprocessing is to be performed.
            preproc_args: The arguments required for the preprocessing task.
            state_id: Optional dictionary mapping node IDs to their respective state IDs.
            **kwargs: Named arguments of parent class. Please see
                [`Job`][fedbiomed.researcher.federated_workflows.jobs.Job]
        """
        super().__init__(**kwargs)

        self._experiment_id = experiment_id
        self._preproc_type = preproc_type.value
        self._preproc_step = preproc_step.value
        self._preproc_id = preproc_id
        self._federated_dataset = federated_dataset
        self._preproc_args = preproc_args
        self._state_id = state_id or {}

    def execute(self) -> Dict[str, PreprocReply]:
        """Executes preprocessing request

        Returns:
            A dictionary mapping node IDs to their respective preprocessing replies.
        """
        preproc_request = dict(
            researcher_id=self._researcher_id,
            experiment_id=self._experiment_id,
            preproc_type=self._preproc_type,
            preproc_step=self._preproc_step,
            preproc_id=self._preproc_id,
        )

        requests = MessagesByNode()
        for node in self._nodes:
            # Note: deepcopy is used to avoid mutation issues across nodes
            # may be later reconsidered depending on size and content of preproc_args
            preproc_args = deepcopy(self._preproc_args)

            requests[node] = PreprocRequest(
                **{
                    **preproc_request,
                    "dataset_id": self._federated_dataset.data()[node]["dataset_id"],
                    "state_id": self._state_id.get(node),
                    "preproc_args": preproc_args,
                }
            )

        with self._reqs.send(requests, self._nodes, self._policies) as responses:
            errors: Dict[str, ErrorMessage] = responses.errors()
            replies: Dict[str, PreprocReply] = responses.replies()

        if errors:
            # Handle errors appropriately (logging, raising exceptions, etc.)
            for node_id, error in errors.items():
                logger.error(
                    "Error message received during preprocessing "
                    f"in node_id={node_id}: {error.errnum}. {error.extra_msg}"
                )

        logger.debug(f"Replies are: {replies} for step {self._preproc_step}")

        return replies

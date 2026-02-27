# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from fedbiomed.common.constants import Stats
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest
from fedbiomed.researcher.requests import MessagesByNode

from ._job import Job


class FARequestJob(Job):
    """FA Request Job class.

    This class represents a Federated Analytics (FA) request job in the Fed-BioMed framework.
    It inherits from the base Job class and is used to handle FA requests.

    Attributes:
        fa_request: The FA request associated with this job.
    """

    def __init__(
        self,
        experiment_id: str,
        fa_id: str,
        federated_dataset: dict,
        fa_args: dict,
        stats: Stats,
        dataset_args: dict,
        dataset_schema: list,
        **kwargs,
    ) -> None:
        """Initialize the FARequestJob with the given FA request.

        Args:
            fa_request: The FA request to be associated with this job.
        """
        super().__init__(**kwargs)

        self._experiment_id = experiment_id
        self._fa_id = fa_id
        self._federated_dataset = federated_dataset
        self._fa_args = fa_args
        self._dataset_args = dataset_args
        self._dataset_schema = dataset_schema
        self._stats = stats

    def execute(self) -> Dict[str, FAReply]:
        """Executes federated analytics request

        Returns:
            A dictionary mapping node IDs to their respective FA replies.
        """
        fa_request = dict(
            researcher_id=self._researcher_id,
            experiment_id=self._experiment_id,
            stats=self._stats,
            fa_id=self._fa_id,
            fa_args=self._fa_args,
            dataset_args=self._dataset_args,
            dataset_schema=self._dataset_schema,
        )

        requests = MessagesByNode()
        for node in self._nodes:
            requests[node] = FARequest(
                **{
                    **fa_request,
                    "dataset_id": self._federated_dataset.data()[node]["dataset_id"],
                }
            )

        with self._reqs.send(requests, self._nodes, self._policies) as responses:
            errors: Dict[str, ErrorMessage] = responses.errors()
            replies: Dict[str, FAReply] = responses.replies()

        if errors:
            # Handle errors appropriately (logging, raising exceptions, etc.)
            for node_id, error in errors.items():
                logger.error(
                    "Error message received during federated analytics "
                    f"in node_id={node_id}: {error.errnum}. {error.extra_msg}"
                )

        return replies, errors

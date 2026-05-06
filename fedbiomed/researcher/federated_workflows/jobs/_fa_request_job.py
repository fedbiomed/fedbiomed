# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.requests import MessagesByNode

from ._job import Job


class FARequestJob(Job):
    """Federated Analytics (FA) request job.

    Sends an FA request to all participating nodes and collects their replies.
    """

    def __init__(
        self,
        experiment_id: str,
        fa_id: str,
        federated_dataset: FederatedDataSet,
        stats_args: Optional[dict],
        stats: Optional[list],
        dataset_schema: Optional[list],
        **kwargs,
    ) -> None:
        """Initialize the FARequestJob.

        Args:
            experiment_id: ID of the experiment this job belongs to.
            fa_id: Unique identifier for this FA job.
            federated_dataset: Federated dataset mapping node IDs to dataset metadata.
            stats_args: Keyword arguments passed to the statistics functions.
            stats: List of statistic names to compute.
            dataset_schema: Optional schema descriptor used to validate the dataset.
            **kwargs: Forwarded to the base `Job` constructor.
        """
        super().__init__(**kwargs)

        # Validate that exactly one stats specification is provided and that schema is not mixed with stats_args.
        if not stats and not stats_args:
            raise FedbiomedError(
                "At least one of 'stats' or 'stats_args' must be provided."
            )
        if stats_args and stats:
            raise FedbiomedError(
                "'stats_args' and 'stats' are mutually exclusive. "
                "Use 'stats_args' for fine-grained control or 'stats' for a flat list."
            )
        if stats_args and dataset_schema is not None:
            raise FedbiomedError(
                "'stats_args' and 'dataset_schema' are mutually exclusive. "
                "Schema selection is encoded in the structure of 'stats_args'."
            )

        self._experiment_id = experiment_id
        self._fa_id = fa_id
        self._federated_dataset = federated_dataset
        self._stats_args = stats_args
        self._dataset_schema = dataset_schema
        self._stats = stats

    def execute(self) -> Dict[str, FAReply]:
        """Send the FA request to all nodes and collect replies.

        Raises:
            FedbiomedError: If any node returns an error, or if no replies are received.

        Returns:
            A dictionary mapping node IDs to their `FAReply`.
        """
        fa_request = dict(
            researcher_id=self._researcher_id,
            experiment_id=self._experiment_id,
            stats=self._stats,
            fa_id=self._fa_id,
            stats_args=self._stats_args,
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
            # Log errors encountered during the execution of the FA request
            for node_id, error in errors.items():
                logger.error(
                    f"Node {node_id} analytics error [{error.errnum}]: {error.extra_msg}"
                )
            # Treat any node error as fatal
            raise FedbiomedError(
                f"FA request execution failed with errors from nodes: {', '.join(errors.keys())}"
            )

        # Ensure there is at least one successful reply
        if not replies:
            raise FedbiomedError("No successful replies received.")

        return replies

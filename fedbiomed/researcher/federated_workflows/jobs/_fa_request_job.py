from typing import Dict

from fedbiomed.common.logger import logger
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest

from ._job import Job


class FARequestJob(Job):
    """FA Request Job class.

    This class represents a Federated Analytics (FA) request job in the Fed-BioMed framework.
    It inherits from the base Job class and is used to handle FA requests.

    Attributes:
        fa_request: The FA request associated with this job.
    """

    def __init__(self, experiment_id, fa_id, fa_args) -> None:
        """Initialize the FARequestJob with the given FA request.

        Args:
            fa_request: The FA request to be associated with this job.
        """
        self._experiment_id = experiment_id
        self._fa_id = fa_id
        self._fa_args = fa_args

        super().__init__()

    def execute(self) -> Dict[str, FAReply]:
        fa_request = dict(
            researcher_id=self._researcher_id,
            experiment_id=self.experiment_id,
            analytics_type="mean",
            fa_id=self._fa_id,
            fa_args=self._fa_args,
        )

        requests = {}
        for node in self._nodes:
            requests[node] = FARequest(
                {**fa_request, "dataset_id": self._data.data()[node].dataset_id}
            )

        with self._reqs.send(requests, self._nodes, self._policies) as responses:
            errors: Dict[str, ErrorMessage] = responses.errors()
            replies: Dict[str, FAReply] = responses.replies()

        if errors:
            # Handle errors appropriately (logging, raising exceptions, etc.)
            for node_id, error in errors.items():
                logger.log(
                    "Error message received during federated analytics "
                    f"in node_id={node_id}: {error.errnum}. {error.extra_msg}"
                )

        return replies, errors

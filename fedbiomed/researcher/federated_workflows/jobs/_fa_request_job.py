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

    def __init__(self, experiment_id, fa_id, fa_arguments) -> None:
        """Initialize the FARequestJob with the given FA request.

        Args:
            fa_request: The FA request to be associated with this job.
        """
        self._fa_arguments = fa_arguments
        self._experiment_id = experiment_id
        self._fa_id = fa_id

        super().__init__()

    def execute(self) -> Dict[str, FAReply]:
        fa_request = FARequest(
            researcher_id=self._researcher_id,
            experiment_id=self.experiment_id,
            fa_id=self.fa_id,
            fa_arguments=self.fa_arguments,
        )

        with self._reqs.send(fa_request, self._nodes, self._policies) as responses:
            errors: Dict[str, ErrorMessage] = responses.errors()
            replies: Dict[str, FAReply] = responses.replies()

        if errors:
            # Handle errors appropriately (logging, raising exceptions, etc.)
            for node_id, error in errors.items():
                logger.log(
                    f"Error from node during federeated analytics{node_id}: {error}"
                )

        return replies, errors

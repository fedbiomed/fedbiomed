# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
implementation of Preprocess Job class of the node component
"""

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.message import ErrorMessage, PreprocReply, PreprocRequest

from ._base_job import _BaseJob


class PreprocJob(_BaseJob):
    """
    This class represents the preprocessing part executed by a node in a given preprocessing job.
    """

    def __init__(
        self,
        root_dir: str,
        db_path: str,
        node_id: str,
        node_name: str,
        request: PreprocRequest,
    ) -> None:
        """Constructor of the class

        Args:
            root_dir: Root fedbiomed directory where node instance files will be stored.
            db_path: Path to node database file.
            node_id: Node id
            node_name: Node name (Hospital name)
            request: FARequest message object containing all information about the FA task
        """
        super().__init__(root_dir, db_path, node_id, node_name, request)

        self.experiment_id = request.experiment_id
        self.preproc_type = request.preproc_type
        self.preproc_step = request.preproc_step
        self.preproc_id = request.preproc_id
        self.preproc_args = request.preproc_args
        self.state_id = request.state_id

    def _build_args_for_dataset(self, dataset_entry):
        pass

    def run(self) -> PreprocReply | ErrorMessage:
        """Execute preprocessing job.

        Returns:
            PreprocReply message if successful, ErrorMessage otherwise.
        """
        try:
            # Placeholder for actual preprocessing logic
            # This is where the preprocessing would be performed
            # For now, we just simulate a successful preprocessing step

            # TODO: Parse request message and implement actual preprocessing logic
            # Simulated request of preprocessing
            msg = (
                f"Node {self._node_name} ({self._node_id}): "
                f"Preprocessing step {self.preproc_step} of type {self.preproc_type} "
                f"for experiment {self.experiment_id} with args {self.preproc_args}"
            )

            # Simulated output of preprocessing
            preproc_output = {
                "details": f"Preprocessing step {self.preproc_step} completed.",
                "preproc_step": (self.preproc_step + 1),
                "new_parameters": (self.preproc_args["parameters"] + 5),  # Placeholder
            }

            self.preproc_step += 1  # Increment step for the reply

            return PreprocReply(
                request_id=self._request_id,
                researcher_id=self._researcher_id,
                experiment_id=self.experiment_id,
                node_id=self._node_id,
                node_name=self._node_name,
                msg=msg,
                preproc_output=preproc_output,
                state_id=self.state_id,
            )

        except Exception as e:
            return self._build_error_msg(
                f"Preprocessing job failed: {str(e)}",
                errnum=ErrorNumbers.FB326.value,
            )

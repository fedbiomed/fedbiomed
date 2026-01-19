# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Preprocess Job class of the node component
"""

from fedbiomed.common.constants import ErrorNumbers, HarmonizationStep, PreprocType
from fedbiomed.common.message import ErrorMessage, PreprocReply, PreprocRequest
from fedbiomed.node.dataset_manager import DatasetManager

from ._base_job import _BaseJob


class PreprocJob(_BaseJob):
    """
    This class represents the preprocessing part executed by a node in a given preprocessing job.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_manager: DatasetManager,
        node_id: str,
        node_name: str,
        request: PreprocRequest,
        allow_preproc: bool,
    ) -> None:
        """Constructor of the class

        Args:
            root_dir: Root fedbiomed directory where node instance files will be stored.
            dataset_manager: DatasetManager instance to retrieve datasets
            node_id: Node id
            node_name: Node name (Hospital name)
            request: FARequest message object containing all information about the FA task
            allow_preproc: True if preprocessing is allowed on this node, False otherwise
        """
        super().__init__(root_dir, dataset_manager, node_id, node_name, request)

        self._experiment_id = request.experiment_id
        self._preproc_type_raw = request.preproc_type
        self._preproc_step_raw = request.preproc_step
        self._preproc_id = request.preproc_id
        self._preproc_args_raw = request.preproc_args
        self._state_id = request.state_id
        self._allow_preproc = allow_preproc

    def _build_args_for_dataset(self, dataset_entry):
        pass

    def run(self) -> PreprocReply | ErrorMessage:
        """Execute preprocessing job.

        Returns:
            PreprocReply message if successful, ErrorMessage otherwise.
        """
        if not self._allow_preproc:
            return self._build_error_msg(
                "Preprocessing is not allowed on this node by node configuration.",
                errnum=ErrorNumbers.FB326.value,
            )

        # Further check message parameters if needed
        try:
            self._preproc_type = PreprocType(self._preproc_type_raw)
        except ValueError:
            return self._build_error_msg(
                f"Received invalid preproc_type: {self._preproc_type_raw}",
                errnum=ErrorNumbers.FB326.value,
            )
        try:
            self._preproc_step = HarmonizationStep(self._preproc_step_raw)
        except ValueError:
            return self._build_error_msg(
                f"Received invalid preproc_step: {self._preproc_step_raw}",
                errnum=ErrorNumbers.FB326.value,
            )
        # To be added in real implementation: check content of preproc_args
        self._preproc_args = self._preproc_args_raw

        # Placeholder for actual preprocessing logic
        # This is where the preprocessing would be performed
        # For now, we just simulate a successful preprocessing step

        # TODO: Parse request message and implement actual preprocessing logic
        # Simulated request of preprocessing
        msg = (
            f"Node {self._node_name} ({self._node_id}): "
            f"Preprocessing step {self._preproc_step.name} of type {self._preproc_type.name} "
            f"for experiment {self._experiment_id} with args {self._preproc_args}"
        )

        # Simulated output of preprocessing
        preproc_output = {
            "dummy": f"Preprocessing step {self._preproc_step.name} completed.",
        }

        try:
            return PreprocReply(
                request_id=self._request_id,
                researcher_id=self._researcher_id,
                experiment_id=self._experiment_id,
                node_id=self._node_id,
                node_name=self._node_name,
                msg=msg,
                preproc_output=preproc_output,
                state_id=self._state_id,
            )
        except Exception as e:
            return self._build_error_msg(
                f"Preprocessing job failed: {str(e)}",
                errnum=ErrorNumbers.FB326.value,
            )

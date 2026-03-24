# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Preprocess Job class of the node component
"""

from typing import Callable, Dict

from fedbiomed.common.constants import ErrorNumbers, HarmonizationStep, PreprocType
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ErrorMessage, PreprocReply, PreprocRequest
from fedbiomed.node.dataset_manager import DatasetManager

from ._base_job import _BaseJob
from ._fedcombat_jobs import _FedComBat_jobs

_preproc_type_to_jobs: Dict[PreprocType, Callable] = {
    # PreprocType.NONE is not valid here
    PreprocType.FEDCOMBAT: _FedComBat_jobs,
    # To be added in the future: other preprocessing types and their corresponding job classes
}


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
        # Here we can check content of some preproc_args
        # Only checks common to all preproc types and steps can be implemented here,
        # otherwise the check should be done in the specific preproc implementation
        self._preproc_args = self._preproc_args_raw

        try:
            preproc_type_jobs = _preproc_type_to_jobs[self._preproc_type]
        except KeyError:
            return self._build_error_msg(
                f"Unsupported preprocessing type: {self._preproc_type.name}",
                errnum=ErrorNumbers.FB326.value,
            )

        try:
            preproc_job_class = preproc_type_jobs()
            preproc_output = preproc_job_class(self._preproc_step, self._preproc_args)
        except Exception as e:
            return self._build_error_msg(
                f"Preprocessing job failed: {str(e)}",
                errnum=ErrorNumbers.FB326.value,
            )

        msg = (
            f"Node {self._node_name} ({self._node_id}): "
            f"Preprocessing step {self._preproc_step.name} of type {self._preproc_type.name} "
            f"for experiment {self._experiment_id} with args {self._preproc_args}"
        )
        exclude_args = ["biological_model", "global_bias_model"]
        filtered_args = {
            k: v for k, v in self._preproc_args.items() if k not in exclude_args
        }
        if isinstance(preproc_output, dict):
            preproc_output_summary = {
                "type": type(preproc_output).__name__,
                "keys": list(preproc_output.keys()),
            }
        else:
            preproc_output_summary = {
                "type": type(preproc_output).__name__,
            }
        logger.info(
            f"Preprocessing executed successfully for {self._preproc_type.name} / {self._preproc_step.name} "
            f"with request id {self._request_id} preproc_id {self._preproc_id} dataset_id {self._dataset_id} "
            f"preproc_args {filtered_args} except {exclude_args} and output_summary {preproc_output_summary}"
        )
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
                f"Preprocessing job cannot reply: {str(e)}",
                errnum=ErrorNumbers.FB326.value,
            )

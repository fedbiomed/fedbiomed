# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
implementation of Federated Analytics Job class of the node component
"""

from typing import Dict

from fedbiomed.common.constants import AnalyticsTypes, ErrorNumbers
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest

from ._base_job import _BaseJob, _InternalJobError


class FAJob(_BaseJob):
    """
    This class represents the training part execute by a node in a given Federated Analytics job.
    """

    def __init__(
        self,
        root_dir: str,
        db_path: str,
        node_id: str,
        node_name: str,
        request: FARequest,
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

        self._analytics_type = request.analytics_type
        self._dataset_id = request.dataset_id
        self._experiment_id = request.experiment_id
        self._fa_id = request.fa_id
        self._fa_args = request.fa_args if request.fa_args is not None else {}

    def _build_args_for_dataset(self, dataset_entry: dict) -> dict:
        """Build arguments for dataset initialization.

        Args:
            dataset_entry: dataset entry from dataset manager

        Returns:
            Dict of arguments for dataset initialization
        """
        columns = list(dataset_entry.get("dtypes", {}).keys())
        if self._fa_args.get("col_names"):
            input_columns = self._fa_args["col_names"]
            if not all(col in columns for col in input_columns):
                raise _InternalJobError(
                    f"One or more invalid column names for federated analytics on node='{self._node_id}': "
                    f"requested columns '{input_columns}' not in available columns {columns}"
                )
        else:
            input_columns = columns

        return {"input_columns": input_columns}

    def run(self) -> FAReply | ErrorMessage:
        """Run FA job and return FAReply message or ErrorMessage in case of failure."""
        # Retrieve dataset ready-to-use from self._dataset_id

        if self._analytics_type not in [AnalyticsTypes.MEAN.value]:
            return self._build_error_msg(
                msg=f"Analytics type '{self._analytics_type}' not supported.",
                errnum=ErrorNumbers.FB325.value,
            )

        try:
            dataset = self._build_dataset(DataReturnFormat.SKLEARN, ["csv"])
        except _InternalJobError as e:
            return self._build_error_msg(msg=repr(e), errnum=ErrorNumbers.FB325.value)

        # TODO: implement FA job / needs to be adapted
        output: Dict = dataset.mean()

        return FAReply(
            request_id=self._request_id,
            researcher_id=self._researcher_id,
            experiment_id=self._experiment_id,
            fa_id=self._fa_id,
            analytics_type=self._analytics_type,
            node_id=self._node_id,
            node_name=self._node_name,
            output=output,
        )

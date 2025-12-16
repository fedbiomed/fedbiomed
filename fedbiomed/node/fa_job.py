# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
implementation of Federated Analytics Job class of the node component
"""

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest
from fedbiomed.node.dataset_manager import REGISTRY_CONTROLLERS, DatasetManager


class FAJob:
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
        self._dir = root_dir
        self._db_path = db_path
        self._node_id = node_id
        self._node_name = node_name
        self._dataset_id = request.dataset_id
        self._experiment_id = request.experiment_id
        self._fa_id = request.fa_id
        self._researcher_id = request.researcher_id
        self._request_id = request.request_id
        self._fa_args = request.fa_args if request.fa_args is not None else {}

    def _build_error_msg(self, msg: str, errnum: str) -> ErrorMessage:
        """Build error message for FA job failure."""
        return ErrorMessage(
            request_id=self._request_id,
            researcher_id=self._researcher_id,
            node_id=self._node_id,
            node_name=self._node_name,
            extra_msg=msg,
            errnum=errnum,
        )

    def _build_dataset(self) -> Dataset | ErrorMessage:
        """Build dataset instance ready-to-use from dataset id."""
        # dataset manager to get metadata about dataset, dlp and loading block
        dataset_manager = DatasetManager(self._db_path)

        # recover dataset entry
        dataset_entry = dataset_manager.dataset_table.get_by_id(self._dataset_id)
        if dataset_entry is None:
            msg = f"Did not found proper data in local datasets on node={self._node_id}"
            return self._build_error_msg(msg=msg, errnum=ErrorNumbers.FB313.value)

        # validate data type
        data_type = dataset_entry.get("data_type")
        # FIXME: temporal exception
        if data_type != "csv":
            return self._build_error_msg(
                msg=f"Data type '{data_type}' not supported yet.",
                errnum=ErrorNumbers.FB313.value,
            )

        if data_type not in REGISTRY_CONTROLLERS:
            msg = f"Data type '{data_type}' not supported."
            return self._build_error_msg(msg=msg, errnum=ErrorNumbers.FB313.value)

        # get controller parameters
        controller_kwargs = {
            "root": dataset_entry.get("path"),
            **dataset_entry.get("dataset_parameters", {}),
        }

        # recover dlp if any
        if "dlp_id" in dataset_entry:
            dlp_metadata = dataset_manager.get_dlp_by_id(dataset_entry["dlp_id"])
            try:
                controller_kwargs["dlp"] = DataLoadingPlan().deserialize(*dlp_metadata)
            except FedbiomedError as e:
                msg = f"Cannot recover dlp on node={self._node_id}: {repr(e)}"
                return self._build_error_msg(msg=msg, errnum=ErrorNumbers.FB313.value)

        # build dataset instance
        _, _, dataset_cls = REGISTRY_CONTROLLERS[data_type]
        dataset = dataset_cls()
        dataset.complete_initialization(controller_kwargs, DataReturnFormat.SKLEARN)

        return dataset

    def run(self) -> FAReply | ErrorMessage:
        """Run FA job and return FAReply message or ErrorMessage in case of failure."""
        # Retrieve dataset ready-to-use from self._dataset_id
        dataset = self._build_dataset()
        if isinstance(dataset, ErrorMessage):
            return dataset

        # TODO: implement FA job / needs to be adapted
        output = {"mean": dataset.mean()}

        return FAReply(
            request_id=self._request_id,
            researcher_id=self._researcher_id,
            experiment_id=self._experiment_id,
            fa_id=self._fa_id,
            node_id=self._node_id,
            node_name=self._node_name,
            output=output,
        )

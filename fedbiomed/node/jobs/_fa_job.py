# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Federated Analytics Job class of the node component
"""

from typing import Dict

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers, FedbiomedError, Stats
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ErrorMessage, FAReply, FARequest
from fedbiomed.node.dataset_manager import DatasetManager

from ._base_job import _BaseJob, _InternalJobError


class FAJob(_BaseJob):
    """
    This class represents the training part execute by a node in a given Federated Analytics job.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_manager: DatasetManager,
        node_id: str,
        node_name: str,
        request: FARequest,
        allow_fa: bool,
    ) -> None:
        """Constructor of the class

        Args:
            root_dir: Root fedbiomed directory where node instance files will be stored.
            dataset_manager: DatasetManager instance to retrieve datasets
            node_id: Node id
            node_name: Node name (Hospital name)
            request: FARequest message object containing all information about the FA task
            allow_fa: True if federated analytics is allowed on this node, False otherwise
        """
        super().__init__(root_dir, dataset_manager, node_id, node_name, request)

        self._stats = request.stats
        self._dataset_id = request.dataset_id
        self._experiment_id = request.experiment_id
        self._fa_id = request.fa_id
        self._fa_args = request.fa_args
        self._dataset_schema = request.dataset_schema
        self._allow_fa = allow_fa

    def _build_args_for_dataset(self, dataset_entry: dict) -> dict:
        """Generate dataset arguments based on the dataset type by default."""
        data_type = dataset_entry.get("data_type")
        dataset_type = DatasetTypes.get_type_by_value(data_type)

        match dataset_type:
            case None:
                # This should not happen, but this check is added for safety
                raise FedbiomedError(
                    f"Dataset entry contains unsupported dataset type '{data_type}'."
                )
            case DatasetTypes.TABULAR:
                # Take keys in 'dtypes' and pass them as ``input_columns`` to the dataset constructor
                return {"input_columns": list(dataset_entry.get("dtypes", {}))}
            case DatasetTypes.IMAGES | DatasetTypes.DEFAULT | DatasetTypes.MEDNIST:
                # For image datasets, no dataset arguments are passed by default
                return {}
            case DatasetTypes.MEDICAL_FOLDER:
                # Take keys in 'shape' and pass them as ``data_modalities`` to the dataset constructor
                return {"data_modalities": list(dataset_entry.get("shape", {}))}
            case _:
                raise FedbiomedError(
                    f"Dataset arguments by default are not implemented for dataset type '{data_type}'."
                )

    def run(self) -> FAReply | ErrorMessage:
        """Run FA job and return FAReply message or ErrorMessage in case of failure."""

        if not self._allow_fa:
            return self._build_error_msg(
                "Federated Analytics are not allowed on this node by node configuration.",
                errnum=ErrorNumbers.FB325.value,
            )

        # Validate that all requested stats are valid enum values
        if not isinstance(self._stats, list):
            self._stats = [self._stats]
        if not all(stat in [_.value for _ in Stats] for stat in self._stats):
            return self._build_error_msg(
                msg=f"Analytics type '{self._stats}' contain unsupported values.",
                errnum=ErrorNumbers.FB325.value,
            )

        try:
            dataset = self._build_dataset(DataReturnFormat.SKLEARN)
        except _InternalJobError as e:
            return self._build_error_msg(msg=repr(e), errnum=ErrorNumbers.FB325.value)

        # Use compute_stats to handle all statistics
        if not hasattr(dataset, "compute_stats"):
            return self._build_error_msg(
                msg="Dataset does not support analytics method 'compute_stats'.",
                errnum=ErrorNumbers.FB325.value,
            )

        logger.debug(
            f"FA Request received and database built, executing analytics: compute_stats (requested: {self._stats})"
        )

        # Prepare kwargs
        kwargs = {
            "stats": self._stats,
            "fa_args": self._fa_args,
            "dataset_schema": self._dataset_schema,
        }

        try:
            output: Dict = dataset.compute_stats(**kwargs)
            logger.debug(f"Analytics executed, output: {output}")
        except Exception as e:
            return self._build_error_msg(
                msg=(
                    f"Error during execution of analytics '{self._stats}' "
                    f"on node='{self._node_id}': {repr(e)}"
                ),
                errnum=ErrorNumbers.FB325.value,
            )

        return FAReply(
            request_id=self._request_id,
            researcher_id=self._researcher_id,
            experiment_id=self._experiment_id,
            fa_id=self._fa_id,
            stats=self._stats[0] if len(self._stats) == 1 else self._stats,
            node_id=self._node_id,
            node_name=self._node_name,
            output=output,
        )

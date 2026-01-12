# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Federated Analytics Job class of the node component
"""

from typing import Dict

from fedbiomed.common.analytics import (
    DatasetArgumentsFA,
    validate_dataset_arguments_for_fa,
)
from fedbiomed.common.constants import AnalyticsTypes, DatasetTypes, ErrorNumbers
from fedbiomed.common.dataset_types import DataReturnFormat
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
    ) -> None:
        """Constructor of the class

        Args:
            root_dir: Root fedbiomed directory where node instance files will be stored.
            dataset_manager: DatasetManager instance to retrieve datasets
            node_id: Node id
            node_name: Node name (Hospital name)
            request: FARequest message object containing all information about the FA task
        """
        super().__init__(root_dir, dataset_manager, node_id, node_name, request)

        self._analytics_type = request.analytics_type
        self._dataset_id = request.dataset_id
        self._experiment_id = request.experiment_id
        self._fa_id = request.fa_id
        self._fa_args = request.fa_args
        self._dataset_args = request.dataset_args

    def _build_args_for_dataset(self, dataset_entry: dict) -> dict:
        """Build arguments for dataset initialization.

        Args:
            dataset_entry: dataset entry from dataset manager

        Returns:
            Dict of arguments for dataset initialization
        """

        type_ = dataset_entry.get("data_type")
        dataset_type = DatasetTypes.get_type_by_value(type_)
        validate_dataset_arguments_for_fa(self._dataset_args, dataset_type)

        args = {}

        if self._dataset_args:
            for key, value in self._dataset_args.items():
                args.update(
                    {DatasetArgumentsFA[dataset_type].get(key).get("arg_name"): value}
                )

        return args

    def run(self) -> FAReply | ErrorMessage:
        """Run FA job and return FAReply message or ErrorMessage in case of failure."""
        # Retrieve dataset ready-to-use from self._dataset_id

        if self._analytics_type not in [t.value for t in AnalyticsTypes]:
            return self._build_error_msg(
                msg=f"Analytics type '{self._analytics_type}' not supported.",
                errnum=ErrorNumbers.FB325.value,
            )

        try:
            dataset = self._build_dataset(DataReturnFormat.SKLEARN)
        except _InternalJobError as e:
            return self._build_error_msg(msg=repr(e), errnum=ErrorNumbers.FB325.value)

        if hasattr(dataset, self._analytics_type) is False:
            return self._build_error_msg(
                msg=f"Dataset does not support analytics type '{self._analytics_type}'.",
                errnum=ErrorNumbers.FB325.value,
            )

        print(
            "FA Request received and database built, executing analytics:",
            self._analytics_type,
        )
        analytics = getattr(dataset, self._analytics_type)

        try:
            output: Dict = analytics(**self._fa_args if self._fa_args else {})
            print("Analytics executed, output:", output)
        except Exception as e:
            return self._build_error_msg(
                msg=(
                    f"Error during execution of analytics '{self._analytics_type}' "
                    f"on node='{self._node_id}': {repr(e)}"
                ),
                errnum=ErrorNumbers.FB325.value,
            )

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

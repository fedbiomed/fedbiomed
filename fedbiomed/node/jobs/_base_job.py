# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
implementation of the base Job class of the node component
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.message import ErrorMessage, RequestReply
from fedbiomed.node.dataset_manager import REGISTRY_CONTROLLERS, DatasetManager


class _InternalJobError(FedbiomedError):
    """Internal error raised during job execution on node."""


class _BaseJob(ABC):
    """
    This class represents the payload executed by a node in a job.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_manager: DatasetManager,
        node_id: str,
        node_name: str,
        request: RequestReply,
    ) -> None:
        """Constructor of the class

        Args:
            root_dir: Root fedbiomed directory where node instance files will be stored.
            dataset_manager: DatasetManager instance to retrieve datasets
            node_id: Node id
            node_name: Node name (Hospital name)
            request: Message object containing all information about the task
        """
        self._dir = root_dir
        self._dataset_manager = dataset_manager
        self._node_id = node_id
        self._node_name = node_name
        self._dataset_id = (
            request.dataset_id if hasattr(request, "dataset_id") else None
        )
        self._researcher_id = request.researcher_id  # we can assume this exists
        self._request_id = request.request_id
        self._message = request.__class__.__name__

    def _build_error_msg(
        self, msg: str, errnum: str = ErrorNumbers.FB313.value
    ) -> ErrorMessage:
        """Build error message for job failure."""
        return ErrorMessage(
            request_id=self._request_id,
            researcher_id=self._researcher_id,
            node_id=self._node_id,
            node_name=self._node_name,
            extra_msg=msg,
            errnum=errnum,
        )

    def _build_dataset(
        self,
        return_type: DataReturnFormat,
        data_types: Optional[List[str]],
    ) -> Dataset:
        """Build dataset instance ready-to-use from dataset id.

        Args:
            return_type: format in which data should be returned
            data_types: list of accepted data types for the dataset

        Raises:
            _InternalJobError: if dataset cannot be recovered or initialized.
        """
        if not self._dataset_id:
            raise _InternalJobError(
                f"This job {self.__class__.__name__} has no associated `dataset_id` "
                f"in message {self._message}."
            )

        # recover dataset entry
        dataset_entry = self._dataset_manager.dataset_table.get_by_id(self._dataset_id)
        if dataset_entry is None:
            raise _InternalJobError(
                f"Cannot found request dataset in local datasets: dataset_id='{self._dataset_id}' "
                f"on node='{self._node_id}'"
            )

        # validate data type
        data_type = dataset_entry.get("data_type")
        if data_types and data_type not in data_types:
            raise _InternalJobError(
                f"{data_type} not supported. {self.__class__.__name__} job currently "
                f"only supports data_types: {data_types}, got '{data_type}'"
            )

        if data_type not in REGISTRY_CONTROLLERS:
            raise _InternalJobError(
                f"Data type '{data_type}' not supported in jobs, available types: "
                f"{list(REGISTRY_CONTROLLERS.keys())}"
            )

        # get controller parameters
        controller_kwargs = {
            "root": dataset_entry.get("path"),
            **dataset_entry.get("dataset_parameters", {}),
        }

        # recover dlp if any
        if "dlp_id" in dataset_entry:
            dlp_metadata = self._dataset_manager.get_dlp_by_id(dataset_entry["dlp_id"])
            try:
                controller_kwargs["dlp"] = DataLoadingPlan().deserialize(*dlp_metadata)
            except FedbiomedError as e:
                raise _InternalJobError(
                    f"Cannot recover dlp on node={self._node_id}: {repr(e)}"
                ) from e

        # build dataset instance
        _, _, dataset_cls = REGISTRY_CONTROLLERS[data_type]

        # dataset = dataset_cls(input_columns=input_columns)
        dataset = dataset_cls(**self._build_args_for_dataset(dataset_entry))

        dataset.complete_initialization(controller_kwargs, return_type)

        return dataset

    @abstractmethod
    def _build_args_for_dataset(self, dataset_entry: dict) -> dict:
        """Build arguments for dataset initialization.

        Args:
            dataset_entry: dataset entry from dataset manager

        Returns:
            Dict of arguments for dataset initialization
        """

    @abstractmethod
    def run(self) -> RequestReply | ErrorMessage:
        """Run job and return reply message or ErrorMessage in case of failure."""

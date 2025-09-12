from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from tinydb import where

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.db import (
    DB,
    DatasetMetadata,
    Dlb,
    Dlp,
    ImagesMetadata,
    MedicalFolderDlp,
    MedicalFolderMetadata,
    MednistMetadata,
    MnistMetadata,
    TabularMetadata,
)
from fedbiomed.common.exceptions import FedbiomedError

dataset_types = ["csv", "default", "mednist", "images", "medical-folder", "flamby"]


class DlbDB(DB):
    """CRUD specialized for Data Loading Blocks documents keyed by 'dlb_id'."""

    def create(self, entry: Dlb) -> int:
        """Creates a new DLB entry.

        Args:
            entry: DLB dataclass instance to create (Dlb or subclass).
        Returns:
            The document ID of the created entry.
        Raises:
            FedbiomedError: If dlb_id is missing or already exists.
        """
        if not entry.dlb_id:
            raise FedbiomedError("DLB entry requires 'dlb_id'.")
        if self._database.search(where("dlb_id") == entry.dlb_id):
            raise FedbiomedError(f"DLB with id {entry.dlb_id} already exists.")

        entry_dict = asdict(entry)
        return self._database.create(entry_dict)

    def get_by_id(self, dlb_id: str) -> Optional[Dlb]:
        """Get a single DLB by dlb_id (or None if missing)."""
        result = self._get_by("dlb_id", dlb_id)
        if not result:
            return None
        return Dlb(**result)

    def delete_by_id(self, dlb_id: str) -> List[int]:
        """Delete by dlb_id. Returns the list of removed doc IDs."""
        return self._delete_by("dlb_id", dlb_id)

    def update_by_id(self, value: Dict[str, Any]) -> List[int]:
        """Partial update of a DLB entry by dlb_id. Returns list of updated doc IDs.

        Args:
            value: Dict containing at least 'dlb_id' and fields to update.
        Raises:
            FedbiomedError: If 'dlb_id' missing.
        """
        if not value.get("dlb_id"):
            raise FedbiomedError("DLB update requires 'dlb_id'.")
        return self._update_by("dlb_id", value)


class DlpDB(DB):
    """CRUD specialized for Data Loading Plans documents keyed by 'dlp_id'."""

    def create(self, entry: Dlp) -> int:
        """Creates a new DLP entry.

        Args:
            entry: DLP entry to create.
        Returns:
            The document ID of the created entry.
        Raises:
            FedbiomedError: If dlp_id is missing or already exists.
        """
        if not entry.dlp_id:
            raise FedbiomedError("Dataset entry requires 'dlp_id'.")
        if self._database.search(where("dlp_id") == entry.dlp_id):
            raise FedbiomedError(f"Dataset with id {entry.dlp_id} already exists.")

        # Convert dataclass to dictionary before inserting
        entry_dict = asdict(entry)

        return self._database.create(entry_dict)

    def get_by_id(self, dlp_id) -> Optional[Dlp]:
        """Get a single DLP by dlp_id (or None if missing)."""
        result = self._get_by("dlp_id", dlp_id)
        if not result:
            return None

        # Raise an error if the target_dataset_type is unknown
        if result.get("target_dataset_type") not in dataset_types:
            raise FedbiomedError(
                f"DLP with id {dlp_id} has invalid target_dataset_type "
                f"{result.get('target_dataset_type')}. "
                f"Should be one of {dataset_types}."
            )

        # Otherwise return the appropriate subclass based on the content
        if result.get("target_dataset_type") == "medical-folder":
            return MedicalFolderDlp(**result)

        # TODO: Check if this function should return Dlp and Dlbs in the new design
        return Dlp(**result)

    def delete_by_id(self, dlp_id: str) -> List[int]:
        """Delete by dlp_id. Returns the list of removed doc IDs."""
        return self._delete_by("dlp_id", dlp_id)

    def update_by_id(self, value: Dict[str, Any]) -> List[int]:
        """Update a DLP entry with the values in 'value'. Returns list of updated doc IDs."""
        return self._update_by("dlp_id", value)

    def list_by_dataset_type(self, dataset_type: str) -> List[Dict[str, Any]]:
        """List all DLPs for a given dataset type.

        Raises:
            FedbiomedError: if dataset_type is invalid."""
        if not isinstance(dataset_type, str):
            raise FedbiomedError(
                f"Wrong input type for target_dataset_type. "
                f"Expected str, got {type(dataset_type)} instead."
            )
        if dataset_type not in [t.value for t in DatasetTypes]:
            raise FedbiomedError(
                "target_dataset_type should be of the values defined in "
                "fedbiomed.common.constants.DatasetTypes"
            )
        return self._get_all_by("dataset_type", dataset_type)


class DatasetDB(DB):
    """CRUD specialized for dataset documents keyed by 'dataset_id'."""

    def search_conflicting_tags(self, tags: Union[tuple, list]) -> list:
        """Searches for registered data that have conflicting tags with the given tags

        Args:
            tags:  List of tags

        Raises:
            FedbiomedError: If there are datasets existing with conflicting tags.
        """

        def _conflicting_tags(val):
            return all(t in val for t in tags) or all(t in tags for t in val)

        conflicting = self._database.search(self._query.tags.test(_conflicting_tags))

        if len(conflicting) > 0:
            msg = (
                f"{ErrorNumbers.FB322.value}, one or more registered dataset has conflicting tags: "
                f" {' '.join([c['name'] for c in conflicting])}"
            )
            raise FedbiomedError(msg)

    def create(self, entry: DatasetMetadata) -> int:
        """Creates a new dataset entry.
        Args:
            entry: Dataset entry to create.
        Returns:
            The document ID of the created entry.
        Raises:
            FedbiomedError: If dataset_id is missing or already exists,
            or if tags conflict with existing dataset.
        """

        # Check that dataset_id is present and unique
        if not entry.dataset_id:
            raise FedbiomedError("Dataset entry requires 'dataset_id'.")
        if self._database.search(where("dataset_id") == entry.dataset_id):
            raise FedbiomedError(f"Dataset with id {entry.dataset_id} already exists.")

        # Check that there is not an existing dataset with conflicting tags
        self.search_conflicting_tags(entry.tags)

        # Check that name is unique if provided
        if entry.name is not None:
            if self._database.search(where("name") == entry.name):
                raise FedbiomedError(f"Dataset with name {entry.name} already exists.")

        # Convert dataclass to dictionary before inserting
        entry_dict = asdict(entry)

        return self._database.create(entry_dict)

    def get_by_id(self, dataset_id) -> Optional[DatasetMetadata]:
        """Get a single dataset by dataset_id (or None if missing)."""
        result = self._get_by("dataset_id", dataset_id)
        if not result:
            return None

        # Raise an error if the data_type is unknown
        if result.get("data_type") not in dataset_types:
            raise FedbiomedError(
                f"Dataset with id {dataset_id} has invalid target_dataset_type "
                f"{result.get('target_dataset_type')}. "
                f"Should be one of {dataset_types}."
            )

        # Otherwise return the appropriate subclass based on the content
        if result.get("data_type") == "medical-folder":
            return MedicalFolderMetadata(**result)
        elif result.get("data_type") == "images":
            return ImagesMetadata(**result)
        elif result.get("data_type") in "mednist":
            return MednistMetadata(**result)
        elif result.get("data_type") == "csv":
            return TabularMetadata(**result)
        elif result.get("data_type") == "default":
            return MnistMetadata(**result)
        else:
            return DatasetMetadata(**result)

    def get_by_tag(self, tags) -> List[DatasetMetadata]:
        """Get the list of datasets which contain all the given tags (or None if missing)."""
        result = self._get_all_by("tags", tags)
        return [
            DatasetMetadata(**r) for r in result
        ]  # Convert each document back to Dataset

    def delete_by_id(self, dataset_id: str) -> List[int]:
        """Delete by dataset_id. Returns the list of removed doc IDs."""
        return self._delete_by("dataset_id", dataset_id)

    def update_by_id(self, value: Dict[str, Any]) -> List[int]:
        """Update a Dataset entry with the values in 'value'. Returns list of updated doc IDs.

        Raises:
            FedbiomedError: If tags conflict with existing dataset.
        """
        if not value.get("dataset_id"):
            raise FedbiomedError("Dataset entry requires 'dataset_id'.")

        # Check that there is not an existing dataset with conflicting tags
        if value.get("tags") is not None:
            self.search_conflicting_tags(value.get("tags"))

        return self._update_by("dataset_id", value)

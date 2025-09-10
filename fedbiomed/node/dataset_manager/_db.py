from typing import Any, Dict, List, Optional, Union

from tinydb import where
from tinydb.table import Document

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.db import DB
from fedbiomed.common.exceptions import FedbiomedError


class DlpDB(DB):
    """CRUD specialized for Data Loading Plans documents keyed by 'dlp_id'."""

    def create(self, entry: Dict[str, Any]) -> int:
        """Creates a new DLP entry.

        Args:
            entry: DLP entry to create.
        Returns:
            The document ID of the created entry.
        Raises:
            FedbiomedError: If dlp_id is missing or already exists.
        """
        if not entry.get("dlp_id"):
            raise FedbiomedError("Dataset entry requires 'dlp_id'.")
        if self._database.search(where("dlp_id") == entry.get("dlp_id")):
            raise FedbiomedError(
                f"Dataset with id {entry.get('dlp_id')} already exists."
            )

        return self._database.create(entry)

    def get_by_id(self, dlp_id) -> Optional[Dict[str, Any]]:
        """Get a single dlp by dlp_id (or None if missing)."""
        return self._get_by("dlp_id", dlp_id)

    def delete_by_id(self, dlp_id: str) -> List[int]:
        """Delete by dlp_id. Returns the list of removed doc IDs."""
        return self._delete_by("dlp_id", dlp_id)

    def update_by_id(self, value: Dict[str, Any]) -> List[int]:
        """Update a DLP entry with the values in 'value'. Returns list of updated doc IDs."""
        return self._update_by("dlp_id", value)

    def list_by_dataset_type(self, dataset_type: str) -> List[Document]:
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
        return self._get_all_by("target_dataset_type", dataset_type)


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

    def create(self, entry: Dict[str, Any]) -> int:
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
        if not entry.get("dataset_id"):
            raise FedbiomedError("Dataset entry requires 'dataset_id'.")
        if self._database.search(where("dataset_id") == entry.get("dataset_id")):
            raise FedbiomedError(
                f"Dataset with id {entry.get('dataset_id')} already exists."
            )

        # Check that there is not an existing dataset with conflicting tags
        self.search_conflicting_tags(entry.get("tags"))

        # Check that name is unique if provided
        if entry.get("name") is not None:
            if self._database.search(where("name") == entry.get("name")):
                raise FedbiomedError(
                    f"Dataset with name {entry.get('name')} already exists."
                )

        return self._database.create(entry)

    def get_by_id(self, dataset_id) -> Optional[Dict[str, Any]]:
        """Get a single dataset by dataset_id (or None if missing)."""
        return self._get_by("dataset_id", dataset_id)

    def get_by_tag(self, tags) -> List[Document]:
        """Get the list of datasets which contain all the given tags (or None if missing)."""
        return self._get_all_by("tags", tags)

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

from typing import Any, List, Optional, Type, Union

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.db import TinyTableConnector
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._db_dataclasses import DatasetEntry, DlbEntry, DlpEntry


class BaseTable(TinyTableConnector):
    """Base class for all database tables with common validation functionality"""

    # Dataclass is expected to have from_dict and to_dict methods
    _dataclass: Type = None

    def _validate_and_convert_to_dict(self, data: dict) -> dict:
        """Generic validation using the table's dataclass"""
        try:
            # Single-pass validation and conversion
            entry = self._dataclass.from_dict(data)
            return entry.to_dict()
        except (KeyError, TypeError, ValueError) as e:
            model_name = self._dataclass.__name__
            raise FedbiomedError(f"{model_name} validation failed: {str(e)}") from e

    def insert(self, entry: dict) -> int:
        """Insert an entry with validation"""
        validated_entry = self._validate_and_convert_to_dict(entry)
        return super().insert(validated_entry)

    def get_validated_entry(self, entry_id: str) -> Optional[Any]:
        """Get entry by ID and return as validated model instance"""
        data = self.get_by_id(entry_id)
        return None if not data else self._dataclass.from_dict(data)

    def update_by_id(self, id_value, update):
        """Update an entry by its ID with validation"""
        # Fetch existing entry. If not found, raise error
        entry = self.get_by_id(id_value)
        if entry is None:
            raise FedbiomedError(f"No entry found with {self._id_name}={id_value}")
        # Merge and validate updated entry
        entry.update(update)
        updated_entry = self._validate_and_convert_to_dict(entry)
        return super().update_by_id(id_value, updated_entry)


class DatasetTable(BaseTable):
    _table_name = "Datasets"
    _id_name = "dataset_id"
    _dataclass = DatasetEntry

    def insert(self, entry: dict) -> int:
        """Insert a dataset entry with validation"""
        conflicting_tags = self.search_conflicting_tags(entry["tags"])
        if conflicting_tags:
            _msg = (
                f"{ErrorNumbers.FB322.value}: "
                "One or more registered dataset has conflicting tags: "
                f" {' '.join([c['name'] for c in conflicting_tags])}"
            )
            logger.critical(_msg)
            raise FedbiomedError(_msg)

        return super().insert(entry)

    def search_by_tags(self, tags: Union[tuple, list]) -> list:
        """Searches for data with given tags.

        Args:
            tags:  List of tags

        Returns:
            The list of matching datasets
        """
        return self.get_all_by_condition("tags", lambda x: set(tags).issubset(x))

    def search_conflicting_tags(self, tags: Union[tuple, list]) -> list:
        """Searches for registered data that have conflicting tags with the given tags

        Args:
            tags:  List of tags

        Returns:
            The list of conflicting datasets
        """
        return self.get_all_by_condition(
            "tags",
            lambda x: set(tags).issubset(x) or set(x).issubset(tags),
        )

    def modify_database_info(self, dataset_id: str, modified_dataset: dict):
        """Modifies a dataset in the database.

        Args:
            dataset_id: ID of the dataset to modify.
            modified_dataset: key-value pairs to replace in the existing entry.

        Raises:
            FedbiomedError: conflicting tags with existing dataset
        """
        # Check that there are not existing dataset with conflicting tags
        if "tags" in modified_dataset:
            # the dataset to modify is ignored (can conflict with its previous tags)
            conflicting_ids = [
                _["dataset_id"]
                for _ in self.search_conflicting_tags(modified_dataset["tags"])
                if _["dataset_id"] != dataset_id
            ]
            if len(conflicting_ids) > 0:
                msg = (
                    f"{ErrorNumbers.FB322.value}, one or more registered dataset has conflicting tags: "
                    f" {' '.join([_['name'] for _ in conflicting_ids])}"
                )
                logger.critical(msg)
                raise FedbiomedError(msg)

        _ = self.update_by_id(dataset_id, modified_dataset)


class DlpTable(BaseTable):
    _table_name = "Dlps"
    _id_name = "dlp_id"
    _dataclass = DlpEntry

    def list_by_target_dataset_type(
        self, target_dataset_type: Optional[str] = None
    ) -> List[dict]:
        """Return all existing DataLoadingPlans.

        Args:
            target_dataset_type: return only dlps matching the requested target type.

        Returns:
            An array of dict, each dict is a DataLoadingPlan
        """
        if target_dataset_type not in [t.value for t in DatasetTypes]:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: target_dataset_type should be of the "
                "values defined in 'fedbiomed.common.constants.DatasetTypes'"
            )

        return self.get_all_by_value("target_dataset_type", target_dataset_type)


class DlbTable(BaseTable):
    _table_name = "Dlbs"
    _id_name = "dlb_id"
    _dataclass = DlbEntry

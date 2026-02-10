# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional, Type, Union

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.db import TinyTableConnector
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._db_dataclasses import DatasetEntry, DlbEntry, DlpEntry


class BaseTable(TinyTableConnector):
    """Base class for database tables in this file (Dataset, DLP, DLB) with common
    validation functionality that uses dataclasses for schema enforcement.
    Insert/Update methods are wrapped. The rest do not change (TinyTableConnector).
    """

    # Dataclass is expected to have from_dict and to_dict methods
    _dataclass: Type = None

    def _get_operation_prefix(self) -> str:
        """Get operation prefix based on table name for security logging."""
        table_mapping = {
            "Datasets": "dataset",
            "Dlps": "dlp",
            "Dlbs": "dlb",
        }
        return table_mapping.get(self._table_name, "table")

    def _validate_and_convert_to_dict(self, data: dict) -> dict:
        """Generic validation using the table's dataclass"""
        try:
            # Single-pass validation and conversion
            entry = self._dataclass.from_dict(data)
            return entry.to_dict()
        except (KeyError, TypeError, ValueError) as e:
            model_name = self._dataclass.__name__
            raise FedbiomedError(f"{model_name} validation failed: {str(e)}") from e

    def _security_ids(
        self, payload: Optional[dict] = None, id_value: Optional[str] = None
    ) -> dict:
        """Build a consistent set of identifier fields for security logging.

        Notes:
        - Always includes the table primary ID field (self._id_name) when id_value is provided.
        - Also includes dataset_id and dlp_id when present in payload.
        """
        ids: dict = {}
        if payload:
            if "dataset_id" in payload:
                ids["dataset_id"] = payload["dataset_id"]
            if "dlp_id" in payload:
                ids["dlp_id"] = payload["dlp_id"]

        if id_value is not None and self._id_name:
            ids[self._id_name] = id_value

        return ids

    def insert(self, entry: dict) -> int:
        """Insert an entry with validation and security logging"""
        validated_entry = self._validate_and_convert_to_dict(entry)
        operation_prefix = self._get_operation_prefix()
        try:
            result = super().insert(validated_entry)
        except Exception as e:
            logger.security_event(
                operation=f"{operation_prefix}_create",
                status="failure",
                level="error",
                error=str(e),
                **self._security_ids(validated_entry),
                entry_name=validated_entry.get("name")
                or validated_entry.get("dlp_name"),
            )
            raise

        logger.security_event(
            operation=f"{operation_prefix}_create",
            status="success",
            level="info",
            **self._security_ids(validated_entry, id_value=result),
            entry_name=validated_entry.get("name") or validated_entry.get("dlp_name"),
        )
        return result

    def get_validated_entry(self, entry_id: str) -> Optional[Any]:
        """Get entry by ID and return as validated model instance"""
        data = self.get_by_id(entry_id)
        return None if not data else self._dataclass.from_dict(data)

    def all(self) -> List[dict]:
        """Get all entries with security logging."""
        result = super().all()

        operation_prefix = self._get_operation_prefix()
        logger.security_event(
            operation=f"{operation_prefix}_list",
            status="success",
            level="info",
            record_count=len(result),
        )
        return result

    def get_by_id(self, id_value: str) -> Optional[dict]:
        """Get entry by ID with security logging."""
        result = super().get_by_id(id_value)

        operation_prefix = self._get_operation_prefix()
        logger.security_event(
            operation=f"{operation_prefix}_read",
            status="success" if result else "not_found",
            level="info",
            **self._security_ids(result, id_value=id_value),
            entry_name=(result or {}).get("name") or (result or {}).get("dlp_name"),
        )
        return result

    def update_by_id(self, id_value, update):
        """Update an entry by its ID with validation and security logging

        Args:
             id_value: the ID of the entry to update
             update: dictionary of fields to update

        Raises:
            FedbiomedError: if validation fails or entry not found
        """
        operation_prefix = self._get_operation_prefix()

        try:
            # Prevent changing the ID field
            if self._id_name in update and update[self._id_name] != id_value:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Cannot change the field '{self._id_name}'"
                )

            # Fetch existing entry. If not found, raise error
            entry = super().get_by_id(id_value)
            if entry is None:
                raise FedbiomedError(f"No entry found with {self._id_name}={id_value}")

            # Merge and validate updated entry
            entry.update(update)
            updated_entry = self._validate_and_convert_to_dict(entry)
            result = super().update_by_id(id_value, updated_entry)
        except Exception as e:
            logger.security_event(
                operation=f"{operation_prefix}_update",
                status="failure",
                level="error",
                error=str(e),
                modified_fields=list(update.keys())
                if isinstance(update, dict)
                else None,
                **self._security_ids(
                    entry if "entry" in locals() else None, id_value=id_value
                ),
                entry_name=(entry or {}).get("name")
                if "entry" in locals() and entry
                else None,
            )
            raise

        logger.security_event(
            operation=f"{operation_prefix}_update",
            status="success",
            level="info",
            modified_fields=list(update.keys()),
            **self._security_ids(updated_entry, id_value=id_value),
            entry_name=updated_entry.get("name") or updated_entry.get("dlp_name"),
        )
        return result

    def delete_by_id(self, id_value: str):
        """Delete an entry by ID with security logging.

        Args:
            id_value: ID of the entry to delete

        Raises:
            FedbiomedError: if deletion fails
        """
        operation_prefix = self._get_operation_prefix()

        # Get entry info before deletion for logging
        entry = super().get_by_id(id_value)

        try:
            result = super().delete_by_id(id_value)
        except Exception as e:
            logger.security_event(
                operation=f"{operation_prefix}_delete",
                status="failure",
                level="error",
                error=str(e),
                **self._security_ids(entry, id_value=id_value),
                entry_name=(entry or {}).get("name") or (entry or {}).get("dlp_name"),
                data_type=(entry or {}).get("data_type"),
            )
            raise

        logger.security_event(
            operation=f"{operation_prefix}_delete",
            status="success",
            level="info",
            **self._security_ids(entry, id_value=id_value),
            entry_name=(entry or {}).get("name") or (entry or {}).get("dlp_name"),
            data_type=(entry or {}).get("data_type"),
        )
        return result


class DatasetTable(BaseTable):
    _table_name = "Datasets"
    _id_name = "dataset_id"
    _dataclass = DatasetEntry

    def insert(self, entry: dict) -> int:
        """Insert a dataset entry with validation and security logging"""
        # Dataset-specific validation: check for conflicting tags
        conflicting_datasets = self.search_conflicting_tags(entry["tags"])
        if conflicting_datasets:
            _msg = (
                f"{ErrorNumbers.FB322.value}: One or more registered datasets present "
                f"tags conflicting with your entry. Conflicting dataset names: "
                f"{', '.join([_['name'] for _ in conflicting_datasets])}."
            )
            # Merged logging: both critical console message and security log
            logger.critical(
                _msg,
                extra={
                    "is_security": True,
                    "operation": "dataset_create",
                    "status": "failure",
                    "error": "conflicting_tags",
                    "attempted_tags": entry.get("tags"),
                    "conflicting_datasets": [d["name"] for d in conflicting_datasets],
                },
            )
            raise FedbiomedError(_msg)

        # Call parent insert which now includes security logging
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

    def update_by_id(self, dataset_id: str, modified_dataset: dict):
        """Modifies a dataset in the database.

        Args:
            dataset_id: ID of the dataset to modify.
            modified_dataset: key-value pairs to replace in the existing entry.

        Raises:
            FedbiomedError: conflicting tags with existing dataset
        """
        # Dataset-specific validation: check for conflicting tags
        if "tags" in modified_dataset:
            # the dataset to modify is ignored (can conflict with its previous tags)
            conflicting_datasets = [
                _
                for _ in self.search_conflicting_tags(modified_dataset["tags"])
                if _["dataset_id"] != dataset_id
            ]
            if len(conflicting_datasets) > 0:
                msg = (
                    f"{ErrorNumbers.FB322.value}, : One or more registered datasets "
                    f"are conflicting with your new tags. Conflicting dataset names: "
                    f"{', '.join([_['name'] for _ in conflicting_datasets])}."
                )
                # Merged logging: both critical console message and security log
                logger.critical(
                    msg,
                    extra={
                        "is_security": True,
                        "operation": "dataset_update",
                        "status": "failure",
                        "dataset_id": dataset_id,
                        "error": "conflicting_tags",
                        "attempted_tags": modified_dataset.get("tags"),
                        "conflicting_datasets": [
                            d["name"] for d in conflicting_datasets
                        ],
                    },
                )
                raise FedbiomedError(msg)

        # Call parent update_by_id which now includes security logging
        return super().update_by_id(dataset_id, modified_dataset)


class DlpTable(BaseTable):
    _table_name = "Dlps"
    _id_name = "dlp_id"
    _dataclass = DlpEntry

    def insert(self, entry: dict) -> int:
        """Insert a DLP entry with validation and security logging

        Raises:
            FedbiomedError:
            - target_dataset_type value not valid
            - bad data loading plan name (size, not unique)
        """
        # DLP-specific validation
        if entry["target_dataset_type"] not in [t.value for t in DatasetTypes]:
            _msg = (
                f"{ErrorNumbers.FB632.value}: target_dataset_type should be of the "
                "values defined in 'fedbiomed.common.constants.DatasetTypes'"
            )
            logger.critical(
                _msg,
                extra={
                    "is_security": True,
                    "operation": "dlp_create",
                    "status": "failure",
                    "error": "invalid_target_dataset_type",
                    "attempted_type": entry.get("target_dataset_type"),
                },
            )
            raise FedbiomedError(_msg)

        if len(entry["dlp_name"]) < 4:
            _msg = (
                f"{ErrorNumbers.FB316.value}: Cannot save data loading plan, "
                "DLP name needs to have at least 4 characters."
            )
            logger.error(
                _msg,
                extra={
                    "is_security": True,
                    "operation": "dlp_create",
                    "status": "failure",
                    "error": "dlp_name_too_short",
                    "attempted_name": entry.get("dlp_name"),
                },
            )
            raise FedbiomedError(_msg)

        if self.get_all_by_value("dlp_name", entry["dlp_name"]):
            _msg = (
                f"{ErrorNumbers.FB316.value}: Cannot save data loading plan, "
                "DLP name needs to be unique."
            )
            logger.error(
                _msg,
                extra={
                    "is_security": True,
                    "operation": "dlp_create",
                    "status": "failure",
                    "error": "dlp_name_not_unique",
                    "attempted_name": entry.get("dlp_name"),
                },
            )
            raise FedbiomedError(_msg)

        # Call parent insert which now includes security logging
        return super().insert(entry)

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

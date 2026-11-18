# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional, Type

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.db import TinyTableConnector
from fedbiomed.common.exceptions import FedbiomedError

from ._db_dataclasses import (
    NodeConnectionStateEntry,
    NodeConnectionStateHistoryEntry,
    NodeProcessStateEntry,
    NodeProcessStateHistoryEntry,
)


def _delete_older_than(table, query, cutoff: str) -> List[int]:
    """Remove history entries updated before an ISO-8601 UTC instant.

    Entries are searched before being removed: TinyDB rewrites the whole database
    file on a removal, matching or not.

    Args:
        table: Table holding entries stamped with `updated_at`.
        query: Query object of that table.
        cutoff: Retention limit, formatted as the entries' `updated_at`.

    Returns:
        The document IDs of the removed entries.
    """
    condition = query.updated_at.exists() & (query.updated_at < cutoff)
    if not table.search(condition, stacklevel=5):
        return []

    return table.remove(condition, stacklevel=5)


class BaseTable(TinyTableConnector):
    """Base class for database tables in this file with common
    validation functionality that uses dataclasses for schema enforcement.
    Insert/Update methods are wrapped. The rest do not change (TinyTableConnector).
    """

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
        """Update an entry by its ID with validation

        Args:
             id_value: the ID of the entry to update
             update: dictionary of fields to update

        Raises:
            FedbiomedError: if validation fails or entry not found
        """
        # Prevent changing the ID field
        if self._id_name in update and update[self._id_name] != id_value:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Cannot change the field '{self._id_name}'"
            )

        # Fetch existing entry. If not found, raise error
        entry = self.get_by_id(id_value)
        if entry is None:
            raise FedbiomedError(f"No entry found with {self._id_name}={id_value}")

        # Merge and validate updated entry
        entry.update(update)
        updated_entry = self._validate_and_convert_to_dict(entry)
        return super().update_by_id(id_value, updated_entry)


class NodeProcessStateTable(BaseTable):
    """Database table for the current managed node process state."""

    _table_name = "NodeProcessState"
    _id_name = "node_id"
    _dataclass = NodeProcessStateEntry

    def update_or_insert_by_id(self, node_id: int, entry: dict):
        """Update the process state for a node, or insert it if absent."""
        if self._id_name in entry and entry[self._id_name] != node_id:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Cannot change the field '{self._id_name}'"
            )

        entry[self._id_name] = node_id
        if self.get_by_id(node_id) is None:
            return super().insert(entry)
        return super().update_by_id(node_id, entry)


class NodeProcessStateHistoryTable(BaseTable):
    """Append-only database table for node process state history."""

    _table_name = "NodeProcessStateHistory"
    _id_name = "node_id"
    _dataclass = NodeProcessStateHistoryEntry

    def insert(self, entry: dict) -> int:
        """Insert a history entry, allowing multiple entries for the same node_id."""
        validated_entry = super()._validate_and_convert_to_dict(entry)
        return self._table.insert(validated_entry, stacklevel=4)

    def delete_older_than(self, cutoff: str) -> List[int]:
        """Remove entries updated before the given ISO-8601 UTC instant.

        Args:
            cutoff: Retention limit, formatted as the entries' `updated_at`.

        Returns:
            The document IDs of the removed entries.
        """
        return _delete_older_than(self._table, self._query, cutoff)


class NodeConnectionStateTable(BaseTable):
    """Database table for the node's current connection state."""

    _table_name = "NodeConnectionState"
    _id_name = "node_id"
    _dataclass = NodeConnectionStateEntry

    def replace_by_id(self, node_id: str, entry: dict):
        """Replace the connection state of a node, or insert it if absent.

        Replaces rather than merges, so that fields the previous state carried and
        the new one does not - a failure reason, a peer certificate - do not survive it.
        """
        if self._id_name in entry and entry[self._id_name] != node_id:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Cannot change the field '{self._id_name}'"
            )

        entry[self._id_name] = node_id
        self.delete_by_id(node_id)
        return super().insert(entry)


class NodeConnectionStateHistoryTable(BaseTable):
    """Append-only database table for node connection state history."""

    _table_name = "NodeConnectionStateHistory"
    _id_name = "node_id"
    _dataclass = NodeConnectionStateHistoryEntry

    def insert(self, entry: dict) -> int:
        """Insert a history entry, allowing multiple entries for the same node_id."""
        validated_entry = super()._validate_and_convert_to_dict(entry)
        return self._table.insert(validated_entry, stacklevel=4)

    def delete_older_than(self, cutoff: str) -> List[int]:
        """Remove entries updated before the given ISO-8601 UTC instant.

        Args:
            cutoff: Retention limit, formatted as the entries' `updated_at`.

        Returns:
            The document IDs of the removed entries.
        """
        return _delete_older_than(self._table, self._query, cutoff)

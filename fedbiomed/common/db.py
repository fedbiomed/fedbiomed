# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Interfaces with a tinyDB database for converting search results to dict.
"""

from typing import Any, Callable, Dict, List, Union

from tinydb import Query, TinyDB
from tinydb.table import Document, Table


def cast_(func):
    """Decorator function for typing casting"""

    # Wraps TinyDb get, all, search and upsert methods
    def wrapped(*args, **kwargs):
        add_docs = kwargs.get("add_docs")
        if add_docs is not None:
            kwargs.pop("add_docs")

        document = func(*args, **kwargs)
        if isinstance(document, list):
            if document and isinstance(document[0], Document):
                casted = [dict(r) for r in document]
            else:
                casted = document
        elif isinstance(document, Document):
            casted = dict(document)
        else:
            # Plain python type
            casted = document

        if add_docs:
            return casted, document
        else:
            return casted

    return wrapped


class DBTable(Table):
    """Extends TinyDB table to cast Document type to dict"""

    @cast_
    def insert(self, *args, **kwargs):
        return super().insert(*args, **kwargs)

    @cast_
    def search(self, *args, **kwargs):
        return super().search(*args, **kwargs)

    @cast_
    def get(self, *args, **kwargs):
        return super().get(*args, **kwargs)

    @cast_
    def all(self, *args, **kwargs):
        return super().all(*args, **kwargs)

    @cast_
    def update(self, *args, **kwargs):
        return super().update(*args, **kwargs)

    @cast_
    def delete(self, *args, **kwargs):
        return super().remove(*args, **kwargs)


class TinyDBConnector:
    """Singleton TinyDB connector"""

    _instance = None
    _db = None

    def __new__(cls, db_path: str):
        # Ensure only one instance of TinyDBConnector
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Use DBTable as the default table class for this TinyDB instance
            cls._db = TinyDB(db_path, table_class=DBTable)
        return cls._instance

    @property
    def db(self):
        """Return the shared TinyDB instance"""
        return self._db

    def table(self, name: str) -> DBTable:
        """Return a table with the given name, ensuring it is a DBTable instance."""
        # Get the table from the underlying DB instance
        table_instance = self._db.table(name)

        # If it's not already a DBTable, wrap it. This handles cases where
        # the table was cached by TinyDB before the table_class was set.
        if not isinstance(table_instance, DBTable):
            table_instance.__class__ = DBTable

        return table_instance


class TinyTableConnector:
    """Base class for tables in TinyDB with common operations"""

    _table_name = None
    _id_name = None

    def __init__(self, path: str):
        if type(self) is TinyTableConnector:
            raise TypeError("`TinyTableConnector` cannot be instantiated directly")

        self._query = Query()
        self._table = TinyDBConnector(db_path=path).table(self._table_name)

    def get_by_id(self, id_value: str) -> Union[dict, None]:
        """Get a document by its ID.

        Args:
            id_value: The ID value to search for.

        Returns:
            The document if found, otherwise None.
        """
        response = self._table.search(self._query[self._id_name] == id_value)
        assert len(response) < 2, (
            f"Multiple entries found for {self._id_name}={id_value}, "
            "which should be unique."
        )
        return response[0] if response else None

    def create(self, entry: dict) -> int:
        """Insert a new document.

        Args:
            entry: The document to insert.

        Returns:
            The document ID of the inserted entry.
        """
        if not isinstance(entry, dict):
            raise TypeError(f"Expected entry to be dict, got {type(entry)}.")
        if self._id_name not in entry:
            raise ValueError(f"Entry must contain '{self._id_name}' field.")
        if self.get_by_id(entry[self._id_name]) is not None:
            raise KeyError(
                f"Entry with {self._id_name}={entry[self._id_name]} already exists."
            )
        return self._table.insert(entry)

    def all(self) -> List[dict]:
        """Get all entries (or empty list if none found).

        Returns:
            The list of entries as dicts, or empty list if none found.
        """
        return self._table.all()

    def get_all_by_value(self, by: str, value: Any) -> List[dict]:
        """Get all entries by a field value (or empty list if none found).

        Args:
            by: field name to search by.
            value: field value to search for.

        Returns:
            The list of entries as dicts, or empty list if none found.
        """
        return self._table.search(
            self._query[by].one_of(value)
            if isinstance(value, (list, tuple))
            else self._query[by] == value
        )

    def get_all_by_condition(self, by: str, cond: Callable) -> List[dict]:
        """Search entries by a test condition on a field.

        Args:
            by: field name to search by.
            cond: A function that takes the field value and returns a boolean.

        Returns:
            The list of entries as dicts, or empty list if none found.
        """
        return self._table.search(self._query[by].test(cond))

    def delete_by_id(self, id_value: Union[str, list[str]]) -> List[int]:
        """Delete a document by its ID.

        Args:
            id_value: The ID value or list of ID values to delete.

        Returns:
            The list of removed doc IDs.
        """
        id_value = [id_value] if isinstance(id_value, str) else id_value

        for value in id_value:
            if not isinstance(value, str):
                raise TypeError(f"Expected id to be string, got {type(value)} instead.")
            if not id_value:
                raise ValueError("Expected id not to be an empty str.")

        return self._table.delete(self._query[self._id_name].one_of(id_value))

    def update_by_id(self, id_value: str, update: Dict[str, Any]) -> dict:
        """Update a document by its ID.

        Args:
            id_value: The ID value to update.
            update: A dictionary of fields to update.

        Returns:
            Updated document.
        """
        if self.get_by_id(id_value) is None:
            raise KeyError(f"No entry found with {self._id_name}={id_value}")

        _ = self._table.update(update, self._query[self._id_name] == id_value)

        return self.get_by_id(id_value)

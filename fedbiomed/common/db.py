# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Interfaces with a tinyDB database for converting search results to dict.
"""

from typing import Any, Dict, List, Optional

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


class DB:
    def __init__(self, path, table_name: str = "database"):
        self._db = TinyDB(path)
        self._database = DBTable(storage=self._db.storage, name=table_name)
        self._query = Query()

    def _get_by(self, by, value) -> Optional[Dict[str, Any]]:
        """Get a single entry by a field value (or None if missing).
        Args:
            by: field name to search by.
            value: field value to search for.
        Returns:
            The entry as a dict, or None if not found.
        """
        return self._database.get(self._query[by] == value)

    def _get_all_by(self, by, value) -> List[Document]:
        """Get all entries by a field value (or empty list if none found).
        Args:
            by: field name to search by.
            value: field value to search for.
        Returns:
            The list of entries as dicts, or empty list if none found.
        """
        return self._database.search(
            self._query[by].test(lambda x: set(value).issubset(set(x)))
        )

    def _list(self) -> List[Document]:
        """List all entries in the table."""
        return self._database.all()

    def _delete_by(self, by, value) -> List[int]:
        """Delete by a field value. Returns the list of removed doc IDs.
        Args:
            by: field name to delete by.
            value: field value to delete by.
        Returns:
            The list of removed document IDs.
        """
        return self._database.delete(self._query[by] == value)

    def _update_by(self, by, value: Dict[str, Any]) -> List[int]:
        """Partial update by a field value. Returns list of updated doc IDs.
        Args:
            by: field name to update by.
            value: field value to update by.
        Returns:
            The list of updated document IDs.
        """
        return self._database.update(value, self._query[by] == value.get(by))


class DBTable(Table):
    """Extends TinyDB table to cast Document type to dict"""

    @cast_
    def create(self, *args, **kwargs):
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

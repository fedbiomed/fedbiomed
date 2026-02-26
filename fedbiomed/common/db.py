# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Interfaces with a tinyDB database for converting search results to dict.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from tinydb import Query, TinyDB
from tinydb.table import Document, Table

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger


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


def _is_forbidden(key: str) -> bool:
    """Set of keys/strings that are not allowed in the database entries"""
    forbidden_strings = {"certificate", "key", "secagg_elem"}
    for forbidden in forbidden_strings:
        if forbidden in key.lower():
            return True
    return False


def _strip_forbidden_keys(value: Any) -> Any:
    """Recursively removes forbidden keys from dict-like payloads.

    This is used to avoid storing or logging sensitive material such as
    private keys or certificates.
    """
    if isinstance(value, dict):
        return {
            k: _strip_forbidden_keys(v)
            for k, v in value.items()
            if not _is_forbidden(str(k))
        }
    if isinstance(value, (list, tuple)):
        stripped = [_strip_forbidden_keys(v) for v in value]
        return tuple(stripped) if isinstance(value, tuple) else stripped
    return value


def _truncate_for_logging(value: Any, max_len: int) -> Any:
    """Recursively truncates long stringified values for logging."""
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            out[k] = _truncate_for_logging(v, max_len=max_len)
        return out

    if isinstance(value, (list, tuple)):
        truncated = [_truncate_for_logging(v, max_len=max_len) for v in value]
        return tuple(truncated) if isinstance(value, tuple) else truncated

    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return value


class DBTable(Table):
    """Extends TinyDB table to cast Document type to dict"""

    def _run_with_security_logging(
        self,
        *,
        operation: str,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Runs a TinyDB operation with consistent security logging.

        Args:
            operation: Logical operation name (eg. 'insert', 'search').
            fn: Function to execute.
            args: Positional args forwarded to fn.
            kwargs: Keyword args forwarded to fn.
        """
        logging_stacklevel = kwargs.pop(
            "stacklevel", 4
        )  # Adjust stack level to point to caller of DBTable method

        LOG_VALUE_MAX_LENGTH = 50

        kwargs_stripped = _strip_forbidden_keys(kwargs)
        args_stripped = _strip_forbidden_keys(args)

        # Build log fields from ARGS:
        # - If it's a single dict positional arg (TinyDB insert/update common case),
        #   log its items as top-level fields (after stripping) so you get key:value entries.
        # - Otherwise, log each positional argument separately as arg0, arg1, ...
        arg_fields: dict[str, Any] = {}
        if len(args_stripped) == 1 and isinstance(args_stripped[0], dict):
            # merge payload fields (already stripped)
            arg_fields.update(args_stripped[0])
        else:
            for i, v in enumerate(args_stripped):
                arg_fields[f"arg{i}"] = v

        kwargs_for_log = _truncate_for_logging(
            kwargs_stripped, max_len=LOG_VALUE_MAX_LENGTH
        )
        args_for_log = _truncate_for_logging(arg_fields, max_len=LOG_VALUE_MAX_LENGTH)

        with logger.security_context(operation=operation, table=self.name):
            try:
                doc_id = fn(*args, **kwargs)
                doc_id_for_log = doc_id
                if operation in {"get", "all", "search"}:
                    # For read operations, log the number of documents returned
                    if isinstance(doc_id, list):
                        doc_id_for_log = f"{len(doc_id)} documents"
                    elif doc_id is not None:
                        doc_id_for_log = "1 document"
                    else:
                        doc_id_for_log = "0 documents"
            except Exception as e:
                logger.security_event(
                    status="failure",
                    details=str(e),
                    **args_for_log,
                    **kwargs_for_log,
                    stacklevel=logging_stacklevel,
                )
                raise FedbiomedError(
                    f"Failed to {operation} in table {self.name} with error: {e}"
                ) from e
            logger.security_event(
                status="success",
                doc_id=doc_id_for_log,
                **args_for_log,
                **kwargs_for_log,
                stacklevel=logging_stacklevel,
            )

        return doc_id

    @cast_
    def insert(self, *args, **kwargs):
        # insert may receive the document as positional arg; sanitize payload to
        # avoid forbidden keys reaching the DB storage.
        return self._run_with_security_logging(
            operation="insert",
            fn=super().insert,
            args=args,
            kwargs=kwargs,
        )

    @cast_
    def search(self, *args, **kwargs):
        return self._run_with_security_logging(
            operation="search",
            fn=super().search,
            args=args,
            kwargs=kwargs,
        )

    @cast_
    def get(self, *args, **kwargs):
        return self._run_with_security_logging(
            operation="get",
            fn=super().get,
            args=args,
            kwargs=kwargs,
        )

    @cast_
    def all(self, *args, **kwargs):
        return self._run_with_security_logging(
            operation="all",
            fn=super().all,
            args=args,
            kwargs=kwargs,
        )

    @cast_
    def update(self, *args, **kwargs):
        # update takes a dict of fields as positional arg in the common case
        return self._run_with_security_logging(
            operation="update",
            fn=super().update,
            args=args,
            kwargs=kwargs,
        )

    @cast_
    def remove(self, *args, **kwargs):
        return self._run_with_security_logging(
            operation="delete",
            fn=super().remove,
            args=args,
            kwargs=kwargs,
        )


class TinyDBConnector:
    """Singleton TinyDB connector"""

    _instance = None

    def __new__(cls, db_path: str):
        # Ensure only one instance of TinyDBConnector
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Use DBTable as the default table class for this TinyDB instance
            cls._instance._db = TinyDB(db_path)
        return cls._instance

    @property
    def db(self) -> TinyDB:
        """Return the shared TinyDB instance"""
        return self._db

    def table(self, name: str) -> DBTable:
        """Return a table with the given name, ensuring it is a DBTable instance."""
        # Get the table from the underlying DB instance, forcing cache_size=0
        table = self._db.table(name, cache_size=0)
        return DBTable(table.storage, table.name)


class TinyTableConnector:
    """Base class for tables in TinyDB with common operations"""

    _table_name = None
    _id_name = None

    def __init__(self, path: str):
        if type(self) is TinyTableConnector:
            raise TypeError("`TinyTableConnector` cannot be instantiated directly")

        self._query = Query()
        self._table = TinyDBConnector(db_path=path).table(self._table_name)

    def get_by_id(self, id_value: str) -> Optional[dict]:
        """Get a document by its ID.

        Args:
            id_value: The ID value to search for.

        Returns:
            The document if found, otherwise None.
        """
        response = self._table.search(
            self._query[self._id_name] == id_value, stacklevel=5
        )
        assert len(response) < 2, (
            f"Multiple entries found for {self._id_name}={id_value}, "
            "which should be unique."
        )
        return response[0] if response else None

    def insert(self, entry: dict) -> int:
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
        _ = self._table.insert(entry, stacklevel=5)
        return entry[self._id_name]

    def all(self) -> List[dict]:
        """Get all entries (or empty list if none found).

        Returns:
            The list of entries as dicts, or empty list if none found.
        """
        return self._table.all(stacklevel=5)

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
            else self._query[by] == value,
            stacklevel=5,
        )

    def get_all_by_condition(self, by: str, cond: Callable) -> List[dict]:
        """Search entries by a test condition on a field.

        Args:
            by: field name to search by.
            cond: A function that takes the field value and returns a boolean.

        Returns:
            The list of entries as dicts, or empty list if none found.
        """
        return self._table.search(self._query[by].test(cond), stacklevel=5)

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

        return self._table.remove(
            self._query[self._id_name].one_of(id_value), stacklevel=5
        )

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

        _ = self._table.update(
            update, self._query[self._id_name] == id_value, stacklevel=5
        )

        return self.get_by_id(id_value)

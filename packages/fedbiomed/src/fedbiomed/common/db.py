# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Access to the components' databases: the SQLite `Database` with `DatabaseEntry`,
parent of its tables' ORM models (`*Entry` classes), and the TinyDB tables not
moved to it yet.
"""

from __future__ import annotations

import inspect as pyinspect
import re
import sys
from contextlib import contextmanager
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Self, Union

from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.orm import (
    DeclarativeBase,
    MappedAsDataclass,
    ORMExecuteState,
    Session,
)
from tinydb import Query, TinyDB
from tinydb.table import Document, Table

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

LOG_VALUE_MAX_LENGTH = 250
REDACTED = "<redacted>"

# Entry fields whose values must never reach the logs. Case-insensitive.
SENSITIVE_FIELDS = {"certificate", "key", "secagg_elem", "context"}


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


def _is_sensitive(key: str) -> bool:
    """Whether an entry field holds material that must not reach the logs."""
    return any(field in key.lower() for field in SENSITIVE_FIELDS)


def _redact(obj: Any) -> Any:
    """Replaces the values of sensitive fields, at any depth, before logging.

    Entries are redacted by field name rather than relying on log truncation.
    """
    if isinstance(obj, dict):
        return {
            k: REDACTED if _is_sensitive(str(k)) else _redact(v) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_redact(v) for v in obj)
    return obj


def to_print_str(obj, max_len=None) -> str:
    """
    Convert any Python object to its 'printable' string representation.
    Always returns a string, never crashes.
    Optionally trims to max_len characters.
    """
    try:
        # Use str() which mimics print()
        s = str(obj)
    except Exception:
        # fallback: repr() if str() fails
        try:
            s = repr(obj)
        except Exception:
            s = "<unprintable object>"

    # Remove leading/trailing whitespace
    s = s.strip()

    # Trim if needed
    if max_len:
        s = f"{s[:max_len]}..."

    return s


def printtable_args_for_log(obj: Any, max_len=None) -> str:
    """Convert any Python object to a string suitable for logging, with error handling."""

    try:
        obj = _redact(obj)
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                parts.append(f"{k}={to_print_str(v, max_len=LOG_VALUE_MAX_LENGTH)}")
            return "{" + ", ".join(parts) + "}"
        else:
            return to_print_str(obj, max_len=LOG_VALUE_MAX_LENGTH)
    except Exception:
        return "<unprintable object>"


def _security_log(operation: str, default_stacklevel: int = 3):
    """Decorator for DBTable methods to add security logging."""

    def _security_log(func):
        """The actual decorator function that will be applied to the method.
        It takes the method as an argument and returns a wrapped version of it with security logging."""

        def wrapper(self, *args, **kwargs):
            """Wrapper function of the actual decorator
            This wrapper will be called with the same args/kwargs as the original method, and is responsible for:
            Extracting/logging relevant information from args/kwargs for security events."""

            logging_stacklevel = kwargs.pop("stacklevel", default_stacklevel)

            kwargs_for_log = printtable_args_for_log(
                kwargs, max_len=LOG_VALUE_MAX_LENGTH
            )
            args_for_log = printtable_args_for_log(args, max_len=LOG_VALUE_MAX_LENGTH)

            with logger.security_context(operation=operation, table=self.name):
                try:
                    result = func(self, *args, **kwargs)

                    result_for_log = result
                    if operation in {"get", "all", "search"}:
                        if isinstance(result, list):
                            result_for_log = f"{len(result)} documents"
                        elif result is not None:
                            result_for_log = "1 document"
                        else:
                            result_for_log = "0 documents"

                except Exception as e:
                    logger.security_event(
                        status="failure",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        db_args=args_for_log,
                        db_kwargs=kwargs_for_log,
                        stacklevel=logging_stacklevel,
                    )
                    logger.debug(
                        f"Failed to {operation} in table {self.name} with error: {repr(e)}; "
                        f"db_args={args_for_log}; db_kwargs={kwargs_for_log}",
                        extra={
                            "db_args": args_for_log,
                            "db_kwargs": kwargs_for_log,
                        },
                        stacklevel=logging_stacklevel,
                    )
                    raise FedbiomedError(
                        f"Failed to {operation} in table {self.name} with error: {e}"
                    ) from e

                logger.security_event(
                    status="success",
                    doc_id=result_for_log,
                    db_args=args_for_log,
                    db_kwargs=kwargs_for_log,
                    stacklevel=logging_stacklevel,
                )
                logger.debug(
                    f"Successfully performed {operation} in table {self.name}, result: {result_for_log}; "
                    f"db_args={args_for_log}; db_kwargs={kwargs_for_log}",
                    extra={
                        "db_args": args_for_log,
                        "db_kwargs": kwargs_for_log,
                    },
                    stacklevel=logging_stacklevel,
                )
            return result

        return wrapper

    return _security_log


class DBTable(Table):
    """Extends TinyDB table to cast Document type to dict"""

    @cast_
    @_security_log("insert", default_stacklevel=3)
    def insert(self, *args, **kwargs):
        return super().insert(*args, **kwargs)

    @cast_
    @_security_log("search", default_stacklevel=3)
    def search(self, *args, **kwargs):
        return super().search(*args, **kwargs)

    @cast_
    @_security_log("get", default_stacklevel=3)
    def get(self, *args, **kwargs):
        return super().get(*args, **kwargs)

    @cast_
    @_security_log("all", default_stacklevel=3)
    def all(self, *args, **kwargs):
        return super().all(*args, **kwargs)

    @cast_
    @_security_log("update", default_stacklevel=3)
    def update(self, *args, **kwargs):
        # update takes a dict of fields as positional arg in the common case
        return super().update(*args, **kwargs)

    @cast_
    @_security_log("remove", default_stacklevel=3)
    def remove(self, *args, **kwargs):
        return super().remove(*args, **kwargs)


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
            self._query[self._id_name] == id_value, stacklevel=4
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
        _ = self._table.insert(entry, stacklevel=4)
        return entry[self._id_name]

    def all(self) -> List[dict]:
        """Get all entries (or empty list if none found).

        Returns:
            The list of entries as dicts, or empty list if none found.
        """
        return self._table.all(stacklevel=4)

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
            stacklevel=4,
        )

    def get_all_by_condition(self, by: str, cond: Callable) -> List[dict]:
        """Search entries by a test condition on a field.

        Args:
            by: field name to search by.
            cond: A function that takes the field value and returns a boolean.

        Returns:
            The list of entries as dicts, or empty list if none found.
        """
        return self._table.search(self._query[by].test(cond), stacklevel=4)

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
            self._query[self._id_name].one_of(id_value), stacklevel=4
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
            update, self._query[self._id_name] == id_value, stacklevel=4
        )
        return self.get_by_id(id_value)


class Database:
    """This process's connections to its component's SQLite file (one per process)."""

    def __init__(self, path: str):
        """Opens the SQLite file at `path`, creating it if needed."""
        # timeout: seconds to wait for another process holding the write lock
        self._engine = create_engine(f"sqlite:///{path}", connect_args={"timeout": 30})
        event.listen(self._engine, "connect", self._on_connect)
        event.listen(self._engine, "begin", self._on_begin)

    def create_tables(self, models: List[type[DeclarativeBase]]) -> None:
        """Creates the tables of `models` that do not exist yet."""
        for model in models:
            # Takes the write lock: processes starting together create it one at a time
            model.__table__.create(self._engine, checkfirst=True)

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        """Session in one transaction: committed at the end, rolled back on error."""
        with Session(self._engine) as session, session.begin():
            yield session

    @staticmethod
    def _on_connect(dbapi_connection, _connection_record) -> None:
        """Turns on foreign keys and leaves starting transactions to `_on_begin`."""
        dbapi_connection.isolation_level = None
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    @staticmethod
    def _on_begin(connection) -> None:
        """Takes the write lock at begin, so checks hold against the other processes."""
        # Waits for another process holding the lock up to the engine's timeout
        connection.exec_driver_sql("BEGIN IMMEDIATE")


# One Database per process: a process only passes its component's path
@cache
def get_database(db_path: str) -> Database:
    """The process's database of a component, at `db_path` with the `.sqlite` suffix."""
    # Next to the component's TinyDB file, which other tables still use
    return Database(str(Path(db_path).with_suffix(".sqlite")))


class DatabaseEntry(MappedAsDataclass, DeclarativeBase, kw_only=True):
    """Parent of the ORM models (`*Entry` classes).

    Provides row/dict conversion, security log and FB634 errors.
    """

    # A JSON value changed in place (`tags.append`) is not saved: assign a new value

    class SecurityLog(Enum):
        """What a model's table writes to the security log."""

        ALL = "all"  # reads and writes
        WRITES = "writes"
        NONE = "none"

    __security_log__ = SecurityLog.ALL
    # Columns whose values never reach the security log
    __sensitive_fields__: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Builds a row from an entry; `None` counts as not given."""
        try:
            return cls(**{k: v for k, v in data.items() if v is not None})
        except TypeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: Invalid data input for "
                f"{cls.__tablename__}: {e}"
            ) from e

    def to_dict(self) -> dict:
        """Returns the row's columns as a dict, without the ones that are not set."""
        columns = (attr.key for attr in inspect(self).mapper.column_attrs)
        return {k: v for k in columns if (v := getattr(self, k)) is not None}

    @staticmethod
    def _record_writes(session: Session, _flush_context, _instances) -> None:
        """Records the entries the session is about to insert, update or remove."""
        for operation, entries in (
            ("insert", session.new),
            ("update", session.dirty),
            ("remove", session.deleted),
        ):
            for entry in entries:
                if not isinstance(entry, DatabaseEntry) or (
                    entry.__security_log__ == DatabaseEntry.SecurityLog.NONE
                ):
                    continue
                state = inspect(entry)
                doc_id = state.mapper.primary_key_from_instance(entry)[0]
                if operation == "insert":
                    args = entry.to_dict()
                elif operation == "update":
                    if not session.is_modified(entry):
                        continue
                    args = {
                        attr.key: attr.value
                        for attr in state.attrs
                        if attr.history.has_changes()
                    }
                else:
                    args = {}
                DatabaseEntry._record(session, operation, type(entry), doc_id, args)

    @staticmethod
    def _record_statements(state: ORMExecuteState) -> None:
        """Records ORM statements: reads, and updates or deletes by statement."""
        # Execution parameters carry the key of `session.get`
        args = {**state.statement.compile().params, **(state.parameters or {})}
        for mapper in state.all_mappers:
            if not issubclass(mapper.class_, DatabaseEntry):
                continue
            logged = mapper.class_.__security_log__
            if state.is_select and logged == DatabaseEntry.SecurityLog.ALL:
                operation = "select"
            elif state.is_update and logged != DatabaseEntry.SecurityLog.NONE:
                operation = "update"
            elif state.is_delete and logged != DatabaseEntry.SecurityLog.NONE:
                operation = "remove"
            else:
                continue
            DatabaseEntry._record(state.session, operation, mapper.class_, None, args)

    @staticmethod
    def _record(session: Session, operation: str, model: type, doc_id, args) -> None:
        """Keeps a security event until the transaction ends."""
        session.info.setdefault("security_events", []).append(
            {
                "operation": operation,
                "table": model.__tablename__,
                "doc_id": doc_id,
                "db_args": DatabaseEntry._log_args(model, args),
            }
        )

    @staticmethod
    def _log_args(model: type, args: dict) -> str:
        """Arguments as written to the log: the model's sensitive values redacted."""
        parts = []
        for key, value in args.items():
            # Statement parameters are named after their column: `key_1` for `key`
            if re.sub(r"_\d+$", "", key) in model.__sensitive_fields__:
                text = "<redacted>"
            else:
                # Keeps the log lines short: 250 characters of each value
                text = f"{str(value).strip()[:250]}..."
            parts.append(f"{key}={text}")
        return "{" + ", ".join(parts) + "}"

    @staticmethod
    def _write_success(session: Session) -> None:
        """Writes the events of a committed transaction."""
        DatabaseEntry._write_events(session, {"status": "success"})

    @staticmethod
    def _write_failure(session: Session) -> None:
        """Writes the events of a rolled back transaction, with its error if any."""
        outcome = {"status": "failure"}
        error = sys.exc_info()[1]  # the error being handled, which caused the rollback
        if error is not None:
            outcome["error_type"] = type(error).__name__
            outcome["error_message"] = str(error)
        DatabaseEntry._write_events(session, outcome)

    @staticmethod
    def _write_events(session: Session, outcome: dict) -> None:
        """Writes the transaction's recorded events with its outcome."""
        events = session.info.pop("security_events", [])
        if not events:
            return
        stacklevel = DatabaseEntry._caller_stacklevel()
        for recorded in events:
            logger.security_event(stacklevel=stacklevel, **recorded, **outcome)

    @staticmethod
    def _caller_stacklevel() -> int:
        """Stack level of the manager code that ran the transaction, past SQLAlchemy."""
        frame = pyinspect.currentframe().f_back
        # 1: the method calling `security_event`
        level = 1
        while frame is not None and frame.f_globals["__name__"].startswith(
            ("sqlalchemy", "contextlib", "fedbiomed.common.db")
        ):
            frame = frame.f_back
            level += 1
        return level

    @staticmethod
    def _raise_database_error(context) -> None:
        """Raises FB634 naming the failing table; logs the statement without values."""
        reason = context.original_exception
        logger.debug(f"Database error: {reason}; statement: {context.statement}")
        compiled = getattr(context.execution_context, "compiled", None)
        table = getattr(getattr(compiled, "statement", None), "table", None)
        where = f"{table.name}: " if table is not None else ""
        raise FedbiomedError(f"{ErrorNumbers.FB634.value}: {where}{reason}") from reason


# Every session and engine of the process: only `DatabaseEntry` rows are logged
event.listen(Session, "before_flush", DatabaseEntry._record_writes)
event.listen(Session, "do_orm_execute", DatabaseEntry._record_statements)
event.listen(Session, "after_commit", DatabaseEntry._write_success)
event.listen(Session, "after_rollback", DatabaseEntry._write_failure)
event.listen(Engine, "handle_error", DatabaseEntry._raise_database_error)

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Interfaces with a tinyDB database for converting search results to dict.
"""

from __future__ import annotations

import inspect as pyinspect
import sys
import types
from functools import cache
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Self,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from sqlalchemy import JSON, Engine, create_engine, event, inspect
from sqlalchemy.orm import (
    DeclarativeBase,
    MappedAsDataclass,
    ORMExecuteState,
    Session,
    sessionmaker,
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

# What a model writes to the security log: reads and writes, writes, nothing
LOG_LEVELS = ("all", "writes", "none")


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


class Base(MappedAsDataclass, DeclarativeBase, kw_only=True):
    """Parent of the ORM models.

    Models are dataclasses: the constructor takes keyword arguments only and
    refuses unknown or missing fields.

    An entry is checked as a whole by `validate`, right before the session
    writes it, for an insert (`session.add`) as for an update (fields
    assigned on a loaded entry). Changes are written when the
    `with factory.begin()` block ends; a JSON value changed in place
    (`entry.tags.append(...)`) is not seen as a change: assign a new value.

    `__log_level__` sets what reaches the security log for the model's table:
    `"all"` (reads and writes), `"writes"` or `"none"`. Events are written
    when the transaction ends: `success` after the commit, `failure` after a
    rollback.
    """

    __log_level__ = "all"

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Builds a row from an entry, refusing unknown or missing fields."""
        try:
            return cls(**data)
        except TypeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Invalid data input for {cls.__name__}: {e}"
            ) from e

    def to_dict(self) -> dict:
        """Returns the row's columns as a dict, without the ones that are not set."""
        columns = (attr.key for attr in inspect(self).mapper.column_attrs)
        return {k: v for k in columns if (v := getattr(self, k)) is not None}

    def validate(self) -> None:
        """Checks the entry before it is written.

        JSON values must match their column annotation
        (`Mapped[dict[str, list[str]]]`). Models with more rules override this
        method and call it first.

        Raises:
            FedbiomedError: a value does not match its column.
        """
        hints = get_type_hints(type(self))
        for column_attr in inspect(self).mapper.column_attrs:
            if not isinstance(column_attr.columns[0].type, JSON):
                continue
            key = column_attr.key
            (expected,) = get_args(hints[key])
            value = getattr(self, key)
            if not _matches(value, expected):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Invalid data input for "
                    f"{type(self).__name__}.{key}: expected {expected}, got "
                    f"{printtable_args_for_log({key: value})}"
                )


def _matches(value: Any, expected: Any) -> bool:
    """Whether a JSON value matches a type annotation (list, dict, union, Any)."""
    origin = get_origin(expected)
    if expected is Any:
        return True
    if origin in (Union, types.UnionType):
        return any(_matches(value, option) for option in get_args(expected))
    if origin is list:
        (item_type,) = get_args(expected)
        return isinstance(value, list) and all(_matches(v, item_type) for v in value)
    if origin is dict:
        key_type, value_type = get_args(expected)
        return isinstance(value, dict) and all(
            _matches(k, key_type) and _matches(v, value_type) for k, v in value.items()
        )
    if expected is type(None):
        return value is None
    return isinstance(value, expected)


def _validate_entries(session: Session, _flush_context, _instances) -> None:
    """Validates the entries the session is about to insert or update."""
    for entry in (*session.new, *session.dirty):
        entry.validate()


def _record(session: Session, operation: str, model: type, doc_id, args) -> None:
    """Keeps a security event until the transaction ends."""
    session.info.setdefault("security_events", []).append(
        {
            "operation": operation,
            "table": model.__tablename__,
            "doc_id": doc_id,
            "db_args": printtable_args_for_log(args),
        }
    )


def _record_writes(session: Session, _flush_context, _instances) -> None:
    """Records the entries the session is about to insert, update or remove."""
    for operation, entries in (
        ("insert", session.new),
        ("update", session.dirty),
        ("remove", session.deleted),
    ):
        for entry in entries:
            if entry.__log_level__ == "none":
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
            _record(session, operation, type(entry), doc_id, args)


def _record_statements(state: ORMExecuteState) -> None:
    """Records ORM statements: reads, and updates or deletes by statement."""
    # Execution parameters carry the key of `session.get`
    args = {**state.statement.compile().params, **(state.parameters or {})}
    for mapper in state.all_mappers:
        level = mapper.class_.__log_level__
        if state.is_select and level == "all":
            _record(state.session, "select", mapper.class_, None, args)
        elif state.is_update and level != "none":
            _record(state.session, "update", mapper.class_, None, args)
        elif state.is_delete and level != "none":
            _record(state.session, "remove", mapper.class_, None, args)


def _caller_stacklevel() -> int:
    """Stack level of the code that ran the transaction, for `security_event`.

    Skips the frames of SQLAlchemy, `contextlib` and this module, so the
    event names the caller's `with` block rather than the commit internals.
    """
    frame = pyinspect.currentframe().f_back
    # 1: the function calling `security_event`
    level = 1
    while frame is not None and frame.f_globals["__name__"].startswith(
        ("sqlalchemy", "contextlib", __name__)
    ):
        frame = frame.f_back
        level += 1
    return level


def _write_events(session: Session, status: str, error: BaseException | None) -> None:
    """Writes the transaction's security events with its outcome."""
    events = session.info.pop("security_events", [])
    if not events:
        return
    failure = {}
    if error is not None:
        failure = {"error_type": type(error).__name__, "error_message": str(error)}
    stacklevel = _caller_stacklevel()
    for recorded in events:
        logger.security_event(
            status=status, stacklevel=stacklevel, **recorded, **failure
        )


def _on_commit(session: Session) -> None:
    """Writes the transaction's security events as successful."""
    _write_events(session, "success", None)


def _on_rollback(session: Session) -> None:
    """Writes the transaction's security events as failed.

    A rollback caused by an error runs while that error is being handled, so
    `sys.exc_info` gives it.
    """
    _write_events(session, "failure", sys.exc_info()[1])


def _on_connect(dbapi_connection, _connection_record) -> None:
    """Configures each new SQLite connection.

    The driver's own transaction handling is disabled so that `_on_begin`
    decides how transactions start.
    """
    dbapi_connection.isolation_level = None
    cursor = dbapi_connection.cursor()
    # WAL: readers do not block the writer, across processes
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _on_begin(connection) -> None:
    """Starts every transaction with the write lock.

    Rules checked before writing (tag conflicts) then hold against other
    processes using the same file.
    """
    connection.exec_driver_sql("BEGIN IMMEDIATE")


def _on_error(context) -> None:
    """Raises a database error as `FedbiomedError`.

    The message keeps SQLite's reason and the failing statement; the
    statement's parameters are left out, as they may hold sensitive values.
    """
    raise FedbiomedError(
        f"{ErrorNumbers.FB634.value}: {context.original_exception}; "
        f"statement: {context.statement}"
    ) from context.original_exception


@cache
def _engine(path: str) -> Engine:
    """Returns the engine of a database file, one per file and process."""
    # timeout: seconds to wait for another process holding the write lock
    engine = create_engine(f"sqlite:///{path}", connect_args={"timeout": 30})
    event.listen(engine, "connect", _on_connect)
    event.listen(engine, "begin", _on_begin)
    event.listen(engine, "handle_error", _on_error)
    return engine


def session_factory(path: str, models: List[type[Base]]) -> sessionmaker[Session]:
    """Returns the session factory of a database file.

    Creates the tables of `models` that do not exist yet. Each
    `with factory.begin() as session:` block is one transaction, committed at
    the end or rolled back on error. Objects stay usable after the block.

    Args:
        path: Path of the SQLite file of the component.
        models: Models whose tables the caller uses.

    Raises:
        FedbiomedError: a model's `__log_level__` is not one of `LOG_LEVELS`.
    """
    for model in models:
        if model.__log_level__ not in LOG_LEVELS:
            raise FedbiomedError(
                f"{ErrorNumbers.FB634.value}: {model.__name__}.__log_level__ is "
                f"'{model.__log_level__}', expected one of {LOG_LEVELS}"
            )
    engine = _engine(path)
    Base.metadata.create_all(engine, tables=[model.__table__ for model in models])
    factory = sessionmaker(engine, expire_on_commit=False)
    # Recorded before validation, so a refused entry is logged as a failure
    event.listen(factory, "before_flush", _record_writes)
    event.listen(factory, "before_flush", _validate_entries)
    event.listen(factory, "do_orm_execute", _record_statements)
    event.listen(factory, "after_commit", _on_commit)
    event.listen(factory, "after_rollback", _on_rollback)
    return factory

import inspect as pyinspect
import sqlite3
import threading
import time
from typing import Any

import pytest
from sqlalchemy import JSON, delete, event, func, inspect, select, update
from sqlalchemy.orm import Mapped, mapped_column

import fedbiomed.common.db as db_mod
from fedbiomed.common.db import REDACTED, Base, _engine, _matches, session_factory
from fedbiomed.common.exceptions import FedbiomedError


class Item(Base):
    __tablename__ = "TestItems"

    item_id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    tags: Mapped[list[str]] = mapped_column(JSON)
    data: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    note: Mapped[str | None] = mapped_column(default=None)


class Other(Base):
    __tablename__ = "TestOthers"

    other_id: Mapped[str] = mapped_column(primary_key=True)


class Secret(Base):
    __tablename__ = "TestSecrets"

    secret_id: Mapped[str] = mapped_column(primary_key=True)
    # "key" is a sensitive field name: its values are redacted in messages
    key: Mapped[dict[str, str]] = mapped_column(JSON)


class WriteItem(Base):
    __tablename__ = "TestWriteItems"
    __log_level__ = "writes"

    item_id: Mapped[str] = mapped_column(primary_key=True)


class QuietItem(Base):
    __tablename__ = "TestQuietItems"
    __log_level__ = "none"

    item_id: Mapped[str] = mapped_column(primary_key=True)


class TypoItem(Base):
    __tablename__ = "TestTypoItems"
    __log_level__ = "write"

    item_id: Mapped[str] = mapped_column(primary_key=True)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "db.sqlite")


@pytest.fixture
def sessions(db_path):
    return session_factory(db_path, [Item, Secret, WriteItem, QuietItem])


@pytest.fixture
def events(monkeypatch):
    """Security events written, with the function `stacklevel` points to."""
    written = []

    def security_event(stacklevel, **fields):
        frame = pyinspect.currentframe()
        for _ in range(stacklevel):
            frame = frame.f_back
        written.append({**fields, "caller": frame.f_code.co_name})

    monkeypatch.setattr(db_mod.logger, "security_event", security_event)
    return written


def _item(**kwargs):
    return {"item_id": "i1", "name": "n", "tags": ["a"], **kwargs}


def test_from_dict_builds_entry():
    assert Item.from_dict(_item()).name == "n"


@pytest.mark.parametrize(
    "data, reason",
    [
        (_item(unknown=1), "unexpected keyword argument 'unknown'"),
        ({"item_id": "i1"}, "missing 2 required keyword-only arguments"),
    ],
)
def test_from_dict_rejects_unknown_and_missing_fields(data, reason):
    with pytest.raises(FedbiomedError, match=f"FB632.*Item: .*{reason}"):
        Item.from_dict(data)


def test_to_dict_returns_set_columns_only():
    assert Item.from_dict(_item(note=None)).to_dict() == _item()


def test_from_dict_treats_none_as_not_given():
    with pytest.raises(FedbiomedError, match="missing 1 required"):
        Item.from_dict(_item(name=None))


@pytest.mark.parametrize(
    "value, expected, result",
    [
        ({"a": [1, None]}, Any, True),
        (None, list[str] | None, True),
        ({"k": [1]}, list[int] | dict[str, list[int]], True),
        ([1], list[int] | dict[str, list[int]], True),
        ("x", list[int] | dict[str, list[int]], False),
        (["a", 1], list[str], False),
        ({"a": "b"}, dict[str, str], True),
        ({1: "b"}, dict[str, str], False),
        ({"a": 1}, dict[str, str], False),
        ("a", list[str], False),
    ],
)
def test_matches_checks_json_value_against_annotation(value, expected, result):
    assert _matches(value, expected) is result


def test_session_factory_creates_only_the_given_tables(db_path, sessions):
    tables = inspect(_engine(db_path)).get_table_names()
    assert "TestItems" in tables
    assert "TestOthers" not in tables


def test_engine_is_shared_by_factories_of_the_same_file(db_path, sessions):
    other = session_factory(db_path, [Other])
    assert other.kw["bind"] is sessions.kw["bind"] is _engine(db_path)


def test_connection_uses_wal_and_foreign_keys(db_path, sessions):
    with sessions.begin() as session:
        connection = session.connection()
        assert connection.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_transaction_starts_with_write_lock(db_path, sessions):
    statements = []
    engine = _engine(db_path)

    def listener(_conn, _cursor, statement, *_args):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", listener)
    try:
        with sessions.begin() as session:
            session.get(Item, "i1")
    finally:
        event.remove(engine, "before_cursor_execute", listener)
    assert statements[0] == "BEGIN IMMEDIATE"


def test_entry_is_usable_after_the_transaction(sessions):
    with sessions.begin() as session:
        entry = Item.from_dict(_item())
        session.add(entry)
    assert entry.to_dict() == _item()

    with sessions.begin() as session:
        assert session.get(Item, "i1").to_dict() == _item()


def test_failed_transaction_writes_nothing(sessions):
    with pytest.raises(FedbiomedError):
        with sessions.begin() as session:
            session.add(Item.from_dict(_item(item_id="i2")))
            session.add(Item.from_dict(_item(tags=[1])))

    with sessions.begin() as session:
        assert session.get(Item, "i2") is None


def test_insert_is_validated_when_written(sessions):
    entry = Item.from_dict(_item(tags=[1]))
    with pytest.raises(FedbiomedError, match=r"FB632.*Item.tags: expected list\[str\]"):
        with sessions.begin() as session:
            session.add(entry)


def test_update_reuses_insert_validation(sessions):
    with sessions.begin() as session:
        session.add(Item.from_dict(_item()))

    with pytest.raises(FedbiomedError, match=r"FB632.*Item.data: expected"):
        with sessions.begin() as session:
            session.get(Item, "i1").data = ["not", "a", "dict"]

    with sessions.begin() as session:
        session.get(Item, "i1").tags = ["b"]
    with sessions.begin() as session:
        assert session.get(Item, "i1").tags == ["b"]


def test_validation_error_redacts_sensitive_values():
    entry = Secret(secret_id="s", key={"private": 1})
    with pytest.raises(FedbiomedError, match="<redacted>") as error:
        entry.validate()
    assert "private" not in str(error.value)


def test_database_error_keeps_reason_and_logs_statement_without_parameters(
    sessions, monkeypatch
):
    debug = []
    monkeypatch.setattr(db_mod.logger, "debug", debug.append)
    with sessions.begin() as session:
        session.add(Item.from_dict(_item(name="hidden-value")))

    with pytest.raises(FedbiomedError) as error:
        with sessions.begin() as session:
            session.add(Item.from_dict(_item(name="other-hidden-value")))

    message = str(error.value)
    assert message.startswith("FB634")
    assert "UNIQUE constraint failed: TestItems.item_id" in message
    assert "INSERT" not in message
    (logged,) = debug
    assert 'INSERT INTO "TestItems"' in logged
    assert "hidden-value" not in message + logged


def test_case_insensitive_search_folds_accented_letters(sessions):
    with sessions.begin() as session:
        session.add(Item.from_dict(_item(name="ÉTUDE IRM")))

    with sessions.begin() as session:
        found = session.scalars(select(Item).where(Item.name.icontains("étude")))
        assert [item.item_id for item in found] == ["i1"]
        assert session.scalar(select(func.lower(None))) is None


def test_session_factory_refuses_unknown_log_level(db_path):
    with pytest.raises(
        FedbiomedError, match="FB634.*TypoItem.__log_level__ is 'write'"
    ):
        session_factory(db_path, [TypoItem])


def test_writes_are_logged_after_commit_with_caller(sessions, events):
    with sessions.begin() as session:
        session.add(Item.from_dict(_item()))
        session.flush()
        assert events == []

    assert events == [
        {
            "status": "success",
            "operation": "insert",
            "table": "TestItems",
            "doc_id": "i1",
            "db_args": events[0]["db_args"],
            "caller": "test_writes_are_logged_after_commit_with_caller",
        }
    ]
    assert "name=n" in events[0]["db_args"]
    events.clear()

    with sessions.begin() as session:
        entry = session.get(Item, "i1")
        entry.name = "renamed"
        entry.note = None
    with sessions.begin() as session:
        session.delete(session.get(Item, "i1"))

    writes = [e for e in events if e["operation"] != "select"]
    assert [(e["operation"], e["doc_id"]) for e in writes] == [
        ("update", "i1"),
        ("remove", "i1"),
    ]
    assert writes[0]["db_args"].startswith("{name=renamed")
    assert writes[1]["db_args"] == "{}"


def test_reads_and_statements_are_logged_with_their_parameters(sessions, events):
    with sessions.begin() as session:
        session.add(Item.from_dict(_item()))
    events.clear()

    with sessions.begin() as session:
        session.get(Item, "i1")
        session.scalars(select(Item).where(Item.name == "n")).all()
        session.execute(update(Item).where(Item.item_id == "i1").values(name="x"))
        session.execute(delete(Item).where(Item.item_id == "other"))

    assert [(e["operation"], e["db_args"]) for e in events] == [
        ("select", "{pk_1=i1...}"),
        ("select", "{name_1=n...}"),
        ("update", "{name=x..., item_id_1=i1...}"),
        ("remove", "{item_id_1=other...}"),
    ]


def test_unchanged_entry_is_not_logged_as_update(sessions, events):
    with sessions.begin() as session:
        session.add(Item.from_dict(_item()))
    events.clear()

    with sessions.begin() as session:
        session.get(Item, "i1").name = "n"

    assert [e["operation"] for e in events] == ["select"]


def test_failed_transaction_is_logged_with_its_error(sessions, events):
    with pytest.raises(FedbiomedError):
        with sessions.begin() as session:
            session.add(Item.from_dict(_item(tags=[1])))

    (failure,) = events
    assert failure["status"] == "failure"
    assert failure["operation"] == "insert"
    assert failure["error_type"] == "FedbiomedError"
    assert "Item.tags" in failure["error_message"]


def test_rollback_without_error_is_logged_as_failure(sessions, events):
    session = sessions()
    session.add(Item.from_dict(_item()))
    session.flush()
    session.rollback()

    (failure,) = events
    assert failure["status"] == "failure"
    assert "error_type" not in failure


def test_write_events_redact_sensitive_values(sessions, events):
    with sessions.begin() as session:
        session.add(Secret(secret_id="s", key={"private": "x"}))

    assert REDACTED in events[0]["db_args"]
    assert "private" not in events[0]["db_args"]


@pytest.mark.parametrize(
    "model, logged",
    [
        (WriteItem, ["insert"]),
        (QuietItem, []),
    ],
)
def test_log_level_selects_what_is_logged(sessions, events, model, logged):
    with sessions.begin() as session:
        session.add(model(item_id="i1"))
        session.flush()
        session.scalars(select(model)).all()

    assert [e["operation"] for e in events] == logged


def test_writer_waits_for_another_writer_instead_of_failing(db_path, sessions):
    locked = threading.Event()

    def hold_write_lock():
        connection = sqlite3.connect(db_path, isolation_level=None)
        connection.execute("BEGIN IMMEDIATE")
        locked.set()
        time.sleep(0.5)
        connection.execute("COMMIT")
        connection.close()

    holder = threading.Thread(target=hold_write_lock)
    holder.start()
    locked.wait()
    with sessions.begin() as session:
        session.add(Item.from_dict(_item()))
    holder.join()

    with sessions.begin() as session:
        assert session.get(Item, "i1") is not None

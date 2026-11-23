import inspect as pyinspect
import sqlite3
import threading
import time
from typing import Any

import fedbiomed.common.db as db_mod
import pytest
from fedbiomed.common.db import DatabaseModel, get_database
from fedbiomed.common.exceptions import FedbiomedError
from sqlalchemy import JSON, delete, event, inspect, select, text, update
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Item(DatabaseModel):
    __tablename__ = "TestItems"

    item_id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    tags: Mapped[list[str]] = mapped_column(JSON)
    data: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    note: Mapped[str | None] = mapped_column(default=None)


class Other(DatabaseModel):
    __tablename__ = "TestOthers"

    other_id: Mapped[str] = mapped_column(primary_key=True)


class Secret(DatabaseModel):
    __tablename__ = "TestSecrets"
    __sensitive_fields__ = ("key",)

    secret_id: Mapped[str] = mapped_column(primary_key=True)
    key: Mapped[dict[str, str]] = mapped_column(JSON)


class WriteItem(DatabaseModel):
    __tablename__ = "TestWriteItems"
    __security_log__ = DatabaseModel.SecurityLog.WRITES

    item_id: Mapped[str] = mapped_column(primary_key=True)


class QuietItem(DatabaseModel):
    __tablename__ = "TestQuietItems"
    __security_log__ = DatabaseModel.SecurityLog.NONE

    item_id: Mapped[str] = mapped_column(primary_key=True)


class _OtherBase(DeclarativeBase):
    pass


class OtherBaseItem(_OtherBase):
    """Model of another library: the security log listens to every session."""

    __tablename__ = "TestOtherBaseItems"

    item_id: Mapped[str] = mapped_column(primary_key=True)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "db.sqlite")


@pytest.fixture
def database(db_path):
    database = get_database(db_path)
    database.create_tables([Item, Secret, WriteItem, QuietItem])
    return database


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
    with pytest.raises(FedbiomedError, match=f"FB634.*Item: .*{reason}"):
        Item.from_dict(data)


def test_to_dict_returns_set_columns_only():
    assert Item.from_dict(_item(note=None)).to_dict() == _item()


def test_from_dict_treats_none_as_not_given():
    with pytest.raises(FedbiomedError, match="missing 1 required"):
        Item.from_dict(_item(name=None))


def test_connection_uses_foreign_keys(db_path, database):
    with database.transaction() as session:
        connection = session.connection()
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_entry_is_saved(database):
    with database.transaction() as session:
        session.add(Item.from_dict(_item()))

    with database.transaction() as session:
        assert session.get(Item, "i1").to_dict() == _item()


def test_failed_transaction_writes_nothing(database):
    with database.transaction() as session:
        session.add(Item.from_dict(_item()))

    with pytest.raises(FedbiomedError):
        with database.transaction() as session:
            session.add(Item.from_dict(_item(item_id="i2")))
            session.add(Item.from_dict(_item()))

    with database.transaction() as session:
        assert session.get(Item, "i2") is None


def test_database_error_keeps_reason_and_logs_statement_without_parameters(
    database, monkeypatch
):
    debug = []
    monkeypatch.setattr(db_mod.logger, "debug", debug.append)
    with database.transaction() as session:
        session.add(Item.from_dict(_item(name="hidden-value")))

    with pytest.raises(FedbiomedError) as error:
        with database.transaction() as session:
            session.add(Item.from_dict(_item(name="other-hidden-value")))

    message = str(error.value)
    assert message.startswith("FB634: Database error: Item: ")
    assert "UNIQUE constraint failed: TestItems.item_id" in message
    assert "INSERT" not in message
    (logged,) = debug
    assert 'INSERT INTO "TestItems"' in logged
    assert "hidden-value" not in message + logged


def test_database_error_outside_a_table_names_no_model(database):
    with pytest.raises(
        FedbiomedError, match="^FB634: Database error: no such table: Missing$"
    ):
        with database.transaction() as session:
            session.execute(text("SELECT * FROM Missing"))


def test_writes_are_logged_after_commit_with_caller(database, events):
    with database.transaction() as session:
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

    with database.transaction() as session:
        entry = session.get(Item, "i1")
        entry.name = "renamed"
        entry.note = None
    with database.transaction() as session:
        session.delete(session.get(Item, "i1"))

    writes = [e for e in events if e["operation"] != "select"]
    assert [(e["operation"], e["doc_id"]) for e in writes] == [
        ("update", "i1"),
        ("remove", "i1"),
    ]
    assert writes[0]["db_args"].startswith("{name=renamed")
    assert writes[1]["db_args"] == "{}"


def test_reads_and_statements_are_logged_with_their_parameters(database, events):
    with database.transaction() as session:
        session.add(Item.from_dict(_item()))
    events.clear()

    with database.transaction() as session:
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


def test_unchanged_entry_is_not_logged_as_update(database, events):
    with database.transaction() as session:
        session.add(Item.from_dict(_item()))
    events.clear()

    with database.transaction() as session:
        session.get(Item, "i1").name = "n"

    assert [e["operation"] for e in events] == ["select"]


def test_failed_transaction_is_logged_with_its_error(database, events):
    with database.transaction() as session:
        session.add(Item.from_dict(_item()))
    events.clear()

    with pytest.raises(FedbiomedError):
        with database.transaction() as session:
            session.add(Item.from_dict(_item()))

    (failure,) = events
    assert failure["status"] == "failure"
    assert failure["operation"] == "insert"
    assert failure["error_type"] == "FedbiomedError"
    assert "UNIQUE constraint failed: TestItems.item_id" in failure["error_message"]


def test_rollback_without_error_is_logged_as_failure(database, events):
    with database.transaction() as session:
        session.add(Item.from_dict(_item()))
        session.flush()
        session.rollback()

    (failure,) = events
    assert failure["status"] == "failure"
    assert "error_type" not in failure


def test_write_events_redact_sensitive_values(database, events):
    with database.transaction() as session:
        session.add(Secret(secret_id="s", key={"private": "s3cret"}))

    (inserted,) = events
    assert "s3cret" not in inserted["db_args"]
    assert "secret_id=s" in inserted["db_args"]


def test_statement_parameters_of_sensitive_columns_are_redacted(database, events):
    with database.transaction() as session:
        session.execute(delete(Secret).where(Secret.key == {"private": "s3cret"}))

    (removed,) = events
    assert "s3cret" not in removed["db_args"]
    assert "key_1=" in removed["db_args"]


@pytest.mark.parametrize(
    "model, logged",
    [
        (WriteItem, ["insert"]),
        (QuietItem, []),
    ],
)
def test_security_log_selects_what_is_logged(database, events, model, logged):
    with database.transaction() as session:
        session.add(model(item_id="i1"))
        session.flush()
        session.scalars(select(model)).all()

    assert [e["operation"] for e in events] == logged


def test_rows_of_other_models_are_not_logged(database, events):
    database.create_tables([OtherBaseItem])
    with database.transaction() as session:
        session.add(OtherBaseItem(item_id="i1"))
        session.flush()
        session.scalars(select(OtherBaseItem)).all()

    assert events == []


def test_writer_waits_for_another_writer_instead_of_failing(db_path, database):
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
    with database.transaction() as session:
        session.add(Item.from_dict(_item()))
    holder.join()

    with database.transaction() as session:
        assert session.get(Item, "i1") is not None


def test_get_database_returns_one_instance_per_file(tmp_path):
    first = get_database(str(tmp_path / "a.json"))
    assert get_database(str(tmp_path / "a.json")) is first
    assert get_database(str(tmp_path / "b.json")) is not first


def test_get_database_uses_sqlite_file_next_to_tinydb_file(tmp_path):
    get_database(str(tmp_path / "db_node.json")).create_tables([Item])
    assert (tmp_path / "db_node.sqlite").exists()


def test_create_tables_creates_only_the_given_tables(database):
    tables = inspect(database._engine).get_table_names()
    assert "TestItems" in tables
    assert "TestOthers" not in tables


def test_transaction_starts_with_write_lock(database):
    statements = []
    engine = database._engine

    def listener(_conn, _cursor, statement, *_args):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", listener)
    try:
        with database.transaction() as session:
            session.get(Item, "i1")
    finally:
        event.remove(engine, "before_cursor_execute", listener)
    assert statements[0] == "BEGIN IMMEDIATE"

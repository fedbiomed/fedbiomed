from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from fedbiomed.common.db import TinyDBConnector, TinyTableConnector


class _TestTable(TinyTableConnector):
    _table_name = "test_table"
    _id_name = "id"


@pytest.fixture
def security_event_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr("fedbiomed.common.db.logger.security_event", mock)
    return mock


@pytest.fixture
def table(tmp_path: Path) -> Generator[_TestTable, None, None]:
    # TinyDBConnector is a singleton; reset between tests for isolation.
    TinyDBConnector._instance = None

    db_path = str(tmp_path / "tinydb.json")
    tbl = _TestTable(db_path)

    yield tbl

    # Close TinyDB file handles if they exist.
    if TinyDBConnector._instance is not None:
        try:
            TinyDBConnector._instance.db.close()
        except Exception:
            pass
    TinyDBConnector._instance = None


def _assert_security_call(
    call_kwargs: Dict[str, Any], *, operation: str, table_name: str
):
    assert call_kwargs.get("operation") == operation
    assert call_kwargs.get("table_name") == table_name


def test_get_by_id_logs_search_not_found(
    table: _TestTable, security_event_mock: MagicMock
):
    res = table.get_by_id("abc")
    assert res is None

    security_event_mock.assert_called_once()
    kwargs = security_event_mock.call_args.kwargs
    _assert_security_call(kwargs, operation="Database search", table_name="test_table")
    assert kwargs["id"] == "abc"
    assert kwargs["result"] == "Not found"


def test_insert_logs_search_then_insert(
    table: _TestTable, security_event_mock: MagicMock
):
    security_event_mock.reset_mock()

    _returned_id = table.insert({"id": "1", "x": 1})

    assert security_event_mock.call_count == 2
    first = security_event_mock.call_args_list[0].kwargs
    second = security_event_mock.call_args_list[1].kwargs

    _assert_security_call(first, operation="Database search", table_name="test_table")
    assert first["id"] == "1"
    assert first["result"] == "Not found"

    _assert_security_call(second, operation="Database insert", table_name="test_table")
    assert second["id"] == "1"
    assert second["result"] == "Success"


def test_all_logs_fetch_all(table: _TestTable, security_event_mock: MagicMock):
    table.insert({"id": "1", "x": 1})
    table.insert({"id": "2", "x": 2})

    security_event_mock.reset_mock()
    out = table.all()
    assert len(out) == 2

    security_event_mock.assert_called_once()
    kwargs = security_event_mock.call_args.kwargs
    _assert_security_call(
        kwargs, operation="Database fetch all", table_name="test_table"
    )
    assert kwargs["result"] == "Found 2 entries"


def test_get_all_by_value_logs_search(
    table: _TestTable, security_event_mock: MagicMock
):
    table.insert({"id": "1", "tag": "a"})
    table.insert({"id": "2", "tag": "b"})

    security_event_mock.reset_mock()
    res = table.get_all_by_value(by="tag", value="a")
    assert len(res) == 1

    security_event_mock.assert_called_once()
    kwargs = security_event_mock.call_args.kwargs
    _assert_security_call(kwargs, operation="Database search", table_name="test_table")
    assert kwargs["tag"] == "a"
    assert kwargs["result"] == "Found 1 entries"


def test_get_all_by_condition_logs_search(
    table: _TestTable, security_event_mock: MagicMock
):
    table.insert({"id": "1", "n": 1})
    table.insert({"id": "2", "n": 3})

    def cond(x):
        return x > 1

    security_event_mock.reset_mock()
    res = table.get_all_by_condition(by="n", cond=cond)
    assert len(res) == 1

    security_event_mock.assert_called_once()
    kwargs = security_event_mock.call_args.kwargs
    _assert_security_call(kwargs, operation="Database search", table_name="test_table")
    assert kwargs["search_field"] == "n"
    assert kwargs["search_condition"] is cond
    assert kwargs["result"] == "Found 1 entries"


def test_delete_by_id_logs_delete(table: _TestTable, security_event_mock: MagicMock):
    table.insert({"id": "1", "x": 1})
    table.insert({"id": "2", "x": 2})

    security_event_mock.reset_mock()
    deleted = table.delete_by_id("1")
    assert isinstance(deleted, list)
    assert len(deleted) == 1

    security_event_mock.assert_called_once()
    kwargs = security_event_mock.call_args.kwargs
    _assert_security_call(kwargs, operation="Database delete", table_name="test_table")
    assert kwargs["id"] == ["1"]
    assert kwargs["result"] == "Deleted 1 entries"


def test_update_by_id_logs_search_update_search(
    table: _TestTable, security_event_mock: MagicMock
):
    table.insert({"id": "1", "x": 1})

    security_event_mock.reset_mock()
    updated = table.update_by_id("1", {"x": 10})
    assert updated["x"] == 10

    # update_by_id: get_by_id (Found) -> update (Success) -> get_by_id (Found)
    assert security_event_mock.call_count == 3
    ops = [c.kwargs.get("operation") for c in security_event_mock.call_args_list]
    assert ops == ["Database search", "Database update", "Database search"]

    upd_kwargs = security_event_mock.call_args_list[1].kwargs
    _assert_security_call(
        upd_kwargs, operation="Database update", table_name="test_table"
    )
    assert upd_kwargs["id"] == "1"
    assert upd_kwargs["result"] == "Success"


def test_update_by_id_missing_logs_search_then_raises(
    table: _TestTable, security_event_mock: MagicMock
):
    security_event_mock.reset_mock()

    with pytest.raises(KeyError):
        table.update_by_id("missing", {"x": 1})

    # Only the search attempt should be logged.
    security_event_mock.assert_called_once()
    kwargs = security_event_mock.call_args.kwargs
    _assert_security_call(kwargs, operation="Database search", table_name="test_table")
    assert kwargs["id"] == "missing"
    assert kwargs["result"] == "Not found"

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import Mock

import pytest

import fedbiomed.common.db as db_mod
from fedbiomed.common.db import _security_log
from fedbiomed.common.exceptions import FedbiomedError


def test_security_log_success_strips_forbidden_and_truncates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    security_event = Mock()
    security_context = Mock(side_effect=lambda **_: nullcontext())
    debug = Mock()

    monkeypatch.setattr(db_mod.logger, "security_event", security_event)
    monkeypatch.setattr(db_mod.logger, "security_context", security_context)
    monkeypatch.setattr(db_mod.logger, "debug", debug)

    class _FakeTable:
        name = "my_table"

        @_security_log("insert", default_stacklevel=3)
        def insert(self, payload: dict, **kwargs):
            # Ensure decorator popped stacklevel and didn't pass it through.
            self.kwargs_seen = dict(kwargs)
            return 7

    table = _FakeTable()

    payload = {
        "ok": "value",
        "certificate": "SUPER-SECRET",
        "key": "ALSO-SECRET",
        "nested": {"secagg_elem": "SECRET", "keep": 1},
        "long": "x" * 200,
    }

    result = table.insert(payload, stacklevel=42, extra_kw="y" * 200)

    assert result == 7
    assert "stacklevel" not in table.kwargs_seen

    security_context.assert_called_once_with(operation="insert", table="my_table")
    security_event.assert_called_once()
    debug.assert_called_once()

    logged = security_event.call_args.kwargs
    assert logged["status"] == "success"
    assert logged["doc_id"] == 7
    assert logged["stacklevel"] == 42

    debug_args = debug.call_args
    assert "insert in table" in debug_args.args[0]
    assert "db_args=" in debug_args.args[0]
    assert "db_kwargs=" in debug_args.args[0]
    assert debug_args.kwargs["extra"]["db_args"] == logged["db_args"]
    assert debug_args.kwargs["extra"]["db_kwargs"] == logged["db_kwargs"]
    assert debug_args.kwargs["stacklevel"] == 42

    # Forbidden keys stripped from payload/kwargs logging
    assert "certificate" not in logged["db_args"]
    assert "key" not in logged["db_args"]
    assert logged["db_args"]["nested"] == {"keep": 1}

    # Truncation applied (50 chars + '...')
    assert isinstance(logged["db_args"]["long"], str)
    assert logged["db_args"]["long"].endswith("...")
    assert len(logged["db_args"]["long"]) <= 53
    assert isinstance(logged["db_kwargs"]["extra_kw"], str)
    assert logged["db_kwargs"]["extra_kw"].endswith("...")


def test_security_log_success_summarizes_search_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    security_event = Mock()
    monkeypatch.setattr(db_mod.logger, "security_event", security_event)
    monkeypatch.setattr(db_mod.logger, "security_context", lambda **_: nullcontext())

    class _FakeTable:
        name = "my_table"

        @_security_log("search", default_stacklevel=3)
        def search(self, query: str, **kwargs):
            return ["a", "b", "c"]

    table = _FakeTable()

    _ = table.search("hello")

    logged = security_event.call_args.kwargs
    assert logged["status"] == "success"
    assert logged["doc_id"] == "3 documents"
    assert logged["db_args"]["arg0"] == "hello"
    assert logged["stacklevel"] == 3


def test_security_log_failure_logs_and_wraps_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    security_event = Mock()
    debug = Mock()
    monkeypatch.setattr(db_mod.logger, "security_event", security_event)
    monkeypatch.setattr(db_mod.logger, "security_context", lambda **_: nullcontext())
    monkeypatch.setattr(db_mod.logger, "debug", debug)

    class _FakeTable:
        name = "my_table"

        @_security_log("remove", default_stacklevel=3)
        def remove(self, payload: dict, **kwargs):
            raise ValueError("boom")

    table = _FakeTable()

    with pytest.raises(FedbiomedError) as excinfo:
        table.remove({"certificate": "SECRET", "keep": "ok"}, stacklevel=9)

    assert "Failed to remove" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)

    logged = security_event.call_args.kwargs
    assert logged["status"] == "failure"
    assert logged["details"] == "boom"
    assert logged["stacklevel"] == 9

    debug.assert_called_once()
    debug_args = debug.call_args
    assert "Failed to remove" in debug_args.args[0]
    assert "db_args=" in debug_args.args[0]
    assert "db_kwargs=" in debug_args.args[0]
    assert debug_args.kwargs["extra"]["db_args"] == logged["db_args"]
    assert debug_args.kwargs["extra"]["db_kwargs"] == logged["db_kwargs"]
    assert debug_args.kwargs["stacklevel"] == 9

    # Forbidden keys stripped from args logging even on failure
    assert "certificate" not in logged["db_args"]
    assert logged["db_args"]["keep"] == "ok"

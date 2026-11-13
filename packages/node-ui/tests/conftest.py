import pytest


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch):
    """Keep GUI database state local to each test."""
    from fedbiomed.common.db import TinyDBConnector

    monkeypatch.setattr(TinyDBConnector, "_instance", None)

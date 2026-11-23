import json
import sys
from datetime import datetime, timedelta, timezone

import psutil
import pytest
from fedbiomed.node.db_models import (
    NodeConnectionState,
    NodeConnectionStateHistory,
    NodeProcessState,
    NodeProcessStateHistory,
)
from fedbiomed.node.node_pm import (
    DEFAULT_NODE_ARGS,
    NodeConnectionStateManager,
    NodeProcessManager,
    NodeState,
)
from fedbiomed.transport.client import (
    Channels,
    ClientStatus,
    ResearcherCredentials,
)
from sqlalchemy import inspect, select

# Recent, so the 30-day history retention keeps what tests write
NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _config(mocker, root, node_id="node-1", node_name="Node 1", db_name="node_db.json"):
    config = mocker.MagicMock()
    config.root = str(root)
    config.getpath.return_value = str(root / db_name)

    def _get(section, key):
        values = {
            ("default", "pid"): 12345,
            ("default", "id"): node_id,
            ("default", "name"): node_name,
            ("default", "db"): db_name,
        }
        return values[(section, key)]

    config.get.side_effect = _get
    return config


def _store(manager, model, **fields):
    """Saves a row, replacing the one with the same key."""
    with manager._database.transaction() as session:
        session.merge(model(**fields))


def _stored(manager, model):
    """The rows of a table, without the history's generated `id`."""
    with manager._database.transaction() as session:
        rows = [row.to_dict() for row in session.scalars(select(model))]
    for row in rows:
        row.pop("id", None)
    return rows


def _tables(manager):
    return inspect(manager._database._engine).get_table_names()


@pytest.fixture
def _manager(mocker, tmp_path):
    """Create a manager on a database in `tmp_path`."""
    return NodeProcessManager(_config(mocker, tmp_path))


def _process_state(**kwargs):
    return {
        "node_id": "node-1",
        "node_name": "Node 1",
        "pid": 1234,
        "state": NodeState.STOPPED.value,
        "action": "stop",
        **kwargs,
    }


def test_node_pm_init_creates_missing_tables(mocker, tmp_path):
    """The first manager built on a database creates its empty tables."""
    manager = NodeProcessManager(_config(mocker, tmp_path))

    assert {"NodeProcessStates", "NodeProcessStateHistory"} <= set(_tables(manager))
    assert _stored(manager, NodeProcessState) == []


@pytest.mark.parametrize("background", [True, False])
def test_node_pm_start(mocker, tmp_path, _manager, background):
    config = _config(mocker, tmp_path)
    node_args = {"gpu": False, "debug": True, "background": background}
    manager = _manager

    manager.get_status = mocker.MagicMock(return_value=NodeState.UNKNOWN)
    manager._set_process_state = mocker.MagicMock()
    manager._wait = mocker.MagicMock()

    process = mocker.MagicMock()
    process.pid = 12345

    mock_popen = mocker.patch(
        "fedbiomed.node.node_pm.subprocess.Popen",
        return_value=process,
    )

    manager.start(
        node_args=node_args,
        background=background,
        actor={"source": "gui"},
    )

    mock_popen.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "fedbiomed.node.node_pm",
            "--config",
            config.root,
            "--node-args",
            json.dumps(node_args),
        ],
    )

    mock_popen.assert_called_once()
    manager._set_process_state.assert_called_once_with(
        pid=12345,
        state=NodeState.RUNNING,
        action="start",
        actor={"source": "gui"},
        reason="start_requested",
        node_args=node_args,
        background=background,
    )
    if not background:
        manager._wait.assert_called_once_with(process, actor={"source": "gui"})


@pytest.mark.parametrize(
    "status, should_set_stopped",
    [
        (NodeState.RUNNING, True),
        (NodeState.STOPPED, False),
    ],
)
def test_node_pm_wait(
    mocker,
    _manager,
    status,
    should_set_stopped,
):
    manager = _manager
    manager._set_process_state = mocker.MagicMock()
    manager.get_status = mocker.MagicMock(return_value=status)

    process = mocker.MagicMock()
    process.pid = 12345
    process.wait.return_value = 0

    mocker.patch(
        "fedbiomed.node.node_pm.psutil.Process",
        return_value=process,
    )

    exit_code = manager._wait(
        process=process,
        actor={"source": "cli"},
    )

    assert exit_code == 0
    process.wait.assert_called_once_with()

    if should_set_stopped:
        manager._set_process_state.assert_called_once_with(
            pid=12345,
            state=NodeState.STOPPED,
            action="wait",
            actor={"source": "cli"},
            reason="process_exited_abruptly",
            exit_code=0,
        )
    else:
        manager._set_process_state.assert_not_called()


@pytest.mark.parametrize(
    "wait_side_effect, expected_exit_code, should_kill",
    [
        ([0], 0, False),
        ([psutil.TimeoutExpired(pid=12345, seconds=5), -9], -9, True),
    ],
)
def test_node_pm_stop(
    mocker,
    _manager,
    wait_side_effect,
    expected_exit_code,
    should_kill,
):
    manager = _manager
    manager._set_process_state = mocker.MagicMock()
    manager.get_status = mocker.MagicMock(return_value=NodeState.RUNNING)
    manager._get_pid = mocker.MagicMock(return_value=12345)

    process = mocker.MagicMock()
    process.pid = 12345
    process.wait.side_effect = wait_side_effect
    process.is_running.return_value = False

    mocker.patch(
        "fedbiomed.node.node_pm.psutil.Process",
        return_value=process,
    )

    manager.stop(
        actor={"source": "gui"},
        reason="test_stop",
    )

    process.terminate.assert_called_once_with()

    if should_kill:
        process.kill.assert_called_once_with()
        assert process.wait.call_count == 2
    else:
        process.kill.assert_not_called()
        process.wait.assert_called_once_with(timeout=5)

    manager._set_process_state.assert_any_call(
        pid=12345,
        state=NodeState.STOPPING,
        action="stop",
        actor={"source": "gui"},
        reason="test_stop",
    )

    manager._set_process_state.assert_any_call(
        pid=12345,
        state=NodeState.STOPPED,
        action="stop",
        actor={"source": "gui"},
        reason="test_stop",
        exit_code=expected_exit_code,
    )


@pytest.mark.parametrize(
    "node_args, background, expected_node_args, expected_background",
    [
        (
            None,
            None,
            {
                "gpu": True,
                "gpu_num": 2,
                "gpu_only": True,
                "debug": True,
            },
            True,
        ),
        (
            {"gpu_num": 4},
            None,
            {
                "gpu": True,
                "gpu_num": 4,
                "gpu_only": True,
                "debug": True,
            },
            True,
        ),
        (
            {
                "gpu": False,
                "gpu_num": 0,
                "gpu_only": False,
                "debug": False,
            },
            False,
            {
                "gpu": False,
                "gpu_num": 0,
                "gpu_only": False,
                "debug": False,
            },
            False,
        ),
    ],
)
def test_node_pm_restart_inherits_and_overrides_saved_settings(
    mocker,
    _manager,
    node_args,
    background,
    expected_node_args,
    expected_background,
):
    manager = _manager
    _store(
        manager,
        NodeProcessState,
        **_process_state(
            node_args={
                "gpu": True,
                "gpu_num": 2,
                "gpu_only": True,
                "debug": True,
            },
            background=True,
        ),
    )

    manager.stop = mocker.MagicMock()
    manager.start = mocker.MagicMock()

    manager.restart(
        node_args=node_args,
        background=background,
        actor={"source": "gui"},
    )

    manager.stop.assert_called_once_with(
        actor={"source": "gui"},
        reason="restart_requested",
    )

    manager.start.assert_called_once_with(
        node_args=expected_node_args,
        background=expected_background,
        actor={"source": "gui"},
        reason="restart_requested",
    )


def test_node_pm_restart_uses_cli_defaults_without_saved_state(mocker, _manager):
    manager = _manager
    manager.stop = mocker.MagicMock()
    manager.start = mocker.MagicMock()

    manager.restart(actor={"source": "cli"})

    manager.start.assert_called_once_with(
        node_args=DEFAULT_NODE_ARGS,
        background=False,
        actor={"source": "cli"},
        reason="restart_requested",
    )


def test_node_pm_set_process_state(mocker, _manager):
    manager = _manager
    manager._node_id = "node_id1"
    _store(manager, NodeProcessState, **_process_state(node_id="node_id1"))

    mocker.patch("fedbiomed.node.node_pm._utc_now", return_value=NOW)
    mocker.patch.object(
        NodeProcessManager, "_build_actor", return_value={"source": "local"}
    )
    manager._set_process_state(
        pid=1234,
        state=NodeState.RUNNING,
        action="start",
        actor={"source": "local"},
        reason="start_requested",
        node_args={"gpu": True, "gpu_num": 2},
        background=True,
    )

    (state_entry,) = _stored(manager, NodeProcessState)
    (history_entry,) = _stored(manager, NodeProcessStateHistory)

    assert state_entry["node_id"] == "node_id1"
    assert state_entry["pid"] == 1234
    assert NodeState(state_entry["state"]) == NodeState.RUNNING
    assert state_entry["started_at"] == NOW
    assert state_entry["actor"] == {"source": "local"}
    assert state_entry["node_args"] == {"gpu": True, "gpu_num": 2}
    assert state_entry["background"] is True
    assert history_entry == state_entry


def test_node_pm_set_process_state_preserves_execution_settings(mocker, _manager):
    manager = _manager
    manager._node_id = "node_id1"
    _store(
        manager,
        NodeProcessState,
        **_process_state(
            node_id="node_id1",
            state=NodeState.RUNNING.value,
            started_at=NOW - timedelta(days=1),
            node_args={
                "gpu": True,
                "gpu_num": 3,
                "gpu_only": False,
                "debug": True,
            },
            background=True,
        ),
    )

    mocker.patch("fedbiomed.node.node_pm._utc_now", return_value=NOW)
    manager._set_process_state(
        pid=1234,
        state=NodeState.STOPPED,
        action="stop",
    )

    (state_entry,) = _stored(manager, NodeProcessState)
    assert state_entry["node_args"] == {
        "gpu": True,
        "gpu_num": 3,
        "gpu_only": False,
        "debug": True,
    }
    assert state_entry["background"] is True
    assert state_entry["stopped_at"] == NOW


def test_node_pm_set_process_state_merges_partial_node_args(mocker, _manager):
    manager = _manager
    manager._node_id = "node_id1"
    _store(
        manager,
        NodeProcessState,
        **_process_state(
            node_id="node_id1",
            state=NodeState.RUNNING.value,
            started_at=NOW - timedelta(days=1),
            node_args={
                "gpu": True,
                "gpu_num": 3,
                "gpu_only": False,
                "debug": True,
            },
            background=True,
        ),
    )

    mocker.patch("fedbiomed.node.node_pm._utc_now", return_value=NOW)
    manager._set_process_state(
        pid=1234,
        state=NodeState.RUNNING,
        action="status_check",
        node_args={"gpu_num": 4, "debug": False},
    )

    (state_entry,) = _stored(manager, NodeProcessState)
    assert state_entry["node_args"] == {
        "gpu": True,
        "gpu_num": 4,
        "gpu_only": False,
        "debug": False,
    }
    assert state_entry["background"] is True


def test_node_pm_set_process_state_resets_started_at_after_stop(mocker, _manager):
    manager = _manager
    manager._node_id = "node_id1"
    _store(
        manager,
        NodeProcessState,
        **_process_state(
            node_id="node_id1",
            state=NodeState.STOPPED.value,
            started_at=NOW - timedelta(days=2),
            stopped_at=NOW - timedelta(days=1),
            exit_code=0,
        ),
    )

    mocker.patch("fedbiomed.node.node_pm._utc_now", return_value=NOW)
    mocker.patch.object(
        NodeProcessManager, "_build_actor", return_value={"source": "local"}
    )
    manager._set_process_state(
        pid=1234,
        state=NodeState.RUNNING,
        action="start",
        actor={"source": "local"},
        reason="start_requested",
    )

    (state_entry,) = _stored(manager, NodeProcessState)

    assert state_entry["started_at"] == NOW
    assert "stopped_at" not in state_entry
    assert "exit_code" not in state_entry


def test_node_pm_set_process_state_resets_started_at_for_forced_start(
    mocker,
    _manager,
):
    manager = _manager
    manager._node_id = "node_id1"
    _store(
        manager,
        NodeProcessState,
        **_process_state(
            node_id="node_id1",
            state=NodeState.RUNNING.value,
            pid=1234,
            started_at=NOW - timedelta(days=1),
        ),
    )

    mocker.patch("fedbiomed.node.node_pm._utc_now", return_value=NOW)
    mocker.patch.object(
        NodeProcessManager, "_build_actor", return_value={"source": "cli"}
    )

    manager._set_process_state(
        pid=5678,
        state=NodeState.RUNNING,
        action="start",
        actor={"source": "cli"},
        reason="start_requested",
    )

    (state_entry,) = _stored(manager, NodeProcessState)
    assert state_entry["pid"] == 5678
    assert state_entry["state"] == NodeState.RUNNING.value
    assert state_entry["started_at"] == NOW


def test_node_pm_start_process_already_started(mocker, _manager):
    manager = _manager
    manager.get_status = mocker.MagicMock(return_value=NodeState.RUNNING)

    mock_popen = mocker.patch("fedbiomed.node.node_pm.subprocess.Popen")
    mock_logger = mocker.patch("fedbiomed.node.node_pm.logger")

    manager.start(node_args={"gpu": False}, actor={"source": "gui"})

    mock_logger.warning.assert_called_once_with(
        "Node process is already running. Ignoring start request."
    )
    mock_popen.assert_not_called()


def test_node_pm_force_start_process_already_started(mocker, _manager):
    manager = _manager
    manager.get_status = mocker.MagicMock(return_value=NodeState.RUNNING)
    manager._set_process_state = mocker.MagicMock()

    process = mocker.MagicMock(pid=54321)
    mock_popen = mocker.patch(
        "fedbiomed.node.node_pm.subprocess.Popen",
        return_value=process,
    )
    mock_logger = mocker.patch("fedbiomed.node.node_pm.logger")

    manager.start(
        node_args={"gpu": False},
        background=True,
        actor={"source": "cli"},
        force=True,
    )

    mock_popen.assert_called_once()
    manager._set_process_state.assert_called_once_with(
        pid=54321,
        state=NodeState.RUNNING,
        action="start",
        actor={"source": "cli"},
        reason="start_requested",
        node_args={"gpu": False},
        background=True,
    )
    mock_logger.warning.assert_called_once_with(
        "Forcing node startup while the database reports it as running. "
        "The previous node process might not have closed properly; "
        "this may cause a process leak."
    )


@pytest.mark.parametrize(
    "status",
    [
        (NodeState.STOPPING),
        (NodeState.STOPPED),
    ],
)
def test_node_pm_stop_process_already_stopped(mocker, _manager, status):
    manager = _manager

    manager.get_status = mocker.MagicMock(return_value=status)
    mock_logger = mocker.patch("fedbiomed.node.node_pm.logger")

    manager.stop()

    mock_logger.warning.assert_called_once_with(
        "Node process is already stopped. Ignoring stop request."
    )


@pytest.mark.parametrize(
    "stored_state, _is_process_active, expected_status",
    [
        (NodeState.RUNNING, True, NodeState.RUNNING),
        (NodeState.STOPPED, False, NodeState.STOPPED),
        (None, False, NodeState.UNKNOWN),
    ],
)
def test_node_pm_get_status(
    mocker, _manager, stored_state, _is_process_active, expected_status
):
    manager = _manager
    manager._is_process_active = mocker.MagicMock(return_value=_is_process_active)

    if stored_state is not None:
        _store(
            manager,
            NodeProcessState,
            **_process_state(pid=12345, state=stored_state.value),
        )

    status = manager.get_status()

    assert status == expected_status


def test_node_pm_process_state_returns_stored_entry(mocker, _manager):
    manager = _manager

    manager.get_status = mocker.MagicMock(return_value=NodeState.RUNNING)

    _store(
        manager,
        NodeProcessState,
        **_process_state(
            pid=12345,
            state=NodeState.RUNNING.value,
            action="start",
            reason="start_requested",
            actor={"source": "gui"},
            updated_at=NOW,
            started_at=NOW,
        ),
    )

    state = manager.get_process_state()

    manager.get_status.assert_called_once()

    assert state["pid"] == 12345
    assert state["state"] == NodeState.RUNNING.value
    assert state["node_id"] == "node-1"
    assert state["node_name"] == "Node 1"
    assert state["actor"] == {"source": "gui"}
    assert state.get("node_args") is None
    assert state.get("background") is None


def test_node_pm_process_state_returns_unknown_without_stored_entry(mocker, _manager):
    manager = _manager
    manager.get_status = mocker.MagicMock(return_value=NodeState.UNKNOWN)

    state = manager.get_process_state()

    assert state.get("pid") is None
    assert state["state"] == NodeState.UNKNOWN.value
    assert state["node_id"] == "node-1"
    assert state["node_name"] == "Node 1"
    assert state.get("action") is None


def test_node_pm_process_state_requires_node_id(mocker, tmp_path):
    manager = NodeProcessManager(_config(mocker, tmp_path, node_id=None))

    with pytest.raises(Exception, match="FB327"):
        manager.get_process_state()


def test_node_pm_set_process_state_needs_node_id(mocker, tmp_path):
    manager = NodeProcessManager(_config(mocker, tmp_path, node_id=None))

    manager._set_process_state(pid=1, state=NodeState.RUNNING, action="start")

    assert _stored(manager, NodeProcessState) == []


def test_node_pm_set_process_state_survives_db_failure(mocker, _manager):
    mocker.patch.object(
        _manager._database, "transaction", side_effect=RuntimeError("db is gone")
    )
    warning = mocker.patch("fedbiomed.node.node_pm.logger.warning")

    _manager._set_process_state(pid=1, state=NodeState.RUNNING, action="start")

    warning.assert_called_once()


@pytest.mark.parametrize(
    "state, expected_pid", [(NodeState.RUNNING, 1234), (NodeState.STOPPED, None)]
)
def test_node_pm_get_pid_only_of_a_running_process(_manager, state, expected_pid):
    _store(_manager, NodeProcessState, **_process_state(state=state.value))

    assert _manager._get_pid() == expected_pid


def test_node_pm_cleanup_process_state_history_removes_entries_past_retention(
    mocker, _manager
):
    for age_in_days in (31, 10):
        _store(
            _manager,
            NodeProcessStateHistory,
            **_process_state(
                node_id="node-retention",
                state="running",
                action=f"start-{age_in_days}",
                updated_at=datetime.now(timezone.utc) - timedelta(days=age_in_days),
            ),
        )

    with _manager._database.transaction() as session:
        NodeProcessManager._cleanup_process_state_history(session)

    remaining = _stored(_manager, NodeProcessStateHistory)
    assert [entry["action"] for entry in remaining] == ["start-10"]


def test_node_pm_set_process_state_prunes_history(mocker, _manager):
    """History is pruned as it grows, not only when a manager is built."""
    cleanup = mocker.patch.object(NodeProcessManager, "_cleanup_process_state_history")

    _manager._set_process_state(pid=1, state=NodeState.RUNNING, action="start")

    cleanup.assert_called_once()


@pytest.fixture
def _connection_manager(mocker, tmp_path):
    """Create a connection state manager on a database in `tmp_path`."""
    return NodeConnectionStateManager(_config(mocker, tmp_path))


def _connection_state(**kwargs):
    return {
        "state": ClientStatus.CONNECTED,
        "host": "localhost",
        "port": "50051",
        "mtls": True,
        **kwargs,
    }


def test_node_connection_init_creates_missing_tables(mocker, tmp_path):
    """The GUI builds one of these to read: the first one creates its empty tables."""
    manager = NodeConnectionStateManager(_config(mocker, tmp_path))

    assert {"NodeConnectionStates", "NodeConnectionStateHistory"} <= set(
        _tables(manager)
    )
    assert manager.get_connection_state() is None


def test_node_connection_set_state_prunes_history_past_retention(
    mocker, _connection_manager
):
    """History is pruned as it grows: a node process may run for months."""
    for age_in_days in (31, 10):
        _store(
            _connection_manager,
            NodeConnectionStateHistory,
            node_id="node-1",
            state="connected",
            host="localhost",
            port="50051",
            operation=f"retained-{age_in_days}",
            updated_at=datetime.now(timezone.utc) - timedelta(days=age_in_days),
        )

    _connection_manager.set_state(**_connection_state(operation="node_starting"))

    remaining = _stored(_connection_manager, NodeConnectionStateHistory)
    assert [entry["operation"] for entry in remaining] == [
        "retained-10",
        "node_starting",
    ]


def test_node_connection_set_state_records_state_and_history(
    mocker, _connection_manager
):
    manager = _connection_manager

    manager.set_state(
        **_connection_state(
            researcher_id="researcher-1",
            identity_verified=True,
            operation="researcher_channel_established",
            certificate={"subject": "CN=researcher-1"},
        )
    )

    (entry,) = _stored(manager, NodeConnectionState)
    assert entry["state"] == "connected"
    assert entry["researcher_id"] == "researcher-1"
    assert entry["identity_verified"] is True
    assert entry["certificate"] == {"subject": "CN=researcher-1"}
    assert entry["started_at"] == entry["updated_at"]
    assert _stored(manager, NodeConnectionStateHistory) == [entry]


def _repeated_state(updated_at):
    """A stored row that `_repeat_state` reports again, stamped at `updated_at`."""
    return {
        "node_id": "node-1",
        "state": "disconnected",
        "host": "localhost",
        "port": "50051",
        "operation": "researcher_unavailable",
        "reason": "not available",
        "started_at": NOW,
        "updated_at": updated_at,
    }


def _repeat_state(manager):
    manager.set_state(
        **_connection_state(
            state=ClientStatus.DISCONNECTED,
            operation="researcher_unavailable",
            reason="not available",
        )
    )


def _ago(**delta):
    """A past instant, to the second as the managers store it."""
    return datetime.now(timezone.utc).replace(microsecond=0) - timedelta(**delta)


def test_node_connection_set_state_refreshes_stale_repeated_state(
    mocker, _connection_manager
):
    manager = _connection_manager
    updated_at = _ago(minutes=30)
    _store(manager, NodeConnectionState, **_repeated_state(updated_at))

    _repeat_state(manager)

    (entry,) = _stored(manager, NodeConnectionState)
    assert entry["updated_at"] > updated_at
    assert entry == {**_repeated_state(updated_at), "updated_at": entry["updated_at"]}
    assert _stored(manager, NodeConnectionStateHistory) == []


def test_node_connection_set_state_honours_the_refresh_interval(
    mocker, _connection_manager
):
    """An unreachable researcher asks for a rarer refresh than the default."""
    manager = _connection_manager
    recent = _ago(minutes=10)
    _store(manager, NodeConnectionState, **_repeated_state(recent))

    manager.set_state(
        **_connection_state(
            state=ClientStatus.DISCONNECTED,
            operation="researcher_unavailable",
            reason="not available",
        ),
        refresh_interval=timedelta(minutes=30),
    )
    assert _stored(manager, NodeConnectionState)[0]["updated_at"] == recent

    stale = _ago(minutes=31)
    _store(manager, NodeConnectionState, **_repeated_state(stale))
    manager.set_state(
        **_connection_state(
            state=ClientStatus.DISCONNECTED,
            operation="researcher_unavailable",
            reason="not available",
        ),
        refresh_interval=timedelta(minutes=30),
    )
    assert _stored(manager, NodeConnectionState)[0]["updated_at"] > stale


def test_node_connection_set_state_leaves_recent_repeated_state_alone(
    mocker, _connection_manager
):
    """An unreachable researcher repeats every 2s, and each write locks the file."""
    manager = _connection_manager
    _store(manager, NodeConnectionState, **_repeated_state(_ago(seconds=30)))

    _repeat_state(manager)

    assert _stored(manager, NodeConnectionState) == [_repeated_state(_ago(seconds=30))]
    assert _stored(manager, NodeConnectionStateHistory) == []


def test_node_connection_set_state_carries_last_error_forward(
    mocker, _connection_manager
):
    manager = _connection_manager
    _store(
        manager,
        NodeConnectionState,
        node_id="node-1",
        state="failed",
        host="localhost",
        port="50051",
        reason="handshake failed",
        last_error="handshake failed",
        last_error_at=NOW,
    )

    manager.set_state(**_connection_state(operation="researcher_channel_established"))

    (entry,) = _stored(manager, NodeConnectionState)
    assert entry["last_error"] == "handshake failed"
    assert entry["last_error_at"] == NOW
    assert "reason" not in entry


def test_node_connection_set_state_records_failure_as_last_error(
    mocker, _connection_manager
):
    manager = _connection_manager

    manager.set_state(
        **_connection_state(
            state=ClientStatus.FAILED,
            operation="mtls_identity_rejected",
            reason="identity rejected",
        )
    )

    (entry,) = _stored(manager, NodeConnectionState)
    assert entry["state"] == "failed"
    assert entry["last_error"] == "identity rejected"
    assert entry["last_error_at"] == entry["updated_at"]


def test_node_connection_set_state_survives_db_failure(mocker, _connection_manager):
    manager = _connection_manager
    mocker.patch.object(
        manager._database, "transaction", side_effect=RuntimeError("db is gone")
    )
    warning = mocker.patch("fedbiomed.node.node_pm.logger.warning")

    manager.set_state(**_connection_state())

    warning.assert_called_once()


def test_node_connection_get_connection_state(mocker, _connection_manager):
    manager = _connection_manager

    assert manager.get_connection_state() is None

    _store(
        manager,
        NodeConnectionState,
        node_id="node-1",
        state="connected",
        host="localhost",
        port="50051",
    )

    assert manager.get_connection_state()["state"] == "connected"


def test_node_connection_channel_report_reaches_the_manager(_connection_manager):
    """What the transport reports and what the manager records are one contract."""
    channels = Channels(
        researcher=ResearcherCredentials(
            host="localhost", port="50051", certificate=b"researcher-cert"
        ),
        on_connection_state=_connection_manager.set_state,
    )
    channels.researcher_id = "researcher-1"

    channels.report_state(
        ClientStatus.FAILED,
        operation="mtls_identity_rejected",
        reason="identity rejected",
        identity_verified=False,
    )

    (entry,) = _stored(_connection_manager, NodeConnectionState)
    assert entry["state"] == "failed"
    assert entry["operation"] == "mtls_identity_rejected"
    assert entry["researcher_id"] == "researcher-1"
    assert entry["identity_verified"] is False
    assert (entry["host"], entry["port"]) == ("localhost", "50051")


def test_node_connection_get_connection_history_newest_first(
    mocker, _connection_manager
):
    manager = _connection_manager
    for state in ("failed", "disconnected", "connected"):
        _store(
            manager,
            NodeConnectionStateHistory,
            node_id="node-1",
            state=state,
            host="localhost",
            port="50051",
        )

    history = manager.get_connection_history(limit=2)

    assert [entry["state"] for entry in history] == ["connected", "disconnected"]


def test_node_connection_without_node_id_records_nothing(mocker, tmp_path):
    manager = NodeConnectionStateManager(_config(mocker, tmp_path, node_id=None))

    manager.set_state(**_connection_state())

    assert _stored(manager, NodeConnectionState) == []
    assert manager.get_connection_state() is None
    assert manager.get_connection_history() == []

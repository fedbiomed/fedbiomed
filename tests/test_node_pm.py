import json
import sys
from datetime import datetime, timedelta, timezone

import psutil
import pytest

from fedbiomed.node.dataset_manager._db_tables import NodeProcessStateHistoryTable
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


def _iso(moment: datetime) -> str:
    """Timestamp in the format the managers store."""
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _config(mocker, root, node_id="node-1", node_name="Node 1", db_name="node_db.json"):
    config = mocker.MagicMock()
    config.root = str(root)

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


@pytest.fixture
def _manager(mocker, tmp_path):
    """Create a manager with safe mocked DB tables.

    Do not patch _cleanup_process_state_history here: some tests need the real
    implementation.
    """
    state_table = mocker.MagicMock()
    history_table = mocker.MagicMock()

    state_table.get_by_id.return_value = None

    mocker.patch.object(
        NodeProcessManager,
        "_get_state_table",
        return_value=state_table,
    )
    mocker.patch.object(
        NodeProcessManager,
        "_get_history_table",
        return_value=history_table,
    )

    manager = NodeProcessManager(_config(mocker, tmp_path))

    # Test-only handles so tests can configure/assert the shared table mocks.
    # Direct access to manager._state_table and manager._history_table does not exist
    # in the current implementation.
    manager._state_table = state_table
    manager._history_table = history_table

    return manager


def test_node_pm_init_writes_nothing(mocker, tmp_path):
    """Constructing a manager opens no table: a reader leaves the database alone."""
    state_table = mocker.patch("fedbiomed.node.node_pm.NodeProcessStateTable")
    history_table = mocker.patch("fedbiomed.node.node_pm.NodeProcessStateHistoryTable")

    NodeProcessManager(_config(mocker, tmp_path))

    state_table.assert_not_called()
    history_table.assert_not_called()


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
    manager._state_table.get_by_id.return_value = {
        "node_args": {
            "gpu": True,
            "gpu_num": 2,
            "gpu_only": True,
            "debug": True,
        },
        "background": True,
    }

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
    manager._state_table.get_by_id.return_value = None
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

    state_table = manager._state_table
    history_table = manager._history_table

    state_table.get_by_id.return_value = {
        "started_at": None,
        "stopped_at": None,
    }

    mocker.patch("fedbiomed.node.node_pm._utc_now", return_value="utc-now")
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

    state_table.update_or_insert_by_id.assert_called_once()
    assert state_table.update_or_insert_by_id.call_args.args[0] == "node_id1"

    history_table.insert.assert_called_once()

    state_entry = state_table.update_or_insert_by_id.call_args.args[1]
    history_entry = history_table.insert.call_args.args[0]

    assert state_entry["node_id"] == "node_id1"
    assert state_entry["pid"] == 1234
    assert NodeState(state_entry["state"]) == NodeState.RUNNING
    assert state_entry["started_at"] == "utc-now"
    assert state_entry["actor"] == {"source": "local"}
    assert state_entry["node_args"] == {"gpu": True, "gpu_num": 2}
    assert state_entry["background"] is True
    assert history_entry == state_entry


def test_node_pm_set_process_state_preserves_execution_settings(mocker, _manager):
    manager = _manager
    manager._node_id = "node_id1"
    manager._state_table.get_by_id.return_value = {
        "state": NodeState.RUNNING.value,
        "started_at": "start-time",
        "node_args": {
            "gpu": True,
            "gpu_num": 3,
            "gpu_only": False,
            "debug": True,
        },
        "background": True,
    }

    mocker.patch("fedbiomed.node.node_pm._utc_now", return_value="stop-time")
    manager._set_process_state(
        pid=1234,
        state=NodeState.STOPPED,
        action="stop",
    )

    state_entry = manager._state_table.update_or_insert_by_id.call_args.args[1]
    assert state_entry["node_args"] == {
        "gpu": True,
        "gpu_num": 3,
        "gpu_only": False,
        "debug": True,
    }
    assert state_entry["background"] is True
    assert state_entry["stopped_at"] == "stop-time"


def test_node_pm_set_process_state_merges_partial_node_args(mocker, _manager):
    manager = _manager
    manager._node_id = "node_id1"
    manager._state_table.get_by_id.return_value = {
        "state": NodeState.RUNNING.value,
        "started_at": "start-time",
        "node_args": {
            "gpu": True,
            "gpu_num": 3,
            "gpu_only": False,
            "debug": True,
        },
        "background": True,
    }

    mocker.patch("fedbiomed.node.node_pm._utc_now", return_value="update-time")
    manager._set_process_state(
        pid=1234,
        state=NodeState.RUNNING,
        action="status_check",
        node_args={"gpu_num": 4, "debug": False},
    )

    state_entry = manager._state_table.update_or_insert_by_id.call_args.args[1]
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

    state_table = manager._state_table
    state_table.get_by_id.return_value = {
        "state": NodeState.STOPPED.value,
        "started_at": "previous-start",
        "stopped_at": "previous-stop",
    }

    mocker.patch("fedbiomed.node.node_pm._utc_now", return_value="new-start")
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

    state_entry = state_table.update_or_insert_by_id.call_args.args[1]

    assert state_entry["started_at"] == "new-start"
    assert "stopped_at" not in state_entry


def test_node_pm_set_process_state_resets_started_at_for_forced_start(
    mocker,
    _manager,
):
    manager = _manager
    manager._node_id = "node_id1"

    state_table = manager._state_table
    state_table.get_by_id.return_value = {
        "state": NodeState.RUNNING.value,
        "pid": 1234,
        "started_at": "previous-start",
    }

    mocker.patch("fedbiomed.node.node_pm._utc_now", return_value="forced-start")
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

    state_entry = state_table.update_or_insert_by_id.call_args.args[1]
    assert state_entry["pid"] == 5678
    assert state_entry["state"] == NodeState.RUNNING.value
    assert state_entry["started_at"] == "forced-start"


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
    state_table = manager._state_table
    manager._is_process_active = mocker.MagicMock(return_value=_is_process_active)

    if stored_state is None:
        state_table.get_by_id.return_value = None
    else:
        state_table.get_by_id.return_value = {
            "pid": 12345,
            "state": stored_state,
        }

    status = manager.get_status()

    assert status == expected_status
    state_table.get_by_id.assert_called_with("node-1")


def test_node_pm_process_state_returns_stored_entry(mocker, _manager):
    manager = _manager

    manager.get_status = mocker.MagicMock(return_value=NodeState.RUNNING)
    manager._get_pid = mocker.MagicMock(return_value=12345)

    state_table = manager._state_table

    stored = {
        "pid": 12345,
        "state": NodeState.RUNNING,
        "node_id": "node-1",
        "node_name": "Node 1",
        "action": "start",
        "reason": "start_requested",
        "actor": {"source": "gui"},
        "updated_at": "utc-now",
        "started_at": "utc-now",
        "stopped_at": None,
        "exit_code": None,
    }

    state_table.get_by_id.return_value = stored

    state = manager.get_process_state()

    manager._get_pid.assert_called_once()
    manager.get_status.assert_called_once()
    state_table.get_by_id.assert_called_with("node-1")

    assert state.pid == 12345
    assert state.state == NodeState.RUNNING
    assert state.node_id == "node-1"
    assert state.node_name == "Node 1"
    assert state.actor == {"source": "gui"}
    assert state.node_args is None
    assert state.background is None


def test_node_pm_process_state_returns_unknown_without_stored_entry(mocker, _manager):
    manager = _manager
    manager.get_status = mocker.MagicMock(return_value=NodeState.UNKNOWN)
    manager._get_pid = mocker.MagicMock(return_value=None)
    manager._state_table.get_by_id.return_value = None

    state = manager.get_process_state()

    assert state.pid is None
    assert state.state == NodeState.UNKNOWN.value
    assert state.node_id == "node-1"
    assert state.node_name == "Node 1"
    assert state.action is None


def test_node_pm_cleanup_process_state_history_removes_entries_past_retention(
    mocker, tmp_path
):
    """Run against a real table: a mocked one hid that this cleanup did nothing."""
    history_table = NodeProcessStateHistoryTable(str(tmp_path / "db.json"))
    entry = {
        "node_id": "node-retention",
        "node_name": "N",
        "pid": 1,
        "state": "running",
    }
    for age_in_days in (31, 10):
        history_table.insert(
            {
                **entry,
                "action": f"start-{age_in_days}",
                "updated_at": _iso(
                    datetime.now(timezone.utc) - timedelta(days=age_in_days)
                ),
            }
        )

    mocker.patch.object(
        NodeProcessManager, "_get_history_table", return_value=history_table
    )
    NodeProcessManager(_config(mocker, tmp_path))._cleanup_process_state_history()

    remaining = history_table.get_all_by_value("node_id", "node-retention")
    assert [entry["action"] for entry in remaining] == ["start-10"]


def test_node_pm_set_process_state_prunes_history(mocker, _manager):
    """History is pruned as it grows, not only when a manager is built."""
    cleanup = mocker.patch.object(NodeProcessManager, "_cleanup_process_state_history")

    _manager._set_process_state(pid=1, state=NodeState.RUNNING, action="start")

    cleanup.assert_called_once()


def test_node_pm_get_table_reinitializes_state_and_history_tables(mocker, tmp_path):
    state_table_constructor = mocker.patch(
        "fedbiomed.node.node_pm.NodeProcessStateTable"
    )
    history_table_constructor = mocker.patch(
        "fedbiomed.node.node_pm.NodeProcessStateHistoryTable"
    )

    mocker.patch.object(NodeProcessManager, "_cleanup_process_state_history")

    manager = NodeProcessManager(_config(mocker, tmp_path))

    manager._get_state_table()
    manager._get_state_table()
    manager._get_history_table()
    manager._get_history_table()

    assert state_table_constructor.call_count == 2
    assert history_table_constructor.call_count == 2


@pytest.fixture
def _connection_manager(mocker, tmp_path):
    """Create a connection state manager with mocked DB tables."""
    state_table = mocker.MagicMock()
    history_table = mocker.MagicMock()

    state_table.get_by_id.return_value = None
    history_table.get_all_by_value.return_value = []

    mocker.patch.object(
        NodeConnectionStateManager,
        "_get_state_table",
        return_value=state_table,
    )
    mocker.patch.object(
        NodeConnectionStateManager,
        "_get_history_table",
        return_value=history_table,
    )

    manager = NodeConnectionStateManager(_config(mocker, tmp_path))

    # Test-only handles on the shared table mocks.
    manager._state_table = state_table
    manager._history_table = history_table

    return manager


def _connection_state(**kwargs):
    return {
        "state": ClientStatus.CONNECTED,
        "host": "localhost",
        "port": "50051",
        "mtls": True,
        **kwargs,
    }


def test_node_connection_init_writes_nothing(mocker, tmp_path):
    """The GUI builds one of these to read: constructing it opens no table."""
    state_table = mocker.patch("fedbiomed.node.node_pm.NodeConnectionStateTable")
    history_table = mocker.patch(
        "fedbiomed.node.node_pm.NodeConnectionStateHistoryTable"
    )

    NodeConnectionStateManager(_config(mocker, tmp_path))

    state_table.assert_not_called()
    history_table.assert_not_called()


def test_node_connection_set_state_prunes_history_past_retention(
    mocker, _connection_manager
):
    """History is pruned as it grows: a node process may run for months."""
    _connection_manager.set_state(**_connection_state(operation="node_starting"))

    cutoff = _connection_manager._history_table.delete_older_than.call_args.args[0]
    assert datetime.fromisoformat(cutoff.replace("Z", "+00:00")) < datetime.now(
        timezone.utc
    ) - timedelta(days=29)


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

    entry = manager._state_table.replace_by_id.call_args.args[1]
    assert entry["state"] == "connected"
    assert entry["researcher_id"] == "researcher-1"
    assert entry["identity_verified"] is True
    assert entry["certificate"] == {"subject": "CN=researcher-1"}
    assert entry["started_at"] == entry["updated_at"]
    manager._history_table.insert.assert_called_once_with(entry)


def _repeated_state(updated_at):
    """A stored row that `_repeated_call` reports again, stamped at `updated_at`."""
    return {
        "node_id": "node-1",
        "state": "disconnected",
        "host": "localhost",
        "port": "50051",
        "operation": "researcher_unavailable",
        "reason": "not available",
        "started_at": "2026-08-01T10:00:00Z",
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


def test_node_connection_set_state_refreshes_stale_repeated_state(
    mocker, _connection_manager
):
    manager = _connection_manager
    manager._state_table.get_by_id.return_value = _repeated_state(
        _iso(datetime.now(timezone.utc) - timedelta(minutes=30))
    )

    _repeat_state(manager)

    manager._state_table.replace_by_id.assert_not_called()
    manager._history_table.insert.assert_not_called()
    node_id, update = manager._state_table.update_by_id.call_args.args
    assert node_id == "node-1"
    assert list(update) == ["updated_at"]


def test_node_connection_set_state_leaves_recent_repeated_state_alone(
    mocker, _connection_manager
):
    """An unreachable researcher repeats every 2s, and each write rewrites the file."""
    manager = _connection_manager
    manager._state_table.get_by_id.return_value = _repeated_state(
        _iso(datetime.now(timezone.utc) - timedelta(seconds=30))
    )

    _repeat_state(manager)

    manager._state_table.update_by_id.assert_not_called()
    manager._state_table.replace_by_id.assert_not_called()
    manager._history_table.insert.assert_not_called()


def test_node_connection_set_state_carries_last_error_forward(
    mocker, _connection_manager
):
    manager = _connection_manager
    manager._state_table.get_by_id.return_value = {
        "node_id": "node-1",
        "state": "failed",
        "host": "localhost",
        "port": "50051",
        "last_error": "handshake failed",
        "last_error_at": "2026-08-01T10:00:00Z",
    }

    manager.set_state(**_connection_state(operation="researcher_channel_established"))

    entry = manager._state_table.replace_by_id.call_args.args[1]
    assert entry["last_error"] == "handshake failed"
    assert entry["last_error_at"] == "2026-08-01T10:00:00Z"
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

    entry = manager._state_table.replace_by_id.call_args.args[1]
    assert entry["state"] == "failed"
    assert entry["last_error"] == "identity rejected"
    assert entry["last_error_at"] == entry["updated_at"]


def test_node_connection_set_state_survives_db_failure(mocker, _connection_manager):
    manager = _connection_manager
    manager._state_table.get_by_id.side_effect = RuntimeError("db is gone")
    warning = mocker.patch("fedbiomed.node.node_pm.logger.warning")

    manager.set_state(**_connection_state())

    warning.assert_called_once()


def test_node_connection_get_connection_state(mocker, _connection_manager):
    manager = _connection_manager

    assert manager.get_connection_state() is None

    manager._state_table.get_by_id.return_value = {
        "node_id": "node-1",
        "state": "connected",
        "host": "localhost",
        "port": "50051",
    }

    assert manager.get_connection_state().state == "connected"


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

    entry = _connection_manager._state_table.replace_by_id.call_args.args[1]
    assert entry["state"] == "failed"
    assert entry["operation"] == "mtls_identity_rejected"
    assert entry["researcher_id"] == "researcher-1"
    assert entry["identity_verified"] is False
    assert (entry["host"], entry["port"]) == ("localhost", "50051")


def test_node_connection_get_connection_history_newest_first(
    mocker, _connection_manager
):
    manager = _connection_manager
    manager._history_table.get_all_by_value.return_value = [
        {
            "node_id": "node-1",
            "state": state,
            "host": "localhost",
            "port": "50051",
        }
        for state in ("failed", "disconnected", "connected")
    ]

    history = manager.get_connection_history(limit=2)

    assert [entry.state for entry in history] == ["connected", "disconnected"]

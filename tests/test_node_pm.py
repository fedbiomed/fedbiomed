import json
import sys

import psutil
import pytest

from fedbiomed.node.node_pm import NodeProcessManager, NodeState


def _config(mocker, node_id="node-1", node_name="Node 1", db_name="node_db.json"):
    config = mocker.MagicMock()
    config.root = "/tmp/node-root"

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


def test_node_pm_01_start(mocker):
    config = _config(mocker)
    node_args = {"gpu": False, "debug": True}

    manager = NodeProcessManager(config)
    manager._init_state_tables = mocker.MagicMock()
    manager._set_process_state = mocker.MagicMock()

    process = mocker.MagicMock()
    process.pid = 12345

    mock_popen = mocker.patch(
        "fedbiomed.node.node_pm.subprocess.Popen",
        return_value=process,
    )

    pid = manager.start(
        node_args=node_args,
        actor={"source": "gui"},
    )

    assert pid == 12345

    manager._init_state_tables.assert_called_once_with()

    mock_popen.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "fedbiomed.node.node_pm",
            "--config",
            config.root,
            "--node-args",
            json.dumps(node_args),
        ]
    )

    manager._set_process_state.assert_called_once_with(
        pid=12345,
        state=NodeState.RUNNING,
        action="start",
        actor={"source": "gui"},
        reason="process_started",
    )


@pytest.mark.parametrize(
    "status, should_set_stopped",
    [
        (NodeState.RUNNING.value, True),
        (NodeState.STOPPED.value, False),
    ],
)
def test_node_pm_02_wait(
    mocker,
    status,
    should_set_stopped,
):
    manager = NodeProcessManager(_config(mocker))
    manager._set_process_state = mocker.MagicMock()
    manager.get_status = mocker.MagicMock(return_value=status)

    process = mocker.MagicMock()
    process.wait.return_value = 0

    mocker.patch(
        "fedbiomed.node.node_pm.psutil.Process",
        return_value=process,
    )

    exit_code = manager.wait(
        pid=12345,
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
def test_node_pm_03_stop(
    mocker,
    wait_side_effect,
    expected_exit_code,
    should_kill,
):
    manager = NodeProcessManager(_config(mocker))
    manager._set_process_state = mocker.MagicMock()
    manager.get_status = mocker.MagicMock(return_value=NodeState.RUNNING.value)

    process = mocker.MagicMock()
    process.pid = 12345
    process.wait.side_effect = wait_side_effect

    mocker.patch(
        "fedbiomed.node.node_pm.psutil.Process",
        return_value=process,
    )

    manager.stop(
        pid=12345,
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


def test_node_pm_04_restart(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager.stop = mocker.MagicMock()
    manager.start = mocker.MagicMock(return_value=67890)

    new_pid = manager.restart(
        pid=12345,
        node_args={"gpu": False},
        actor={"source": "gui"},
    )

    assert new_pid == 67890

    manager.stop.assert_called_once_with(
        pid=12345,
        actor={"source": "gui"},
        reason="restart_requested",
    )

    manager.start.assert_called_once_with(
        pid=12345,
        node_args={"gpu": False},
        actor={"source": "gui"},
    )


def test_node_pm_05_set_process_state(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager._node_id = "node-1"
    manager._state_table = mocker.MagicMock()
    manager._history_table = mocker.MagicMock()
    manager._state_table.get_by_id.return_value = {
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
        reason="process_started",
    )

    manager._state_table.update_or_insert_by_id.assert_called_once()
    assert manager._state_table.update_or_insert_by_id.call_args.args[0] == 1234

    manager._history_table.insert.assert_called_once()

    state_entry = manager._state_table.update_or_insert_by_id.call_args.args[1]
    history_entry = manager._history_table.insert.call_args.args[0]

    assert state_entry["node_id"] == "node-1"
    assert state_entry["pid"] == 1234
    assert state_entry["state"] == NodeState.RUNNING.value
    assert state_entry["started_at"] == "utc-now"
    assert state_entry["stopped_at"] is None
    assert state_entry["actor"] == {"source": "local"}
    assert history_entry == state_entry


@pytest.mark.parametrize(
    "status",
    [
        (NodeState.STARTING.value),
        (NodeState.RUNNING.value),
    ],
)
def test_node_pm_06_start_process_already_started(mocker, status):
    manager = NodeProcessManager(_config(mocker))
    manager.get_status = mocker.MagicMock(return_value=status)
    manager._init_state_tables = mocker.MagicMock()
    mock_popen = mocker.patch("fedbiomed.node.node_pm.subprocess.Popen")
    mock_logger = mocker.patch("fedbiomed.node.node_pm.logger")

    pid = manager.start(node_args={"gpu": False}, pid=123, actor={"source": "gui"})
    assert pid == 123

    mock_logger.warning.assert_called_once_with(
        "Node process 'pid=123' is already running. Ignoring start request."
    )
    mock_popen.assert_not_called()
    manager._init_state_tables.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [
        (NodeState.STOPPING.value),
        (NodeState.STOPPED.value),
    ],
)
def test_node_pm_07_stop_process_already_stopped(mocker, status):
    manager = NodeProcessManager(_config(mocker))
    manager.get_status = mocker.MagicMock(return_value=status)
    mock_logger = mocker.patch("fedbiomed.node.node_pm.logger")

    manager.stop(pid=123)

    mock_logger.warning.assert_called_once_with(
        "Node process 'pid=123' is already stopped. Ignoring stop request."
    )


@pytest.mark.parametrize(
    "state",
    [
        (NodeState.RUNNING.value),
        None,
    ],
)
def test_node_pm_08_get_status(mocker, state):
    manager = NodeProcessManager(_config(mocker))
    manager._init_state_tables = mocker.MagicMock()
    manager._state_table = mocker.MagicMock()
    manager._state_table.get_by_id.return_value = {"state": state}

    status = manager.get_status(12345)

    if state is not None:
        assert status == state
    else:
        assert status == NodeState.UNKNOWN.value
    manager._init_state_tables.assert_called_once_with()
    manager._state_table.get_by_id.assert_called_once_with(12345)


def test_node_pm_09_get_process_state_returns_stored_entry(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager._init_state_tables = mocker.MagicMock()
    manager.get_status = mocker.MagicMock(return_value=NodeState.RUNNING.value)

    stored = {
        "pid": 12345,
        "state": NodeState.RUNNING.value,
        "node_id": "node-1",
        "node_name": "Node 1",
        "action": "start",
        "reason": "process_started",
        "actor": {"source": "gui"},
        "updated_at": "utc-now",
        "started_at": "utc-now",
        "stopped_at": None,
        "exit_code": None,
    }

    manager._state_table = mocker.MagicMock()
    manager._state_table.get_by_id.return_value = stored

    state = manager.get_process_state(12345)

    manager._init_state_tables.assert_called_once_with()
    manager.get_status.assert_called_once_with(12345)
    manager._state_table.get_by_id.assert_called_once_with(12345)

    assert state["pid"] == 12345
    assert state["state"] == NodeState.RUNNING.value
    assert state["node_id"] == "node-1"
    assert state["node_name"] == "Node 1"
    assert state["actor"] == {"source": "gui"}
    assert state["managed_by_current_process"] is False

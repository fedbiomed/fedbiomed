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


@pytest.fixture
def _manager(mocker):
    state_table = mocker.MagicMock()
    history_table = mocker.MagicMock()

    mocker.patch(
        "fedbiomed.node.node_pm.NodeProcessStateTable",
        return_value=state_table,
    )
    mocker.patch(
        "fedbiomed.node.node_pm.NodeProcessStateHistoryTable",
        return_value=history_table,
    )

    manager = NodeProcessManager(_config(mocker))

    return manager


@pytest.mark.parametrize("background", [True, False])
def test_node_pm_01_start(mocker, _manager, background):
    config = _config(mocker)
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
        stdout=mocker.ANY,
        stderr=mocker.ANY,
    )

    mock_popen.assert_called_once()
    if background:
        manager._set_process_state.assert_called_once_with(
            pid=12345,
            state=NodeState.RUNNING,
            action="start",
            actor={"source": "gui"},
            reason="start_requested",
        )
    else:
        manager._wait.assert_called_once_with(process, actor={"source": "gui"})


@pytest.mark.parametrize(
    "status, should_set_stopped",
    [
        (NodeState.RUNNING, True),
        (NodeState.STOPPED, False),
    ],
)
def test_node_pm_02_wait(
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
def test_node_pm_03_stop(
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


def test_node_pm_04_restart(mocker, _manager):
    manager = _manager

    manager.stop = mocker.MagicMock()
    manager.start = mocker.MagicMock(return_value=67890)

    manager.restart(
        node_args={"gpu": False},
        background=False,
        actor={"source": "gui"},
    )

    manager.stop.assert_called_once_with(
        actor={"source": "gui"},
        reason="restart_requested",
    )

    manager.start.assert_called_once_with(
        node_args={"gpu": False},
        background=False,
        actor={"source": "gui"},
        reason="restart_requested",
    )


def test_node_pm_05_set_process_state(mocker, _manager):
    manager = _manager
    manager._node_id = "node_id1"
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
        reason="start_requested",
    )

    manager._state_table.update_or_insert_by_id.assert_called_once()
    assert manager._state_table.update_or_insert_by_id.call_args.args[0] == "node_id1"

    manager._history_table.insert.assert_called_once()

    state_entry = manager._state_table.update_or_insert_by_id.call_args.args[1]
    history_entry = manager._history_table.insert.call_args.args[0]

    assert state_entry["node_id"] == "node_id1"
    assert state_entry["pid"] == 1234
    assert NodeState(state_entry["state"]) == NodeState.RUNNING
    assert state_entry["started_at"] == "utc-now"
    assert state_entry["actor"] == {"source": "local"}
    assert history_entry == state_entry


def test_node_pm_06_start_process_already_started(mocker, _manager):
    manager = _manager
    manager.get_status = mocker.MagicMock(return_value=NodeState.RUNNING)

    mock_popen = mocker.patch("fedbiomed.node.node_pm.subprocess.Popen")
    mock_logger = mocker.patch("fedbiomed.node.node_pm.logger")

    manager.start(node_args={"gpu": False}, actor={"source": "gui"})

    mock_logger.warning.assert_called_once_with(
        "Node process is already running. Ignoring start request."
    )
    mock_popen.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [
        (NodeState.STOPPING),
        (NodeState.STOPPED),
    ],
)
def test_node_pm_07_stop_process_already_stopped(mocker, _manager, status):
    manager = _manager

    manager.get_status = mocker.MagicMock(return_value=status)
    mock_logger = mocker.patch("fedbiomed.node.node_pm.logger")

    manager.stop()

    mock_logger.warning.assert_called_once_with(
        "Node process is already stopped. Ignoring stop request."
    )


@pytest.mark.parametrize(
    "stored_state, pid_exists, expected_status",
    [
        (NodeState.RUNNING, True, NodeState.RUNNING),
        (NodeState.STOPPED, False, NodeState.STOPPED),
        (None, False, NodeState.UNKNOWN),
    ],
)
def test_node_pm_08_get_status(
    mocker, _manager, stored_state, pid_exists, expected_status
):
    manager = _manager
    state_table = manager._state_table

    if stored_state is None:
        state_table.get_by_id.return_value = None
    else:
        state_table.get_by_id.return_value = {
            "pid": 12345,
            "state": stored_state,
        }

    mocker.patch(
        "fedbiomed.node.node_pm.psutil.pid_exists",
        return_value=pid_exists,
    )

    status = manager.get_status()

    assert status == expected_status
    state_table.get_by_id.assert_called_with("node-1")


def test_node_pm_09_get_process_state_returns_stored_entry(mocker, _manager):
    manager = _manager

    manager.get_status = mocker.MagicMock(return_value=NodeState.RUNNING)
    manager._get_pid = mocker.MagicMock(return_value=12345)

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

    manager._state_table = mocker.MagicMock()
    manager._state_table.get_by_id.return_value = stored

    state = manager._get_process_state()

    manager._get_pid.assert_called_once()
    manager.get_status.assert_called_once()
    manager._state_table.get_by_id.assert_called_with("node-1")

    assert state.pid == 12345
    assert state.state == NodeState.RUNNING
    assert state.node_id == "node-1"
    assert state.node_name == "Node 1"
    assert state.actor == {"source": "gui"}

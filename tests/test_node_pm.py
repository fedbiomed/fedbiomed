import sys
import types
from unittest.mock import MagicMock, call, patch

from fedbiomed.node.node_pm import NodeProcessManager, NodeState


def _config(node_id="node-1", node_name="Node 1", db_name="node_db.json"):
    config = MagicMock()
    config.root = "/tmp/node-root"

    def _get(section, key):
        values = {
            ("default", "id"): node_id,
            ("default", "name"): node_name,
            ("default", "db"): db_name,
        }
        return values[(section, key)]

    config.get.side_effect = _get
    return config


def test_node_pm_01_build_actor_filters_unknown_fields():
    with patch("fedbiomed.node.node_pm.getpass.getuser", return_value="local-user"):
        actor = NodeProcessManager._build_actor(
            {
                "source": "api",
                "email": "user@example.com",
                "role": "admin",
                "ignored": "value",
            }
        )

    assert actor == {
        "source": "api",
        "local_username": "local-user",
        "email": "user@example.com",
        "role": "admin",
    }


def test_node_pm_02_set_process_state_writes_current_and_history_entries():
    manager = NodeProcessManager()
    manager._node_id = "node-1"
    manager._node_name = "Node 1"
    manager._process = MagicMock(pid=4321)
    manager._state_table = MagicMock()
    manager._history_table = MagicMock()
    manager._state_table.get_by_id.return_value = {
        "started_at": None,
        "stopped_at": None,
    }

    with patch("fedbiomed.node.node_pm._utc_now", return_value="2026-04-23T12:00:00Z"):
        with patch.object(
            NodeProcessManager, "_build_actor", return_value={"source": "local"}
        ):
            manager._set_process_state(
                state=NodeState.RUNNING,
                action="start",
                actor={"source": "local"},
                reason="process_started",
            )

    manager._state_table.update_or_insert_by_id.assert_called_once()
    manager._history_table.insert.assert_called_once()

    state_entry = manager._state_table.update_or_insert_by_id.call_args.args[1]
    history_entry = manager._history_table.insert.call_args.args[0]

    assert state_entry["node_id"] == "node-1"
    assert state_entry["pid"] == 4321
    assert state_entry["state"] == NodeState.RUNNING.value
    assert state_entry["started_at"] == "2026-04-23T12:00:00Z"
    assert state_entry["stopped_at"] is None
    assert state_entry["actor"] == {"source": "local"}
    assert history_entry == state_entry


def test_node_pm_03_start_spawns_process_and_sets_state_transitions():
    manager = NodeProcessManager()
    manager._init_state_tables = MagicMock()
    manager._set_process_state = MagicMock()
    config = _config()

    fake_cli = types.ModuleType("fedbiomed.node.cli")
    fake_cli.start_node = MagicMock()

    process = MagicMock()
    process.pid = 321

    with patch.dict(sys.modules, {"fedbiomed.node.cli": fake_cli}):
        with patch(
            "fedbiomed.node.node_pm.multiprocessing.Process", return_value=process
        ) as mock_process:
            manager.start(config, {"gpu": False}, actor={"source": "api"})

    manager._init_state_tables.assert_called_once_with(config)
    assert manager._set_process_state.call_args_list == [
        call(
            state=NodeState.STARTING,
            action="start",
            actor={"source": "api"},
            reason="start_requested",
        ),
        call(
            state=NodeState.RUNNING,
            action="start",
            actor={"source": "api"},
            reason="process_started",
        ),
    ]
    mock_process.assert_called_once_with(
        target=fake_cli.start_node,
        name="node-node-1",
        args=(config, {"gpu": False}),
    )
    process.start.assert_called_once_with()
    assert process.daemon is True
    assert manager.process is process


def test_node_pm_04_start_ignores_duplicate_running_process_for_same_node():
    manager = NodeProcessManager()
    manager._process = MagicMock()
    manager._process.is_alive.return_value = True
    manager._process.pid = 654
    manager._node_id = "node-1"
    manager._set_process_state = MagicMock()
    config = _config()

    with patch("fedbiomed.node.node_pm.logger") as mock_logger:
        manager.start(config, {"gpu": False}, actor={"source": "api"})

    manager._set_process_state.assert_called_once_with(
        state=NodeState.RUNNING,
        action="start",
        actor={"source": "api"},
        reason="already_running",
    )
    mock_logger.warning.assert_called_once_with(
        "Node 'node-1' is already running (pid=654). Ignoring start request."
    )


def test_node_pm_05_stop_warns_when_no_process_is_running():
    manager = NodeProcessManager()

    with patch("fedbiomed.node.node_pm.logger") as mock_logger:
        manager.stop()

    mock_logger.warning.assert_called_once_with("No node process is running.")


@patch("fedbiomed.node.node_pm.time.sleep")
def test_node_pm_06_stop_terminates_process_and_cleans_up(mock_sleep):
    manager = NodeProcessManager()
    process = MagicMock()
    process.pid = 777
    process.exitcode = 15
    process.is_alive.side_effect = [True, False, False]
    manager._process = process
    manager._set_process_state = MagicMock()
    manager._cleanup_process = MagicMock()

    manager.stop(actor={"source": "api"}, reason="stop_requested")

    process.terminate.assert_called_once_with()
    process.kill.assert_not_called()
    mock_sleep.assert_called_once_with(0.5)
    assert manager._set_process_state.call_args_list == [
        call(
            state=NodeState.STOPPING,
            action="stop",
            actor={"source": "api"},
            reason="stop_requested",
        ),
        call(
            state=NodeState.STOPPED,
            action="stop",
            actor={"source": "api"},
            reason="stop_requested",
            exit_code=15,
        ),
    ]
    manager._cleanup_process.assert_called_once_with()


@patch("fedbiomed.node.node_pm.time.sleep")
def test_node_pm_07_stop_logs_error_when_process_survives_kill(mock_sleep):
    manager = NodeProcessManager()
    process = MagicMock()
    process.pid = 888
    process.exitcode = -9
    process.is_alive.side_effect = [True, True, True]
    manager._process = process
    manager._set_process_state = MagicMock()
    manager._cleanup_process = MagicMock()

    with patch("fedbiomed.node.node_pm.logger") as mock_logger:
        manager.stop(reason="test_stop")

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert mock_sleep.call_args_list == [call(0.5), call(0.5)]
    manager._cleanup_process.assert_not_called()
    mock_logger.error.assert_called_once_with(
        "Failed to kill node process (pid=888). Process leak - manual intervention required."
    )


def test_node_pm_08_restart_calls_stop_then_start():
    manager = NodeProcessManager()
    manager.stop = MagicMock()
    manager.start = MagicMock()
    config = _config()
    node_args = {"gpu": True}
    actor = {"source": "api"}

    manager.restart(config, node_args, actor=actor)

    manager.stop.assert_called_once_with(actor=actor, reason="restart_requested")
    manager.start.assert_called_once_with(config, node_args, actor=actor)


def test_node_pm_09_get_status_reflects_process_liveness():
    manager = NodeProcessManager()
    assert manager.get_status() == NodeState.STOPPED

    manager._process = MagicMock()
    manager._process.is_alive.return_value = True
    assert manager.get_status() == NodeState.RUNNING

    manager._process.is_alive.return_value = False
    assert manager.get_status() == NodeState.STOPPED

from fedbiomed.node.node_pm import NodeProcessManager, NodeState, _start_node_process


def _config(mocker, node_id="node-1", node_name="Node 1", db_name="node_db.json"):
    config = mocker.MagicMock()
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


def test_node_pm_01_build_actor_filters_unknown_fields(mocker):
    mocker.patch("fedbiomed.node.node_pm.getpass.getuser", return_value="local-user")
    actor = NodeProcessManager._build_actor(
        {
            "source": "gui",
            "email": "user@example.com",
            "role": "admin",
            "ignored": "value",
        }
    )

    assert actor == {
        "source": "gui",
        "local_username": "local-user",
        "email": "user@example.com",
        "role": "admin",
    }


def test_node_pm_02_set_process_state_writes_current_and_history_entries(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager._node_id = "node-1"
    manager._node_name = "Node 1"
    manager._process = mocker.MagicMock(pid=4321)
    manager._state_table = mocker.MagicMock()
    manager._history_table = mocker.MagicMock()
    manager._state_table.get_by_id.return_value = {
        "started_at": None,
        "stopped_at": None,
    }

    mocker.patch("fedbiomed.node.node_pm._utc_now", return_value="2026-04-23T12:00:00Z")
    mocker.patch.object(
        NodeProcessManager, "_build_actor", return_value={"source": "local"}
    )
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


def test_node_pm_03_start_spawns_process_and_sets_state_transitions(mocker):
    config = _config(mocker)
    manager = NodeProcessManager(config)
    manager._init_state_tables = mocker.MagicMock()
    manager._set_process_state = mocker.MagicMock()

    process = mocker.MagicMock()
    process.pid = 321

    mock_process = mocker.patch(
        "fedbiomed.node.node_pm.multiprocessing.Process", return_value=process
    )
    manager.start({"gpu": False}, actor={"source": "gui"})

    manager._init_state_tables.assert_called_once_with()

    assert manager._set_process_state.call_args_list[0].kwargs == {
        "state": NodeState.STARTING,
        "action": "start",
        "actor": {"source": "gui"},
        "reason": "start_requested",
    }
    assert manager._set_process_state.call_args_list[1].kwargs == {
        "state": NodeState.RUNNING,
        "action": "start",
        "actor": {"source": "gui"},
        "reason": "process_started",
    }

    mock_process.assert_called_once_with(
        target=_start_node_process,
        name="node-node-1",
        args=(config, {"gpu": False}),
    )
    process.start.assert_called_once_with()
    assert process.daemon is True
    assert manager.process is process


def test_node_pm_04_start_ignores_duplicate_running_process_for_same_node(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager._process = mocker.MagicMock()
    manager._process.is_alive.return_value = True
    manager._process.pid = 654
    manager._node_id = "node-1"
    manager._set_process_state = mocker.MagicMock()

    mock_logger = mocker.patch("fedbiomed.node.node_pm.logger")
    manager.start({"gpu": False}, actor={"source": "gui"})

    manager._set_process_state.assert_called_once_with(
        state=NodeState.RUNNING,
        action="start",
        actor={"source": "gui"},
        reason="already_running",
    )
    mock_logger.warning.assert_called_once_with(
        "Node 'node-1' is already running (pid=654). Ignoring start request."
    )


def test_node_pm_05_stop_warns_when_no_process_is_running(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager._init_state_tables = mocker.MagicMock()
    manager._set_process_state = mocker.MagicMock()

    mock_logger = mocker.patch("fedbiomed.node.node_pm.logger")
    manager.stop(actor={"source": "gui"})

    manager._init_state_tables.assert_called_once_with()
    manager._set_process_state.assert_called_once_with(
        state=NodeState.STOPPED,
        action="stop",
        actor={"source": "gui"},
        reason="no_process_running",
    )
    mock_logger.warning.assert_called_once_with("No node process is running.")


def test_node_pm_06_stop_terminates_process_and_cleans_up(mocker):
    mock_sleep = mocker.patch("fedbiomed.node.node_pm.time.sleep")
    manager = NodeProcessManager(_config(mocker))
    process = mocker.MagicMock()
    process.pid = 777
    process.exitcode = 15
    process.is_alive.side_effect = [True, False, False]
    manager._process = process
    manager._set_process_state = mocker.MagicMock()
    manager._cleanup_process = mocker.MagicMock()

    manager.stop(actor={"source": "gui"}, reason="stop_requested")

    process.terminate.assert_called_once_with()
    process.kill.assert_not_called()

    assert mock_sleep.call_args_list[0].args == (0.5,)

    assert manager._set_process_state.call_args_list[0].kwargs == {
        "state": NodeState.STOPPING,
        "action": "stop",
        "actor": {"source": "gui"},
        "reason": "stop_requested",
    }
    assert manager._set_process_state.call_args_list[1].kwargs == {
        "state": NodeState.STOPPED,
        "action": "stop",
        "actor": {"source": "gui"},
        "reason": "stop_requested",
        "exit_code": 15,
    }

    manager._cleanup_process.assert_called_once_with()


def test_node_pm_07_stop_logs_error_when_process_survives_kill(mocker):
    mock_sleep = mocker.patch("fedbiomed.node.node_pm.time.sleep")
    manager = NodeProcessManager(_config(mocker))
    process = mocker.MagicMock()
    process.pid = 888
    process.exitcode = -9
    process.is_alive.side_effect = [True, True, True]
    manager._process = process
    manager._set_process_state = mocker.MagicMock()
    manager._cleanup_process = mocker.MagicMock()

    mock_logger = mocker.patch("fedbiomed.node.node_pm.logger")
    manager.stop(reason="test_stop")

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()

    assert mock_sleep.call_args_list[0].args == (0.5,)
    assert mock_sleep.call_args_list[1].args == (0.5,)

    manager._cleanup_process.assert_not_called()
    mock_logger.error.assert_called_once_with(
        "Failed to kill node process (pid=888). Process leak - manual intervention required."
    )


def test_node_pm_08_restart_calls_stop_then_start(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager.stop = mocker.MagicMock()
    manager.start = mocker.MagicMock()
    node_args = {"gpu": True}
    actor = {"source": "gui"}

    manager.restart(node_args, actor=actor)

    manager.stop.assert_called_once_with(actor=actor, reason="restart_requested")
    manager.start.assert_called_once_with(node_args, actor=actor)


def test_node_pm_09_get_status_reflects_process_liveness(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager._state_table = mocker.MagicMock()
    manager._history_table = mocker.MagicMock()
    manager._state_table.get_by_id.return_value = None
    assert manager.get_status() == NodeState.STOPPED

    manager._process = mocker.MagicMock()
    manager._process.is_alive.return_value = True
    assert manager.get_status() == NodeState.RUNNING

    manager._process.is_alive.return_value = False
    manager._state_table.get_by_id.return_value = None
    assert manager.get_status() == NodeState.STOPPED


def test_node_pm_10_get_status_reads_persisted_state(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager._state_table = mocker.MagicMock()
    manager._history_table = mocker.MagicMock()
    manager._state_table.get_by_id.return_value = {
        "node_id": "node-1",
        "state": NodeState.RUNNING.value,
    }

    assert manager.get_status() == NodeState.RUNNING


def test_node_pm_11_get_process_state_returns_persisted_metadata(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager._state_table = mocker.MagicMock()
    manager._history_table = mocker.MagicMock()
    manager._state_table.get_by_id.return_value = {
        "node_id": "node-1",
        "node_name": "Node 1",
        "state": NodeState.RUNNING.value,
        "pid": 1234,
        "action": "start",
        "reason": "process_started",
    }
    manager._process = mocker.MagicMock(pid=1234)
    manager._process.is_alive.return_value = True

    state = manager.get_process_state()

    assert state["node_id"] == "node-1"
    assert state["state"] == NodeState.RUNNING.value
    assert state["pid"] == 1234
    assert state["action"] == "start"
    assert state["managed_by_current_process"] is True


def test_node_pm_12_get_process_state_returns_default_when_no_db_entry(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager._state_table = mocker.MagicMock()
    manager._history_table = mocker.MagicMock()
    manager._state_table.get_by_id.return_value = None

    state = manager.get_process_state()

    assert state["node_id"] == "node-1"
    assert state["node_name"] == "Node 1"
    assert state["state"] == NodeState.STOPPED.value
    assert state["pid"] is None
    assert state["managed_by_current_process"] is False


def test_node_pm_13_stop_persists_dead_managed_process(mocker):
    manager = NodeProcessManager(_config(mocker))
    manager._process = mocker.MagicMock(pid=1234, exitcode=7)
    manager._process.is_alive.return_value = False
    manager._init_state_tables = mocker.MagicMock()
    manager._set_process_state = mocker.MagicMock()
    manager._cleanup_process = mocker.MagicMock()
    mock_logger = mocker.patch("fedbiomed.node.node_pm.logger")

    manager.stop(actor={"source": "gui"})

    manager._init_state_tables.assert_called_once_with()
    manager._set_process_state.assert_called_once_with(
        state=NodeState.STOPPED,
        action="stop",
        actor={"source": "gui"},
        reason="process_exited",
        exit_code=7,
    )
    manager._cleanup_process.assert_called_once_with()
    mock_logger.warning.assert_called_once_with("No node process is running.")

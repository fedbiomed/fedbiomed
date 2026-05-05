import pytest

from fedbiomed.restful import node_bootstrap


@pytest.fixture
def managed_node_process_manager(mocker):
    return mocker.MagicMock()


@pytest.fixture
def node_config(mocker):
    return mocker.MagicMock()


def test_gui_node_bootstrap_01_load_node_args_from_env_returns_defaults_when_missing(
    monkeypatch,
):
    monkeypatch.delenv("FBM_NODE_START_ARGS", raising=False)

    assert node_bootstrap.load_node_args_from_env() == {
        "gpu": False,
        "gpu_num": None,
        "gpu_only": False,
        "debug": False,
    }


def test_gui_node_bootstrap_02_load_node_args_from_env_warns_on_invalid_json(
    monkeypatch,
    mocker,
):
    mock_logger = mocker.MagicMock()
    monkeypatch.setattr(node_bootstrap, "logger", mock_logger)
    monkeypatch.setenv("FBM_NODE_START_ARGS", "{not-valid-json")

    assert node_bootstrap.load_node_args_from_env() == {
        "gpu": False,
        "gpu_num": None,
        "gpu_only": False,
        "debug": False,
    }

    mock_logger.warning.assert_called_once()


def test_gui_node_bootstrap_03_start_node_for_gui_uses_env_args(
    monkeypatch,
    mocker,
    node_config,
):
    mock_get_manager = mocker.patch.object(node_bootstrap, "_get_node_process_manager")
    monkeypatch.delenv("FBM_START_NODE_WITH_RESTFUL", raising=False)
    monkeypatch.setenv(
        "FBM_NODE_START_ARGS",
        '{"gpu": true, "gpu_num": 2, "gpu_only": false, "debug": true}',
    )

    node_bootstrap.start_node_for_gui(node_config)

    mock_get_manager.return_value.start.assert_called_once_with(
        node_config,
        {"gpu": True, "gpu_num": 2, "gpu_only": False, "debug": True},
        actor={"source": "gui"},
    )


def test_gui_node_bootstrap_04_stop_node_for_gui_stops_managed_node(
    monkeypatch,
    mocker,
):
    mock_get_manager = mocker.patch.object(node_bootstrap, "_get_node_process_manager")
    monkeypatch.delenv("FBM_START_NODE_WITH_RESTFUL", raising=False)

    node_bootstrap.stop_node_for_gui()

    mock_get_manager.return_value.stop.assert_called_once_with(
        actor={"source": "gui"}, reason="gui_stopped"
    )


@pytest.mark.parametrize("value", ["false", "False", "0", "no"])
def test_gui_node_bootstrap_start_node_for_gui_skips_when_disabled(
    monkeypatch,
    mocker,
    node_config,
    value,
):
    mock_get_manager = mocker.patch.object(node_bootstrap, "_get_node_process_manager")
    monkeypatch.setenv("FBM_START_NODE_WITH_RESTFUL", value)

    node_bootstrap.start_node_for_gui(node_config)

    mock_get_manager.assert_not_called()

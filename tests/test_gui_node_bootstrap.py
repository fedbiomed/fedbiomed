import os
from unittest.mock import MagicMock, patch

from fedbiomed_gui.server import node_bootstrap


class TestGuiNodeBootstrap:
    def test_01_load_node_args_from_env_returns_defaults_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            assert node_bootstrap.load_node_args_from_env() == {
                "gpu": False,
                "gpu_num": None,
                "gpu_only": False,
                "debug": False,
            }

    @patch("fedbiomed_gui.server.node_bootstrap.logger")
    def test_02_load_node_args_from_env_warns_on_invalid_json(self, mock_logger):
        with patch.dict(
            os.environ, {"FBM_NODE_START_ARGS": "{not-valid-json"}, clear=True
        ):
            assert node_bootstrap.load_node_args_from_env() == {
                "gpu": False,
                "gpu_num": None,
                "gpu_only": False,
                "debug": False,
            }

        mock_logger.warning.assert_called_once()

    @patch("fedbiomed_gui.server.node_bootstrap._get_node_process_manager")
    def test_03_start_node_for_gui_uses_env_args(self, mock_manager):
        node_config = MagicMock()
        with patch.dict(
            os.environ,
            {
                "FBM_NODE_START_ARGS": (
                    '{"gpu": true, "gpu_num": 2, "gpu_only": false, "debug": true}'
                ),
            },
            clear=True,
        ):
            node_bootstrap.start_node_for_gui(node_config)

        mock_manager.return_value.start.assert_called_once_with(
            node_config,
            {"gpu": True, "gpu_num": 2, "gpu_only": False, "debug": True},
            actor={"source": "gui"},
        )

    @patch("fedbiomed_gui.server.node_bootstrap._get_node_process_manager")
    def test_04_stop_node_for_gui_stops_managed_node(self, mock_manager):
        with patch.dict(os.environ, {}, clear=True):
            node_bootstrap.stop_node_for_gui()

        mock_manager.return_value.stop.assert_called_once_with(
            actor={"source": "gui"}, reason="gui_stopped"
        )

    @patch("fedbiomed_gui.server.node_bootstrap._get_node_process_manager")
    def test_05_start_node_for_gui_skips_when_disabled(self, mock_manager):
        node_config = MagicMock()
        with patch.dict(os.environ, {"FBM_START_NODE_WITH_GUI": "false"}, clear=True):
            node_bootstrap.start_node_for_gui(node_config)

        mock_manager.return_value.start.assert_not_called()

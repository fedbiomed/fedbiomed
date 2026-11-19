"""Check that core server commands only parse and forward arguments."""

import argparse
import unittest
from unittest.mock import MagicMock, patch

from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.cli import APIControl, GUIControl


class TestServerControls(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()

    def _start_parser(self, command):
        return (
            self.subparsers.choices[command]
            ._subparsers._group_actions[0]
            .choices["start"]
        )

    def test_api_and_gui_share_start_options(self):
        APIControl(self.subparsers).initialize()
        GUIControl(self.subparsers).initialize()
        for command in ("api", "gui"):
            options = self._start_parser(command)._option_string_actions
            for option in (
                "--port",
                "--host",
                "--cert-file",
                "--key-file",
                "--debug",
                "--development",
                "--data-folder",
            ):
                self.assertIn(option, options)
        self.assertIn("--recreate", self._start_parser("gui")._option_string_actions)
        self.assertNotIn("--recreate", self._start_parser("api")._option_string_actions)

    @patch("fedbiomed.node.cli.importlib.import_module")
    def test_commands_forward_to_optional_packages(self, import_module):
        for control_class, module in (
            (APIControl, "fedbiomed_node_api.cli"),
            (GUIControl, "fedbiomed_gui.cli"),
        ):
            package = MagicMock()
            import_module.return_value = package
            args = argparse.Namespace(path="node")
            control_class(self.subparsers).forward(args, [])
            import_module.assert_called_with(module)
            package.start_server.assert_called_once_with(args)
            import_module.reset_mock()

    @patch("fedbiomed.node.cli.importlib.import_module")
    def test_missing_optional_package_has_install_hint(self, import_module):
        error = ModuleNotFoundError("No module named 'fedbiomed_node_api'")
        error.name = "fedbiomed_node_api"
        import_module.side_effect = error
        with self.assertRaisesRegex(FedbiomedError, r"fedbiomed\[node-api\]"):
            APIControl(self.subparsers).forward(argparse.Namespace(path="node"), [])


if __name__ == "__main__":
    unittest.main()

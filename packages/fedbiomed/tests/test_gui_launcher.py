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

    @patch("fedbiomed.node.cli.importlib.import_module")
    def test_commands_forward_to_optional_packages(self, import_module):
        for control_class, module in (
            (APIControl, "fedbiomed_node_api.cli"),
            (GUIControl, "fedbiomed_gui.cli"),
        ):
            package = MagicMock()
            import_module.return_value = package
            args = argparse.Namespace(path="node")
            control_class(self.subparsers).forward(args)
            import_module.assert_called_with(module)
            package.start_server.assert_called_once_with(args)
            import_module.reset_mock()

    @patch("fedbiomed.node.cli.importlib.import_module")
    def test_missing_optional_package_has_install_hint(self, import_module):
        missing = ModuleNotFoundError("No module named 'fedbiomed_node_api'")
        missing.name = "fedbiomed_node_api"
        import_module.side_effect = missing
        with (
            self.assertLogs("fedbiomed", "CRITICAL") as logs,
            self.assertRaises(SystemExit),
        ):
            APIControl(self.subparsers).forward(argparse.Namespace(path="node"))
        self.assertIn("fedbiomed[node-api]", logs.output[0])

    @patch("fedbiomed.node.cli.importlib.import_module")
    def test_server_errors_are_reported_without_traceback(self, import_module):
        import_module.return_value.start_server.side_effect = FedbiomedError(
            "not built"
        )
        with (
            self.assertLogs("fedbiomed", "CRITICAL") as logs,
            self.assertRaises(SystemExit),
        ):
            GUIControl(self.subparsers).forward(argparse.Namespace(path="node"))
        self.assertIn("not built", logs.output[0])

    @patch("fedbiomed.node.cli.importlib.import_module")
    def test_other_missing_modules_are_not_hidden(self, import_module):
        missing = ModuleNotFoundError("No module named 'flask'")
        missing.name = "flask"
        import_module.side_effect = missing
        with self.assertRaises(ModuleNotFoundError):
            APIControl(self.subparsers).forward(argparse.Namespace(path="node"))


if __name__ == "__main__":
    unittest.main()

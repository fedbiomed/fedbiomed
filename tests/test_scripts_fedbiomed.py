import importlib.machinery
import importlib.util
import argparse
import unittest
from unittest.mock import patch


# Load `scripts/fedbiomed` as a module
scripts_fedbiomed_loader = importlib.machinery.SourceFileLoader('scripts_fedbiomed', '../scripts/fedbiomed')
scripts_fedbiomed_spec = importlib.util.spec_from_loader(scripts_fedbiomed_loader.name, scripts_fedbiomed_loader)
scripts_fedbiomed = importlib.util.module_from_spec(scripts_fedbiomed_spec)
scripts_fedbiomed_loader.exec_module(scripts_fedbiomed)


class TestConfigurationParser(unittest.TestCase):

    def setUp(self):

        self.main_parser = argparse.ArgumentParser()
        self.parser = self.main_parser.add_subparsers()
        self.conf_parser = scripts_fedbiomed.ConfigurationParser(subparser=self.parser)
        self.conf_parser.initialize()
        pass

    def tearDown(self):
        pass

    def test_01_configuration_parser_initialize(self):
        """Tests argument initialization"""
        self.assertTrue("configuration" in self.conf_parser._subparser.choices)
        self.assertTrue(
            "create"
            in self.conf_parser._subparser.choices["configuration"]
            ._subparsers._group_actions[0]
            .choices
        )
        self.assertEqual(
            self.conf_parser._subparser.choices["configuration"]
            ._subparsers._group_actions[0]
            .choices["create"]
            ._defaults["func"]
            .__func__.__name__,
            "create",
        )

    @patch("builtins.print")
    @patch("builtins.open")
    @patch("fedbiomed.node.config.NodeConfig")
    @patch("fedbiomed.researcher.config.ResearcherConfig")
    def test_02_configuration_parser_create(
        self,
        rconfig,
        nconfig,
        mock_open,
        mock_print,
    ):
        args = self.main_parser.parse_args(
            ["configuration", "create", "--component", "NODE", "-uc"]
        )
        self.conf_parser.create(args)
        nconfig.return_value.generate.assert_called_once()

        mock_print.reset_mock()
        args = self.main_parser.parse_args(
            ["configuration", "create", "--component", "RESEARCHER", "-uc"]
        )
        self.conf_parser.create(args)
        rconfig.return_value.generate.assert_called_once()

    @patch("builtins.print")
    @patch("builtins.open")
    @patch("fedbiomed.node.config.NodeConfig")
    @patch("fedbiomed.researcher.config.ResearcherConfig")
    def test_03_configuration_parser_refresh(
        self,
        rconfig,
        nconfig,
        mock_open,
        mock_print,
    ):

        args = self.main_parser.parse_args(
            ["configuration", "refresh", "--component", "NODE", "-n", "config"]
        )
        self.conf_parser.refresh(args)
        nconfig.return_value.refresh.assert_called()


if __name__ == "__main__":
    unittest.main()

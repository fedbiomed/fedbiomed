import os
import sys
import argparse
import tempfile
import unittest

from fedbiomed.cli import ComponentParser


class TestComponentParser(unittest.TestCase):

    def setUp(self):

        self.main_parser = argparse.ArgumentParser()
        self.parser = self.main_parser.add_subparsers()
        self.conf_parser = ComponentParser(subparser=self.parser)
        self.conf_parser.initialize()

        self.tem = tempfile.TemporaryDirectory()
        self.initial_dir = os.getcwd()
        os.chdir(self.tem.name)

    def tearDown(self):
        os.chdir(self.initial_dir)
        # forces the removal of config files for Nodes and Researchers
        if 'fedbiomed.researcher.config' in sys.modules:
            sys.modules.pop('fedbiomed.researcher.config')

        if 'fedbiomed.node.config' in sys.modules:
            sys.modules.pop('fedbiomed.node.config')
        self.tem.cleanup()


    def test_01_component_parser_initialize(self):
        """Tests argument initialization"""
        self.assertTrue("component" in self.conf_parser._subparser.choices)
        self.assertTrue(
            "create"
            in self.conf_parser._subparser.choices["component"]
            ._subparsers._group_actions[0]
            .choices
        )
        self.assertEqual(
            self.conf_parser._subparser.choices["component"]
            ._subparsers._group_actions[0]
            .choices["create"]
            ._defaults["func"]
            .__func__.__name__,
            "create",
        )

    def test_02_component_parser_create_success(self):
        """Tests component creation parser successful"""

        self.temp_component_node = tempfile.TemporaryDirectory()
        self.temp_component_researcher = tempfile.TemporaryDirectory()
        args_list_set = [
            ["component", "create", "--path", self.temp_component_node.name, "--component", "NODE", "-eo"],
            ["component", "create", "--path", self.temp_component_researcher.name, "--component", "RESEARCHER", "-eo"],
            ["component", "create", "--component", "NODE"],
            ["component", "create", "--component", "NODE", "-eo"],
            ["component", "create", "--component", "RESEARCHER"],
            ["component", "create", "--component", "RESEARCHER", "-eo"],
        ]

        for args_list in args_list_set:
            args = self.main_parser.parse_args(args_list)
            self.conf_parser.create(args)

            if 'fedbiomed.researcher.config' in sys.modules:
                sys.modules.pop('fedbiomed.researcher.config')

        self.tem.cleanup()

    def test_03_component_parser_create_fail(self):
        """Tests component creation parser fails"""

        args_list_preset = [
            ["component", "create", "--component", "NODE"],
            ["component", "create", "--component", "RESEARCHER"],
        ]
        for args_list in args_list_preset:
            args = self.main_parser.parse_args(args_list)
            self.conf_parser.create(args)

        # 1. fails at parse
        args_list_set = [
            ["component", "create", "-p", "p1", "-p", "p2"],
        ]

        for args_list in args_list_set:
            with self.assertRaises(SystemExit):
                self.main_parser.parse_args(args_list)

        # 2. fails at create
        args_list_set = [
            ["component", "create", "--component"],
            ["component", "create", "--component", "node"],
            ["component", "create", "--component", "researcher"],
        ]

        for args_list in args_list_set:
            args = self.main_parser.parse_args(args_list)
            with self.assertRaises(SystemExit):
                self.conf_parser.create(args)

        # 3. bad component type
        with self.assertRaises(SystemExit):
            self.conf_parser._get_component_instance('any_path', 'bad_component_type')

        self.tem.cleanup()

        sys.modules.pop('fedbiomed.researcher.config')


if __name__ == "__main__":
    unittest.main()

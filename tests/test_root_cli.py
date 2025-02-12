import argparse
import tempfile
import unittest
from unittest.mock import patch

from fedbiomed.cli import ComponentParser


class TestComponentParser(unittest.TestCase):

    def setUp(self):

        self.main_parser = argparse.ArgumentParser()
        self.parser = self.main_parser.add_subparsers()
        self.conf_parser = ComponentParser(subparser=self.parser)
        self.conf_parser.initialize()

        self.tem = tempfile.TemporaryDirectory()



    def tearDown(self):
        self.tem.cleanup()
        pass

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

    def test_02_component_parser_create(
        self,
    ):
        args = self.main_parser.parse_args(
            ["component", "create", "--path", self.tem.name, "--component", "NODE", "-eo"]
        )
        self.conf_parser.create(args)
        self.tem.cleanup()


        args = self.main_parser.parse_args(
            ["component", "create", "--path", self.tem.name, "--component", "RESEARCHER", "-eo"]
        )
        self.conf_parser.create(args)


if __name__ == "__main__":
    unittest.main()

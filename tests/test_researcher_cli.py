import unittest
import argparse
from unittest.mock import patch
from testsupport.fake_researcher_environ import ResearcherEnviron
from testsupport.base_case import ResearcherTestCase
from fedbiomed.researcher.cli import ResearcherCLI, ResearcherControl
from fedbiomed.common.cli import CommonCLI


class TestResearcherControl(unittest.TestCase):
    """Tests researcher control unit argument parser"""
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers()
        self.control = ResearcherControl(self.subparsers)

    def tearDown(self):
        pass

    def test_01_researcher_control_initialize(self):

        self.control.initialize()
        self.assertTrue("start" in self.subparsers.choices)
        self.assertTrue("--directory" in self.subparsers.choices["start"]._option_string_actions)

    @patch("fedbiomed.researcher.cli.subprocess.Popen")
    def test_02_researcher_control_start(self, sub_process_p_open):

        self.control.initialize()
        args = self.parser.parse_args(["start", "--directory", "./"])
        self.control.start(args)

        sub_process_p_open.assert_called_once()

        sub_process_p_open.return_value.wait.side_effect = KeyboardInterrupt
        sub_process_p_open.return_value.terminate.side_effect = Exception

        with self.assertRaises(KeyboardInterrupt):
            self.control.start(args)

class TestResearcherCLI(ResearcherTestCase):

    def setUp(self) -> None:
        self.cli = ResearcherCLI()

    def tearDown(self) -> None:
        pass

    def test_01_researcher_cli_init(self):
        self.assertEqual(ResearcherCLI.__base__, CommonCLI, 'ResearcherCLI should inherit from CommonCLI')

    @patch('builtins.print')
    def test_02_researcher_initialize(self, mock_print):

        # Tests certificate parser options
        choices = self.cli._subparsers.choices["certificate"]._subparsers._group_actions[0].choices
        self.assertTrue('register' in choices)
        self.assertTrue('list' in choices)
        self.assertTrue('delete' in choices)
        self.assertTrue('generate' in choices)
        self.assertTrue('registration-instructions' in choices)

        self.assertEqual("_register_certificate", choices["register"]._defaults["func"].__func__.__name__)
        self.assertEqual("_generate_certificate", choices["generate"]._defaults["func"].__func__.__name__)
        self.assertEqual("_delete_certificate", choices["delete"]._defaults["func"].__func__.__name__)
        self.assertEqual("_list_certificates", choices["list"]._defaults["func"].__func__.__name__)
        self.assertEqual("_prepare_certificate_for_registration", choices["registration-instructions"].
                         _defaults["func"].__func__.__name__)

        register_options = choices["register"]._positionals._option_string_actions
        self.assertTrue("--party-id" in register_options)
        self.assertTrue("--public-key" in register_options)
        self.assertTrue("--ip" in register_options)
        self.assertTrue("--port" in register_options)

        generate_options = choices["generate"]._positionals._option_string_actions
        self.assertTrue("--path" in generate_options)
        self.assertTrue("--force" in generate_options)


if __name__ == "__main__":
    unittest.main()

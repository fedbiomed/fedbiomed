import unittest
from unittest.mock import patch
from testsupport.fake_researcher_environ import ResearcherEnviron
from testsupport.base_case import ResearcherTestCase
from fedbiomed.researcher.cli import ResearcherCLI
from fedbiomed.common.cli import CommonCLI


class TestResearcherCLI(ResearcherTestCase):

    def setUp(self) -> None:

        self.cli = ResearcherCLI()
        self.assertIsInstance(self.cli._environ, ResearcherEnviron, 'Environ is not set properly')

    def tearDown(self) -> None:
        pass

    def test_01_researcher_cli_init(self):
        self.assertEqual(ResearcherCLI.__base__, CommonCLI, 'ResearcherCLI should inherit from CommonCLI')

    @patch('builtins.print')
    @patch('fedbiomed.common.cli.CommonCLI.parse_args')
    def test_02_researcher_cli_launch_cli(self,
                                          mock_parse_args,
                                          mock_print
                                          ):
        self.cli.launch_cli()

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

        # Test configuration parser option
        self.assertTrue('configuration' in self.cli._subparsers.choices)
        self.assertTrue('create' in self.cli._subparsers.choices["configuration"]._subparsers._group_actions[0].choices)

        self.assertEqual(self.cli._subparsers.choices["configuration"].
                         _subparsers._group_actions[0].choices["create"]._defaults["func"].__func__.__name__,
                         '_create_component_configuration')


if __name__ == "__main__":
    unittest.main()


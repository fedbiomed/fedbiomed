import argparse
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fedbiomed.common.cli import CommonCLI, ComponentParser
from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedError

from fedbiomed.researcher.config import ResearcherConfig

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
            ["component", "create", "--root", self.tem.name, "--component", "NODE", "-eo"]
        )
        self.conf_parser.create(args)

        args = self.main_parser.parse_args(
            ["component", "create", "--root", self.tem.name, "--component", "RESEARCHER", "-eo"]
        )
        self.conf_parser.create(args)

    @patch("builtins.print")
    def test_03_component_parser_refresh(
        self,
        mock_print,
    ):
        args = self.main_parser.parse_args(
            ["component", "create", "--root", self.tem.name, "--component",  "NODE"]
        )
        self.conf_parser.create(args)

        args = self.main_parser.parse_args(
            ["component", "refresh", "--root", self.tem.name, "--component",  "NODE"]
        )
        self.conf_parser.refresh(args)


class TestCommonCLI(unittest.TestCase):

    def setUp(self) -> None:
        self.patch_certificate_manager = patch(
            "fedbiomed.common.cli.CertificateManager.__init__",
            MagicMock(return_value=None),
        )
        self.patch_set_db = patch("fedbiomed.common.cli.CertificateManager.set_db")

        self.mock_certificate_manager = self.patch_certificate_manager.start()
        self.mock_set_db = self.patch_set_db.start()
        self.config = MagicMock()
        self.cli = CommonCLI()
        self.cli.config = self.config


    def tearDown(self) -> None:
        self.patch_certificate_manager.stop()
        self.patch_set_db.stop()

    def test_01_common_cli_getters_and_setters(self):
        self.cli.description = "My CLI"

        self.assertEqual(self.cli.description, "My CLI")
        self.assertEqual(self.cli.parser, self.cli._parser)

        self.assertEqual(self.cli.arguments, None)

        self.assertTrue(self.cli.subparsers)

    def test_02_error_message(self):
        with patch("builtins.print") as patch_print:
            with self.assertRaises(SystemExit):
                self.cli.error("Hello this is error message")
                self.assertEqual(patch_print.call_count, 2)

    def test_03_success_message(self):
        with patch("builtins.print") as patch_print:
            self.cli.success("Hello this is success message")
            self.assertEqual(patch_print.call_count, 2)

    def test_04_bis_cli_initialize_optional(self):

        self.cli.initialize_optional()

        self.assertTrue("certificate-dev-setup" in self.cli._subparsers.choices)
        self.assertTrue("component" in self.cli._subparsers.choices)

    def test_04_common_cli_initialize_magic_dev_environment_parsers(self):
        self.cli.initialize_magic_dev_environment_parsers()

        self.assertTrue("certificate-dev-setup" in self.cli._subparsers.choices)
        self.assertEqual(
            self.cli._subparsers.choices["certificate-dev-setup"]
            ._defaults["func"]
            .__func__.__name__,
            "_create_magic_dev_environment",
        )

    def test_06_common_cli_initialize_certificate_parser(self):
        self.cli.initialize_certificate_parser()
        self.assertTrue("certificate" in self.cli._subparsers.choices)

        choices = (
            self.cli._subparsers.choices["certificate"]
            ._subparsers._group_actions[0]
            .choices
        )

        self.assertTrue("register" in choices)
        self.assertTrue("list" in choices)
        self.assertTrue("delete" in choices)
        self.assertTrue("generate" in choices)
        self.assertTrue("registration-instructions" in choices)

        self.assertEqual(
            "_register_certificate",
            choices["register"]._defaults["func"].__func__.__name__,
        )
        self.assertEqual(
            "_generate_certificate",
            choices["generate"]._defaults["func"].__func__.__name__,
        )
        self.assertEqual(
            "_delete_certificate", choices["delete"]._defaults["func"].__func__.__name__
        )
        self.assertEqual(
            "_list_certificates", choices["list"]._defaults["func"].__func__.__name__
        )
        self.assertEqual(
            "_prepare_certificate_for_registration",
            choices["registration-instructions"]._defaults["func"].__func__.__name__,
        )

        register_options = choices["register"]._positionals._option_string_actions
        self.assertTrue("--party-id" in register_options)
        self.assertTrue("--public-key" in register_options)

        generate_options = choices["generate"]._positionals._option_string_actions
        self.assertTrue("--path" in generate_options)
        self.assertTrue("--force" in generate_options)

    @patch("fedbiomed.common.cli.get_existing_component_db_names")
    @patch("fedbiomed.common.cli.get_all_existing_certificates")
    @patch("fedbiomed.common.cli.CertificateManager.insert")
    @patch("fedbiomed.common.cli.CommonCLI.error")
    def test_07_common_cli_create_magic_dev_environment(
        self,
        mock_cm_error,
        mock_cm_insert,
        mock_get_all_certificates,
        mock_get_components_db_names,
    ):
        mock_get_components_db_names.return_value = {
            "researcher": "db-researcher",
            "node-1": "db-node-1",
            "node-2": "db-node-2",
        }

        certificates = [
            {
                "party_id": "researcher",
                "certificate": "my-certificate",
                "ip": "localhost",
                "port": 1234,
                "component": "researcher",
            },
            {
                "party_id": "node-1",
                "certificate": "my-certificate",
                "ip": "localhost",
                "port": 1235,
                "component": "node",
            },
            {
                "party_id": "node-2",
                "certificate": "my-certificate",
                "ip": "localhost",
                "port": 1236,
                "component": "researcher",
            },
        ]

        mock_get_all_certificates.return_value = certificates

        with patch("fedbiomed.common.cli.ROOT_DIR", "path/to/root"):

            self.cli._create_magic_dev_environment(None)

            self.assertEqual(self.mock_set_db.call_count, 3)

            self.assertEqual(
                mock_cm_insert.call_args_list[0].kwargs,
                {**certificates[1], "upsert": True},
            )
            self.assertEqual(
                mock_cm_insert.call_args_list[1].kwargs,
                {**certificates[2], "upsert": True},
            )

            mock_cm_insert.side_effect = FedbiomedError
            self.cli._create_magic_dev_environment(None)
            self.assertEqual(mock_cm_error.call_count, 6)

            mock_get_all_certificates.return_value = ["test", "test"]
            with patch("builtins.print") as mock_print:
                self.cli._create_magic_dev_environment(None)
                self.assertEqual(mock_print.call_count, 2)

    @patch("builtins.open")
    @patch("builtins.print")
    @patch("os.path.isfile")
    def test_07_common_cli_generate_certificate(
        self, mock_is_file, mock_print, mock_open
    ):
        mock_is_file.return_value = True
        tmp_dir = tempfile.mkdtemp()
        self.cli.initialize_certificate_parser()
        args = self.cli.parser.parse_args(
            ["certificate", "generate", "--path", "dummy/path/" "-f"]
        )

        mock_cli_error = patch(
            "fedbiomed.common.cli.CommonCLI.error", MagicMock(return_value=None)
        )
        mock_cli_error.start()
        self.cli._generate_certificate(args)
        mock_cli_error.stop()

        # Remove tmp directory
        shutil.rmtree(tmp_dir)

        with self.assertRaises(SystemExit):
            self.cli._generate_certificate(args)

        mock_is_file.return_value = True
        args = self.cli.parser.parse_args(
            ["certificate", "generate", "--path", "dummy/path/"]
        )
        with self.assertRaises(SystemExit):
            self.cli._generate_certificate(args)

    @patch("fedbiomed.common.cli.CertificateManager.register_certificate")
    @patch("builtins.open")
    @patch("builtins.print")
    def test_08_common_cli_register_certificate(
        self, mock_print, mock_open, mock_register_certificate
    ):
        self.cli.initialize_certificate_parser()
        args = self.cli.parser.parse_args(
            [
                "certificate",
                "register",
                "--party-id",
                "party-id-1",
                "--public-key",
                "path/to/key",
                "--upsert",
            ]
        )

        self.cli._register_certificate(args)

        mock_register_certificate.assert_called_once_with(
            certificate_path="path/to/key",
            party_id="party-id-1",
            upsert=True,
        )
        self.assertEqual(mock_print.call_count, 2)

        mock_register_certificate.side_effect = FedbiomedError
        with self.assertRaises(SystemExit):
            self.cli._register_certificate(args)

    @patch("fedbiomed.common.cli.CertificateManager.list")
    @patch("builtins.open")
    def test_09_common_cli_list_certificates(self, mock_open, mock_cm_list):
        self.cli.initialize_certificate_parser()
        args = self.cli.parser.parse_args(["certificate", "list"])

        self.cli._list_certificates(args)
        mock_cm_list.assert_called_once()

    @patch("fedbiomed.common.cli.CertificateManager.list")
    @patch("fedbiomed.common.cli.CertificateManager.delete")
    @patch("fedbiomed.common.cli.CommonCLI.error")
    @patch("fedbiomed.common.cli.CommonCLI.success")
    @patch("builtins.input")
    @patch("builtins.open")
    @patch("builtins.print")
    def test_10_common_cli_delete_certificate(
        self,
        mock_print,
        mock_open,
        mock_input,
        mock_success,
        mock_error,
        mock_delete,
        mock_list,
    ):
        self.cli.initialize_certificate_parser()
        args = self.cli.parser.parse_args(["certificate", "delete"])

        mock_list.return_value = [{"party_id": "party-1"}, {"party_id": "party-2"}]
        mock_input.return_value = 1
        self.cli._delete_certificate(args)
        mock_delete.assert_called_once_with(party_id="party-1")
        mock_success.assert_called_once()

        mock_input.side_effect = [ValueError, 1]
        self.cli._delete_certificate(args)
        mock_error.assert_called_once()

    @patch("builtins.open")
    @patch("builtins.print")
    def test_11_common_cli_prepare_certificate_for_registration(
        self , mock_print, mock_open
    ):

        self.cli.initialize_certificate_parser()
        args = self.cli.parser.parse_args(["certificate", "registration-instructions"])

        mock_open.return_value.__enter__.return_value.read.return_value = (
            "test-certificate"
        )
        self.cli._prepare_certificate_for_registration(args)
        self.assertEqual(mock_print.call_args_list[2][0][0], "test-certificate")


    @patch("fedbiomed.common.cli.CertificateManager.list")
    @patch("fedbiomed.common.cli.CommonCLI._create_magic_dev_environment")
    def test_12_common_cli_parse_args(self, mock_dev_environment, mock_list):

        self.cli.initialize_certificate_parser()

        args = self.cli.parser.parse_args(["certificate", "list"])
        sys.argv = ["fedbiomed_run", "certificate", "list"]
        self.cli.parse_args()
        mock_list.assert_called_once_with(verbose=True)

        self.cli.initialize_magic_dev_environment_parsers()
        args = self.cli.parser.parse_args(["certificate-dev-setup"])

        sys.argv = ["fedbiomed_run", "certificate-dev-setup"]
        self.cli.parse_args()
        mock_dev_environment.assert_called_once_with(args, [])

        with self.assertRaises(SystemExit):
            # node argument is not known yet
            sys.argv = ["fedbiomed_run", "node", "dataset"]
            self.cli.parse_args()


if __name__ == "__main__":
    unittest.main()

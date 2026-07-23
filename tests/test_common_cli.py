import sys
from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.cli import CommonCLI
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def set_db():
    with (
        patch(
            "fedbiomed.common.cli.CertificateManager.__init__",
            MagicMock(return_value=None),
        ),
        patch("fedbiomed.common.cli.CertificateManager.set_db") as mock_set_db,
    ):
        yield mock_set_db


@pytest.fixture
def cli(set_db):
    cli = CommonCLI()
    cli.config = MagicMock()
    return cli


def test_common_cli_getters_and_setters(cli):
    cli.description = "My CLI"

    assert cli.description == "My CLI"
    assert cli.parser == cli._parser

    assert cli.arguments is None

    assert cli.subparsers


def test_error_message(cli):
    with patch("builtins.print") as patch_print:
        with pytest.raises(SystemExit):
            cli.error("Hello this is error message")
        assert patch_print.call_count == 2


def test_success_message(cli):
    with patch("builtins.print") as patch_print:
        cli.success("Hello this is success message")
        assert patch_print.call_count == 2


def test_cli_initialize_optional(cli):
    cli.initialize_optional()

    assert "certificate-dev-setup" in cli._subparsers.choices


def test_common_cli_initialize_magic_dev_environment_parsers(cli):
    cli.initialize_magic_dev_environment_parsers()

    assert "certificate-dev-setup" in cli._subparsers.choices
    assert (
        cli._subparsers.choices["certificate-dev-setup"]
        ._defaults["func"]
        .__func__.__name__
        == "_create_magic_dev_environment"
    )


def test_common_cli_initialize_certificate_parser(cli):
    cli.initialize_certificate_parser()
    assert "certificate" in cli._subparsers.choices

    choices = (
        cli._subparsers.choices["certificate"]._subparsers._group_actions[0].choices
    )

    assert "register" in choices
    assert "list" in choices
    assert "delete" in choices
    assert "generate" in choices
    assert "registration-instructions" in choices

    assert choices["register"]._defaults["func"].__func__.__name__ == (
        "_register_certificate"
    )
    assert choices["generate"]._defaults["func"].__func__.__name__ == (
        "_generate_certificate"
    )
    assert choices["delete"]._defaults["func"].__func__.__name__ == (
        "_delete_certificate"
    )
    assert choices["list"]._defaults["func"].__func__.__name__ == "_list_certificates"
    assert choices["registration-instructions"]._defaults["func"].__func__.__name__ == (
        "_prepare_certificate_for_registration"
    )

    register_options = choices["register"]._positionals._option_string_actions
    assert "--party-id" in register_options
    assert "--public-key" in register_options

    generate_options = choices["generate"]._positionals._option_string_actions
    assert "--path" in generate_options


@patch("fedbiomed.common.cli.get_existing_component_db_names")
@patch("fedbiomed.common.cli.get_all_existing_certificates")
@patch("fedbiomed.common.cli.CertificateManager.insert")
@patch("fedbiomed.common.cli.CommonCLI.error")
def test_common_cli_create_magic_dev_environment(
    mock_cm_error,
    mock_cm_insert,
    mock_get_all_certificates,
    mock_get_components_db_names,
    cli,
    set_db,
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
        cli._create_magic_dev_environment(None)

        assert set_db.call_count == 3

        assert mock_cm_insert.call_args_list[0].kwargs == {
            **certificates[1],
            "upsert": True,
        }
        assert mock_cm_insert.call_args_list[1].kwargs == {
            **certificates[2],
            "upsert": True,
        }

        mock_cm_insert.side_effect = FedbiomedError
        cli._create_magic_dev_environment(None)
        assert mock_cm_error.call_count == 6

        mock_get_all_certificates.return_value = ["test", "test"]
        with patch("builtins.print") as mock_print:
            cli._create_magic_dev_environment(None)
            assert mock_print.call_count == 2


@patch("builtins.open")
@patch("builtins.print")
@patch("os.path.isfile")
def test_common_cli_generate_certificate(mock_is_file, mock_print, mock_open, cli):
    """Generation aborts when a certificate already exists at the path."""
    mock_is_file.return_value = True
    cli.initialize_certificate_parser()
    args = cli.parser.parse_args(["certificate", "generate", "--path", "dummy/path/"])
    cli.config.get.return_value = "test"

    with pytest.raises(SystemExit):
        cli._generate_certificate(args)


@patch("fedbiomed.common.cli.CertificateManager.register_certificate")
@patch("builtins.open")
@patch("builtins.print")
def test_common_cli_register_certificate(
    mock_print, mock_open, mock_register_certificate, cli, set_db
):
    cli.initialize_certificate_parser()
    cli.config.COMPONENT_TYPE = "NODE"
    args = cli.parser.parse_args(
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

    cli._register_certificate(args)

    # Registration targets the component's main database.
    set_db.assert_called_once_with(db_path=cli.config.getpath("default", "db"))
    # The registering component is passed along so certificates of the
    # component's own kind are rejected.
    mock_register_certificate.assert_called_once_with(
        certificate_path="path/to/key",
        party_id="party-id-1",
        upsert=True,
        registering_component="NODE",
    )
    assert mock_print.call_count == 2

    mock_register_certificate.side_effect = FedbiomedError
    with pytest.raises(SystemExit):
        cli._register_certificate(args)


@patch("fedbiomed.common.cli.CertificateManager.list")
@patch("builtins.open")
def test_common_cli_list_certificates(mock_open, mock_cm_list, cli):
    cli.initialize_certificate_parser()
    args = cli.parser.parse_args(["certificate", "list"])

    cli._list_certificates(args)
    mock_cm_list.assert_called_once()


@patch("fedbiomed.common.cli.CertificateManager.list")
@patch("fedbiomed.common.cli.CertificateManager.delete")
@patch("fedbiomed.common.cli.CommonCLI.error")
@patch("fedbiomed.common.cli.CommonCLI.success")
@patch("builtins.input")
@patch("builtins.open")
@patch("builtins.print")
def test_common_cli_delete_certificate(
    mock_print,
    mock_open,
    mock_input,
    mock_success,
    mock_error,
    mock_delete,
    mock_list,
    cli,
):
    cli.initialize_certificate_parser()
    args = cli.parser.parse_args(["certificate", "delete"])

    mock_list.return_value = [{"party_id": "party-1"}, {"party_id": "party-2"}]
    mock_input.return_value = 1
    cli._delete_certificate(args)
    mock_delete.assert_called_once_with(party_id="party-1")
    mock_success.assert_called_once()

    mock_input.side_effect = [ValueError, 1]
    cli._delete_certificate(args)
    mock_error.assert_called_once()


@patch("builtins.open")
@patch("builtins.print")
def test_common_cli_prepare_certificate_for_registration(mock_print, mock_open, cli):
    cli.initialize_certificate_parser()
    args = cli.parser.parse_args(["certificate", "registration-instructions"])

    mock_open.return_value.__enter__.return_value.read.return_value = "test-certificate"
    cli._prepare_certificate_for_registration(args)
    assert mock_print.call_args_list[2][0][0] == "test-certificate"


@patch("fedbiomed.common.cli.CertificateManager.list")
@patch("fedbiomed.common.cli.CommonCLI._create_magic_dev_environment")
def test_common_cli_parse_args(mock_dev_environment, mock_list, cli, monkeypatch):
    cli.initialize_certificate_parser()

    monkeypatch.setattr(sys, "argv", ["fedbiomed", "certificate", "list"])
    cli.parse_args()
    mock_list.assert_called_once_with(verbose=True)

    cli.initialize_magic_dev_environment_parsers()
    args = cli.parser.parse_args(["certificate-dev-setup"])

    monkeypatch.setattr(sys, "argv", ["fedbiomed", "certificate-dev-setup"])
    cli.parse_args()
    mock_dev_environment.assert_called_once_with(args, [])

    with pytest.raises(SystemExit):
        # node argument is not known yet
        monkeypatch.setattr(sys, "argv", ["fedbiomed", "node", "dataset"])
        cli.parse_args()

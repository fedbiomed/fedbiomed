import sys
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.x509.oid import NameOID

from fedbiomed.common.cli import CommonCLI
from fedbiomed.common.exceptions import FedbiomedCertificateError, FedbiomedError

_NODE_1 = {"party_id": "NODE_1", "component": "NODE"}
_NODE_2 = {"party_id": "NODE_2", "component": "NODE"}
_NODE_3 = {"party_id": "NODE_3", "component": "NODE"}
_RESEARCHER_1 = {"party_id": "RESEARCHER_1", "component": "RESEARCHER"}
_RESEARCHER_2 = {"party_id": "RESEARCHER_2", "component": "RESEARCHER"}


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


@pytest.mark.parametrize(
    "component,expected",
    [
        ("RESEARCHER_1", ["NODE_1", "NODE_2"]),
        ("NODE_1", ["RESEARCHER_1"]),
        ("NODE_2", ["RESEARCHER_1"]),
    ],
)
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
    component,
    expected,
):
    """Each component receives only the certificates of the parties it talks to.

    A researcher gets the node certificates; a node gets the researcher's and
    nothing else — no other node, not itself, and never a second researcher.
    """
    mock_get_components_db_names.return_value = {component: "db"}
    mock_get_all_certificates.return_value = [
        {**_RESEARCHER_1, "certificate": "cert-r1"},
        {**_NODE_1, "certificate": "cert-n1"},
        {**_NODE_2, "certificate": "cert-n2"},
    ]

    with patch("fedbiomed.common.cli.ROOT_DIR", "path/to/root"):
        cli._create_magic_dev_environment(None)

    assert set_db.call_count == 1
    assert [c.kwargs["party_id"] for c in mock_cm_insert.call_args_list] == expected
    assert all(c.kwargs["upsert"] is True for c in mock_cm_insert.call_args_list)
    mock_cm_error.assert_not_called()


@pytest.mark.parametrize(
    "certificates,reported",
    [
        # A researcher, but a federation of one node
        ([_RESEARCHER_1, _NODE_1], "1 node(s)"),
        # Enough components, but nobody to train them: what a count alone misses
        ([_NODE_1, _NODE_2, _NODE_3], "0 researcher(s)"),
        # A node cannot tell which of two researchers to pin
        ([_RESEARCHER_1, _RESEARCHER_2, _NODE_1, _NODE_2], "2 researcher(s)"),
    ],
)
@patch("fedbiomed.common.cli.get_existing_component_db_names")
@patch("fedbiomed.common.cli.get_all_existing_certificates")
@patch("fedbiomed.common.cli.CertificateManager.insert")
def test_create_magic_dev_environment_requires_a_whole_federation(
    mock_cm_insert,
    mock_get_all_certificates,
    mock_get_components_db_names,
    cli,
    set_db,
    certificates,
    reported,
):
    """One researcher and at least two nodes, checked as a shape not a count.

    Counting components alone would accept a clone with no researcher at all,
    whose node databases would then be set up empty.
    """
    mock_get_components_db_names.return_value = {
        c["party_id"]: "db" for c in certificates
    }
    mock_get_all_certificates.return_value = [
        {**c, "certificate": f"cert-{c['party_id']}"} for c in certificates
    ]

    with patch("builtins.print") as mock_print:
        with pytest.raises(SystemExit):
            cli._create_magic_dev_environment(None)

    # The message names what is missing, and nothing was written
    printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
    assert reported in printed
    mock_cm_insert.assert_not_called()


@patch("fedbiomed.common.cli.get_existing_component_db_names")
@patch("fedbiomed.common.cli.get_all_existing_certificates")
@patch("fedbiomed.common.cli.CertificateManager.insert")
@patch("fedbiomed.common.cli.validate_registering_component")
def test_create_magic_dev_environment_reports_rejected_certificate(
    mock_validate,
    mock_cm_insert,
    mock_get_all_certificates,
    mock_get_components_db_names,
    cli,
    set_db,
):
    """A certificate the shared rule rejects is reported, never skipped quietly.

    Skipping a component's own kind is routine, but any other rejection means
    the certificate contradicts the component it is declared as.
    """
    mock_get_components_db_names.return_value = {"NODE_1": "db-node-1"}
    mock_get_all_certificates.return_value = [
        {**_RESEARCHER_1, "certificate": "cert-r1"},
        {**_NODE_1, "certificate": "cert-n1"},
        {**_NODE_2, "certificate": "cert-n2"},
    ]
    mock_validate.side_effect = FedbiomedCertificateError("restricted to a TLS client")

    with patch("builtins.print"):
        with pytest.raises(SystemExit):
            cli._create_magic_dev_environment(None)

    # Own kind is filtered before validation, so only the researcher was checked
    mock_validate.assert_called_once()
    assert mock_validate.call_args.args[1:] == ("RESEARCHER", "NODE")
    mock_cm_insert.assert_not_called()


def _generating_cli(cli, tmp_path, component, name):
    """Prepares `cli` to regenerate `name` for a component of the given type."""
    cli.initialize_certificate_parser()
    cli.config.COMPONENT_TYPE = component
    cli.config.getpath.return_value = str(tmp_path / f"{name}.pem")
    cli.config.get.side_effect = lambda section, key: {
        ("default", "id"): f"{component}_1",
        ("server", "host"): "researcher-host",
    }[(section, key)]
    return cli


@pytest.mark.parametrize(
    "component,name,common_name",
    [
        ("RESEARCHER", "server_certificate", "researcher-host"),
        ("NODE", "FBM_certificate", "*"),
    ],
)
@patch("builtins.print")
def test_generate_certificate_is_named_and_subjected_as_configured(
    mock_print, cli, tmp_path, component, name, common_name
):
    """The certificate is written where the configuration expects, under its name.

    A researcher is pinned by nodes against its server host, so its certificate
    must carry that name; a wildcard one cannot be verified.
    """
    args = _generating_cli(cli, tmp_path, component, name).parser.parse_args(
        ["certificate", "generate"]
    )

    cli._generate_certificate(args)

    pem = tmp_path / f"{name}.pem"
    assert pem.is_file() and (tmp_path / f"{name}.key").is_file()
    certificate = x509.load_pem_x509_certificate(pem.read_bytes())
    assert (
        certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        == common_name
    )


@patch("builtins.print")
def test_generate_certificate_researcher_is_pinnable(mock_print, cli, tmp_path):
    """A researcher certificate carries the SAN a pinning node verifies against."""
    args = _generating_cli(
        cli, tmp_path, "RESEARCHER", "server_certificate"
    ).parser.parse_args(["certificate", "generate"])

    cli._generate_certificate(args)

    certificate = x509.load_pem_x509_certificate(
        (tmp_path / "server_certificate.pem").read_bytes()
    )
    san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    assert san.value.get_values_for_type(x509.DNSName) == ["researcher-host"]


@patch("builtins.print")
def test_generate_certificate_refuses_to_replace_without_force(
    mock_print, cli, tmp_path
):
    """Regenerating destroys the private key, so it is not done by accident."""
    args = _generating_cli(cli, tmp_path, "NODE", "FBM_certificate").parser.parse_args(
        ["certificate", "generate"]
    )
    cli._generate_certificate(args)
    original = (tmp_path / "FBM_certificate.pem").read_bytes()

    with pytest.raises(SystemExit):
        cli._generate_certificate(args)
    assert (tmp_path / "FBM_certificate.pem").read_bytes() == original

    forced = cli.parser.parse_args(["certificate", "generate", "--force"])
    cli._generate_certificate(forced)
    assert (tmp_path / "FBM_certificate.pem").read_bytes() != original


@patch("fedbiomed.common.cli.CertificateManager.register_certificate")
@patch("builtins.open")
@patch("builtins.print")
def test_common_cli_register_certificate(
    mock_print, mock_open, mock_register_certificate, cli, set_db
):
    cli.initialize_certificate_parser()
    cli.config.COMPONENT_TYPE = "NODE"
    # The party id normally comes from the certificate, not from the command line
    mock_register_certificate.return_value = "RESEARCHER_1"
    args = cli.parser.parse_args(
        ["certificate", "register", "--public-key", "path/to/key", "--upsert"]
    )

    cli._register_certificate(args)

    # Registration targets the component's main database.
    set_db.assert_called_once_with(db_path=cli.config.getpath("default", "db"))
    # The registering component is passed along so certificates of the
    # component's own kind are rejected.
    mock_register_certificate.assert_called_once_with(
        certificate_path="path/to/key",
        party_id=None,
        upsert=True,
        registering_component="NODE",
    )
    # The party actually registered is named, whether or not it was supplied
    printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
    assert "RESEARCHER_1" in printed

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


@pytest.mark.parametrize(
    "certificates,reason",
    [
        ([_NODE_1], "own_component_type_registered"),
        ([_RESEARCHER_1, _RESEARCHER_2], "multiple_certificates_on_node"),
        ([_RESEARCHER_1], None),  # consistent registry: nothing to audit
    ],
)
@patch("fedbiomed.common.cli.logger.security_event")
@patch("fedbiomed.common.cli.CertificateManager.list")
@patch("builtins.open")
def test_list_certificates_audits_inconsistencies(
    mock_open, mock_cm_list, security_event, cli, certificates, reason
):
    """Registry invariants broken by entries predating the checks are audited."""
    cli.config.COMPONENT_TYPE = "NODE"
    mock_cm_list.return_value = certificates
    cli.initialize_certificate_parser()

    cli._list_certificates(cli.parser.parse_args(["certificate", "list"]))

    if reason is None:
        security_event.assert_not_called()
        return
    security_event.assert_called_once()
    event = security_event.call_args.kwargs
    assert event["operation"] == "certificate_registry_inconsistent"
    assert event["status"] == "warning"
    assert event["reason"] == reason
    assert event["party_ids"] == [c["party_id"] for c in certificates]


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


@pytest.mark.parametrize(
    "component,registers_on,not_mentioned",
    [
        ("NODE", "researcher", "fedbiomed node certificate register"),
        ("RESEARCHER", "node", "fedbiomed researcher certificate register"),
    ],
)
@patch("builtins.open")
@patch("builtins.print")
def test_common_cli_prepare_certificate_for_registration(
    mock_print, mock_open, cli, component, registers_on, not_mentioned
):
    """The printed command is the one the *other* kind of party must run.

    A party registers the certificates of the components it talks to, so telling
    the reader to run it on their own kind would be rejected by registration.
    """
    cli.initialize_certificate_parser()
    cli.config.COMPONENT_TYPE = component
    cli.config.get.return_value = f"{component}_1"
    args = cli.parser.parse_args(["certificate", "registration-instructions"])

    mock_open.return_value.__enter__.return_value.read.return_value = "test-certificate"
    cli._prepare_certificate_for_registration(args)

    printed = "\n".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
    assert "test-certificate" in printed
    assert f"fedbiomed {registers_on} certificate register -pk" in printed
    assert not_mentioned not in printed
    # `-pi` is unnecessary: the party id is embedded in the certificate
    assert "-pi " not in printed


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

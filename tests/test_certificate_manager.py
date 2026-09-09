import ipaddress
import os
import stat
import tempfile
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from fedbiomed.common.certificate_manager import (
    CERT_ORGANIZATION,
    CERT_PURPOSE_CLIENT,
    CERT_PURPOSE_SERVER,
    CERTIFICATE_EXPIRY_WARNING_DAYS,
    CertificateManager,
    TrustedCertificateBundle,
    certificate_audit_fields,
    certificate_component_id,
    certificate_expiry,
    certificate_san_names,
    generate_certificate,
    generate_component_certificate,
    is_loopback_name,
    san_entry,
    write_certificate_pair,
)
from fedbiomed.common.constants import CERTS_FOLDER_NAME, ComponentType
from fedbiomed.common.exceptions import FedbiomedCertificateError
from fedbiomed.common.utils import read_file

_NODE_A = "NODE_4f2c8a10-0e7d-4a11-9c33-8b7f0a1d2e44"
_NODE_B = "NODE_9c2b1d70-1111-2222-3333-444455556666"
_NODE_C = "NODE_0a1b2c3d-aaaa-bbbb-cccc-ddddeeeeffff"
_RESEARCHER_A = "RESEARCHER_9c2b1d70-1111-2222-3333-444455556666"
_RESEARCHER_B = "RESEARCHER_7e6d5c40-9999-8888-7777-666655554444"


def _events(security_event, operation):
    """Audit events of one operation recorded by a patched `logger.security_event`.

    Filtering by operation is required: `logger` is a singleton, so a patched
    `security_event` also records the events `DBTable` emits for table access.
    """
    return [
        call
        for call in security_event.call_args_list
        if call.kwargs.get("operation") == operation
    ]


def _self_signed(
    folder, component_id, purpose=CERT_PURPOSE_CLIENT, san=("localhost", "127.0.0.1")
):
    """Generates a self-signed certificate, returns its PEM file path."""
    _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=folder,
        certificate_name=component_id.replace(" ", "_"),
        component_id=component_id,
        purpose=purpose,
        san=list(san),
    )
    return pem_file


def _certificate(org="Hospital", common_name=None, san=None, valid_days=1):
    """A certificate not issued by Fed-BioMed, as PEM bytes.

    Subject fields and names are chosen freely, which is what a certificate issued
    elsewhere may combine in ways Fed-BioMed never generates.

    `valid_days` is how many days from now the certificate expires; a negative value
    yields one that already expired. It is issued a year before it expires, so it is
    valid from a date in the past either way.
    """
    pkey = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    attributes = [x509.NameAttribute(NameOID.ORGANIZATION_NAME, org)]
    if common_name is not None:
        attributes.append(x509.NameAttribute(NameOID.COMMON_NAME, common_name))

    not_after = datetime.now(timezone.utc) + timedelta(days=valid_days)
    name = x509.Name(attributes)
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(pkey.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_after - timedelta(days=365))
        .not_valid_after(not_after)
    )
    if san is not None:
        builder = builder.add_extension(
            x509.SubjectAlternativeName(san), critical=False
        )

    return builder.sign(private_key=pkey, algorithm=hashes.SHA256()).public_bytes(
        serialization.Encoding.PEM
    )


def _third_party(folder, org, common_name=None):
    """A certificate not issued by Fed-BioMed, with an arbitrary subject.

    Carrying no `O=Fed-BioMed`, it embeds no identity, so it is the only kind a
    component id can be chosen for at registration.
    """
    pem_file = os.path.join(folder, f"{org}_{common_name}.pem")
    with open(pem_file, "wb") as file:
        file.write(_certificate(org=org, common_name=common_name))
    return pem_file


def _pem(pem_file):
    with open(pem_file, "rb") as f:
        return f.read()


def _load(pem_file):
    return x509.load_pem_x509_certificate(_pem(pem_file))


# Generating the RSA key dominates this file's runtime, and a label always means the
# same certificate, so both factories below are cached for the whole session.


@pytest.fixture(scope="session")
def third_party_certificate():
    """A certificate issued outside Fed-BioMed, so it embeds no identity.

    Registering one takes an explicit `component_id`; the label only tells two apart.
    """

    @lru_cache(maxsize=None)
    def certificate(label="cert"):
        return _certificate(common_name=label).decode("utf-8")

    return certificate


@pytest.fixture(scope="session")
def issued_certificate():
    """A Fed-BioMed certificate, for a test that needs no file to read it in."""

    @lru_cache(maxsize=None)
    def certificate(component_id):
        with tempfile.TemporaryDirectory() as folder:
            return _pem(_self_signed(folder, component_id)).decode("utf-8")

    return certificate


# -----------------------------------------------------------------------------
# CertificateManager over a real TinyDB
# -----------------------------------------------------------------------------


def test_certificate_manager_initialization(tmp_path, third_party_certificate):
    """A manager opened on a path reads and writes that database."""
    db_path = str(tmp_path / "certs.json")
    certificate = third_party_certificate()
    cm = CertificateManager(
        db_path=db_path, component_type=ComponentType.RESEARCHER.name
    )
    try:
        cm.register(
            certificate=certificate,
            component_id=_NODE_A,
        )
    finally:
        cm.close()

    reopened = CertificateManager(
        db_path=db_path, component_type=ComponentType.RESEARCHER.name
    )
    try:
        assert reopened.get(component_id=_NODE_A)["certificate"] == certificate
    finally:
        reopened.close()


@pytest.mark.parametrize("component_type", ["", "node", "GUI", "RESEARCHER "])
def test_opening_for_something_that_is_not_a_component_type_refused(
    tmp_path, component_type
):
    """The rules a registration is held to are the component's, so it names one.

    Neither of the two component types would select no rules at all.
    """
    with pytest.raises(FedbiomedCertificateError, match="not a component type"):
        CertificateManager(
            db_path=str(tmp_path / "certs.json"), component_type=component_type
        )


def test_manager_holds_the_rules_of_the_component_it_was_opened_for(
    tmp_path, third_party_certificate
):
    """Two components' managers over the same certificate differ by their rules.

    `certificate-dev-setup` opens one manager per component for this reason: a
    certificate is held to the rules of the component receiving it.
    """
    certificate = third_party_certificate()
    researcher_cm = CertificateManager(
        str(tmp_path / "researcher.json"), ComponentType.RESEARCHER.name
    )
    node_cm = CertificateManager(str(tmp_path / "node.json"), ComponentType.NODE.name)
    try:
        # Stating no host, the certificate is one only a researcher registers
        researcher_cm.register(certificate=certificate, component_id=_NODE_A)

        with pytest.raises(FedbiomedCertificateError, match="states no host"):
            node_cm.register(certificate=certificate, component_id=_NODE_A)
    finally:
        researcher_cm.close()
        node_cm.close()


def test_certificate_manager_get(cert_db, third_party_certificate):
    """Only the requested component is returned; an unknown one yields nothing."""
    cert_a = third_party_certificate("a")
    cert_db.researcher_cm.register(
        certificate=cert_a,
        component_id=_NODE_A,
    )
    cert_db.researcher_cm.register(
        certificate=third_party_certificate("b"),
        component_id=_NODE_B,
    )

    assert cert_db.researcher_cm.get(component_id=_NODE_A)["certificate"] == cert_a
    assert cert_db.researcher_cm.get(component_id=_NODE_C) is None


def test_certificate_manager_registering_twice_requires_upsert(
    cert_db, third_party_certificate
):
    """A component can be registered once; registering again needs `upsert`."""
    first, second = third_party_certificate("first"), third_party_certificate("second")
    entry = dict(
        certificate=first,
        component_id=_NODE_A,
    )

    cert_db.researcher_cm.register(**entry)
    assert cert_db.researcher_cm.get(component_id=_NODE_A)["certificate"] == first

    with pytest.raises(FedbiomedCertificateError):
        cert_db.researcher_cm.register(**{**entry, "certificate": second})
    assert cert_db.researcher_cm.get(component_id=_NODE_A)["certificate"] == first

    cert_db.researcher_cm.register(**{**entry, "certificate": second}, upsert=True)
    assert cert_db.researcher_cm.get(component_id=_NODE_A)["certificate"] == second
    # Updating a component replaces its entry rather than adding one
    assert len(cert_db.researcher_cm.list()) == 1


def test_certificate_manager_delete(cert_db, third_party_certificate):
    """Deleting removes only the named component."""
    cert_db.researcher_cm.register(
        certificate=third_party_certificate("a"),
        component_id=_NODE_A,
    )
    cert_db.researcher_cm.register(
        certificate=third_party_certificate("b"),
        component_id=_NODE_B,
    )

    cert_db.researcher_cm.delete(component_id=_NODE_A)

    assert [d["component_id"] for d in cert_db.researcher_cm.list()] == [_NODE_B]


def test_certificate_manager_list(cert_db, third_party_certificate):
    """Tests list method of certificate manager"""
    cert_a = third_party_certificate("a")
    cert_db.researcher_cm.register(
        certificate=cert_a,
        component_id=_NODE_A,
    )

    assert [d["component_id"] for d in cert_db.researcher_cm.list()] == [_NODE_A]

    with patch("builtins.print") as mock_print:
        result = cert_db.researcher_cm.list(verbose=True)
        mock_print.assert_called_once()
    # Printing must not strip the certificate from what the caller receives
    assert result[0]["certificate"] == cert_a


def test_certificate_manager_register_certificate(cert_db):
    """`register_certificate` stores what the file at the given path holds."""

    with pytest.raises(FedbiomedCertificateError):
        cert_db.researcher_cm.register_certificate(
            certificate_path=os.path.join(cert_db.tmp, "missing.pem"),
            component_id=_NODE_A,
        )

    pem_file = _third_party(cert_db.tmp, "Hospital")
    registered = cert_db.researcher_cm.register_certificate(
        certificate_path=pem_file,
        component_id=_NODE_A,
    )

    assert registered == _NODE_A
    with open(pem_file, encoding="UTF-8") as f:
        assert (
            cert_db.researcher_cm.get(component_id=_NODE_A)["certificate"] == f.read()
        )


def test_register_certificate_returns_the_recovered_component_id(cert_db):
    """The caller learns who was registered even when it supplied no component id.

    The identity normally comes from the certificate, so the return value is the
    only way to report which component a registration applied to.
    """
    registered = cert_db.node_cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER),
    )

    assert registered == _RESEARCHER_A


def test_operations_on_a_closed_manager_raise(tmp_path):
    """Using a manager whose handle was released is a clear error, not an
    AttributeError."""
    cm = CertificateManager(str(tmp_path / "certs.json"), ComponentType.RESEARCHER.name)
    cm.close()

    with pytest.raises(FedbiomedCertificateError, match="closed"):
        cm.get(_NODE_A)


def _generate_in(certificate_folder):
    return CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=certificate_folder,
        certificate_name="certificate",
        component_id=_NODE_A,
        purpose=CERT_PURPOSE_CLIENT,
    )


def test_generate_writes_key_and_certificate_files(tmp_path):
    # Production always passes an absolute path (component roots are
    # absolutized before reaching certificate generation).
    key_file, pem_file = _generate_in(str(tmp_path))

    assert key_file == str(tmp_path / "certificate.key")
    assert pem_file == str(tmp_path / "certificate.pem")
    # Both are usable: a loadable certificate and its matching private key
    certificate = _load(pem_file)
    with open(key_file, "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=None)
    assert (
        certificate.public_key().public_numbers() == key.public_key().public_numbers()
    )


# Failing on the key file write, then on the certificate file write.
@pytest.mark.parametrize("side_effect", [Exception, [MagicMock(), Exception]])
def test_generate_raises_when_a_file_cannot_be_written(tmp_path, side_effect):
    with patch("fedbiomed.common.certificate_manager.open", side_effect=side_effect):
        with pytest.raises(FedbiomedCertificateError):
            _generate_in(str(tmp_path))


def test_generate_raises_for_non_existing_folder(tmp_path):
    with pytest.raises(FedbiomedCertificateError):
        _generate_in(str(tmp_path / "no-such-folder"))


def test_generate_rejects_relative_path():
    with pytest.raises(FedbiomedCertificateError):
        _generate_in("relative-dir")


# -----------------------------------------------------------------------------
# Certificate expiry helpers (`notAfter` parsing + reporting)
# -----------------------------------------------------------------------------


@pytest.fixture
def real_cert(tmp_path):
    """A real generated certificate as PEM bytes."""
    pem_file = _self_signed(str(tmp_path), _NODE_A)
    with open(pem_file, "rb") as f:
        return f.read()


def test_certificate_expiry_returns_future_date(real_cert):
    expiry = certificate_expiry(real_cert)
    assert isinstance(expiry, datetime)
    assert expiry > datetime.now(timezone.utc)


def test_certificate_expiry_none_for_unparsable():
    assert certificate_expiry(b"not a certificate") is None


def test_certificate_san_names_reads_the_subject_alternative_names(real_cert):
    assert certificate_san_names(real_cert) == ["localhost", "127.0.0.1", "::1"]


def test_certificate_san_names_accepts_str(real_cert):
    assert certificate_san_names(real_cert.decode()) == certificate_san_names(real_cert)


def test_certificate_san_names_ignores_the_common_name():
    """The Common Name is free text that states no host, whatever it looks like.

    Reading a host or an address back out of it would verify a peer against
    something the TLS layer does not check.
    """
    certificate = _certificate(
        common_name="not-a-host", san=[x509.DNSName("fbm.example.org")]
    )
    assert certificate_san_names(certificate) == ["fbm.example.org"]


@pytest.mark.parametrize(
    "common_name", ["fbm-researcher", "10.0.0.9", "10.0.0.9:50051"]
)
def test_certificate_san_names_empty_when_only_a_common_name(common_name):
    """A certificate stating no SAN is valid for no name, host or address alike."""
    assert certificate_san_names(_certificate(common_name=common_name)) == []


def test_certificate_san_names_ignores_entries_that_name_no_host():
    """A certificate issued elsewhere may hold an e-mail address or a URI in its
    SAN; TLS matches a server against neither."""
    certificate = _certificate(
        san=[
            x509.RFC822Name("admin@hospital.org"),
            x509.UniformResourceIdentifier("https://hospital.org/researcher"),
            x509.DNSName("fbm.example.org"),
        ]
    )
    assert certificate_san_names(certificate) == ["fbm.example.org"]


def test_certificate_san_names_empty_when_no_entry_names_a_host():
    """Such a certificate states no host at all — what a node refuses to connect on."""
    certificate = _certificate(san=[x509.RFC822Name("admin@hospital.org")])
    assert certificate_san_names(certificate) == []


@pytest.mark.parametrize("blank", ["", " "])
def test_certificate_san_names_ignores_a_blank_name(blank):
    """A blank entry states no host, and would verify a channel under an empty name."""
    certificate = _certificate(
        san=[x509.DNSName(blank), x509.DNSName("fbm.example.org")]
    )
    assert certificate_san_names(certificate) == ["fbm.example.org"]


def test_certificate_san_names_keeps_a_name_as_it_is_written():
    """The name is what TLS matches the peer against, so it is reported verbatim."""
    certificate = _certificate(san=[x509.DNSName("Fbm.Example.Org")])
    assert certificate_san_names(certificate) == ["Fbm.Example.Org"]


@pytest.mark.parametrize("certificate", [b"not a certificate", b"", None])
def test_certificate_san_names_empty_for_unparsable(certificate):
    assert certificate_san_names(certificate) == []


def test_component_id_read_from_a_generated_certificate(real_cert):
    """What Fed-BioMed issues identifies itself, in bytes and as text alike."""
    assert certificate_component_id(real_cert) == _NODE_A
    assert certificate_component_id(real_cert.decode()) == _NODE_A


@pytest.mark.parametrize("common_name", [_NODE_A, "node1.hospital-a.example.org"])
def test_component_id_none_when_another_issuer_signed_it(common_name):
    """Another issuer's CommonName is free text, even when it looks like an id.

    Reading it as one would register a certificate under a component id nobody
    in the federation assigned.
    """
    certificate = _certificate(org="Hospital", common_name=common_name)
    assert certificate_component_id(certificate) is None


def test_component_id_none_without_a_common_name():
    """The organization alone identifies nothing."""
    assert certificate_component_id(_certificate(org=CERT_ORGANIZATION)) is None


@pytest.mark.parametrize("certificate", [b"not a certificate", b"", None])
def test_component_id_none_for_unparsable(certificate):
    assert certificate_component_id(certificate) is None


def test_certificate_audit_fields_identify_the_certificate(real_cert):
    fields = certificate_audit_fields(real_cert)
    assert fields["cert_subject"] == f"CN={_NODE_A},O={CERT_ORGANIZATION}"
    assert fields["cert_issuer"] == f"CN={_NODE_A},O={CERT_ORGANIZATION}"
    assert fields["cert_san"] == "localhost,127.0.0.1,::1"
    # Serial as hex, expiry as an ISO-8601 instant
    assert int(fields["cert_serial"], 16) > 0
    assert fields["cert_not_after"].endswith("Z")
    # The certificate itself is never emitted
    assert not any("BEGIN CERTIFICATE" in value for value in fields.values())


def test_certificate_audit_fields_accepts_str(real_cert):
    assert certificate_audit_fields(real_cert.decode()) == certificate_audit_fields(
        real_cert
    )


@pytest.mark.parametrize("certificate", [b"not a certificate", b"", None])
def test_certificate_audit_fields_empty_for_undescribable(certificate):
    """Logging a connection must not raise on a certificate that cannot be read."""
    assert certificate_audit_fields(certificate) == {}


def test_expiring_certificates_filters_by_threshold(cert_db):
    """Each certificate is reported on its own `notAfter`, against the window."""
    for component_id in (_NODE_A, _RESEARCHER_A):
        cert_db.researcher_cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, component_id),
        )

    # Generated cert lasts ~5 years: a wide window catches it, a tight one doesn't
    assert {
        c for c, _ in cert_db.researcher_cm.expiring_certificates(within_days=10000)
    } == {
        _NODE_A,
        _RESEARCHER_A,
    }
    assert cert_db.researcher_cm.expiring_certificates(within_days=1) == []


def test_list_verbose_adds_expires_column(cert_db):
    cert_db.researcher_cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _NODE_A),
    )

    with patch("fedbiomed.common.certificate_manager.tabulate") as tabulate:
        cert_db.researcher_cm.list(verbose=True)

    rows = tabulate.call_args.args[0]
    assert "expires" in rows[0]
    assert "certificate" not in rows[0]


# -----------------------------------------------------------------------------
# `cryptography`-based self-signed certificate generation
# -----------------------------------------------------------------------------


def _san(cert):
    return cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value


def _extensions(cert):
    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    key_usage = cert.extensions.get_extension_for_class(x509.KeyUsage).value
    basic = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
    return eku, key_usage, basic


def test_subject_carries_the_component_id_and_the_organization(tmp_path):
    """The subject states which component this is, and that Fed-BioMed issued it.

    Where the component is reached is the SAN's business, and stays out of it.
    """
    subject = _load(_self_signed(str(tmp_path), _NODE_A)).subject
    assert subject.rfc4514_string() == f"CN={_NODE_A},O={CERT_ORGANIZATION}"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("10.0.0.9", x509.IPAddress(ipaddress.ip_address("10.0.0.9"))),
        ("127.0.0.1", x509.IPAddress(ipaddress.ip_address("127.0.0.1"))),
        ("::1", x509.IPAddress(ipaddress.ip_address("::1"))),
        ("localhost", x509.DNSName("localhost")),
        ("fbm.example.org", x509.DNSName("fbm.example.org")),
    ],
)
def test_san_entry_tells_an_address_from_a_name(name, expected):
    """What a peer verifies depends on the entry type, so the two never mix."""
    assert san_entry(name) == expected


@pytest.mark.parametrize(
    "name, loopback",
    [
        ("localhost", True),
        ("LocalHost", True),
        ("127.0.0.1", True),
        ("127.0.0.53", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("fbm.example.org", False),
        ("10.0.0.9", False),
        ("0.0.0.0", False),
    ],
)
def test_is_loopback_name(name, loopback):
    assert is_loopback_name(name) is loopback


@pytest.mark.parametrize("given", ["localhost", "127.0.0.1", "::1"])
def test_a_loopback_name_issues_all_three_forms(tmp_path, given):
    """A peer on this machine dials whichever form it holds, so naming one names
    all: the certificate carries what it is verified against."""
    pem_file = _self_signed(str(tmp_path), _NODE_A, san=[given])
    assert set(certificate_san_names(_pem(pem_file))) == {
        "localhost",
        "127.0.0.1",
        "::1",
    }
    # The form given stays first, the rest follow it
    assert certificate_san_names(_pem(pem_file))[0] == given


def test_a_loopback_name_keeps_the_other_names_given(tmp_path):
    """The expansion adds to what was asked for, it does not replace it."""
    pem_file = _self_signed(
        str(tmp_path), _NODE_A, san=["fbm.example.org", "localhost"]
    )
    assert certificate_san_names(_pem(pem_file)) == [
        "fbm.example.org",
        "localhost",
        "127.0.0.1",
        "::1",
    ]


def test_ip_produces_ip_san(tmp_path):
    """An address is an `iPAddress` entry, the only place TLS reads one from."""
    certificate = _load(_self_signed(str(tmp_path), _NODE_A, san=["10.0.0.5"]))
    assert _san(certificate).get_values_for_type(x509.IPAddress) == [
        ipaddress.ip_address("10.0.0.5")
    ]
    # The address is nowhere in the subject, which states who the component is
    assert certificate.subject.rfc4514_string() == f"CN={_NODE_A},O={CERT_ORGANIZATION}"


def test_only_the_names_given_are_issued(tmp_path):
    """Nothing is added to what the caller asks for: a peer dialling the component
    by a name it was not issued for, a loopback name included, is the caller's to
    foresee."""
    san = _san(_load(_self_signed(str(tmp_path), _NODE_A, san=["fbm-researcher"])))
    assert san.get_values_for_type(x509.DNSName) == ["fbm-researcher"]
    assert san.get_values_for_type(x509.IPAddress) == []


def test_every_name_given_is_kept_in_order(tmp_path):
    """A component reachable under several names is verifiable under each."""
    _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=str(tmp_path),
        certificate_name="multi",
        component_id=_RESEARCHER_A,
        purpose=CERT_PURPOSE_SERVER,
        san=["fbm-researcher", "fbm.example.org", "10.0.0.9"],
    )
    assert certificate_san_names(_pem(pem_file)) == [
        "fbm-researcher",
        "fbm.example.org",
        "10.0.0.9",
    ]


def test_names_are_not_repeated(tmp_path):
    _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=str(tmp_path),
        certificate_name="dedup",
        component_id=_RESEARCHER_A,
        purpose=CERT_PURPOSE_SERVER,
        san=["localhost", "127.0.0.1", "localhost"],
    )
    assert certificate_san_names(_pem(pem_file)) == ["localhost", "127.0.0.1", "::1"]


def test_certificate_issued_for_no_name_carries_none(tmp_path):
    """A node certificate is resolved by fingerprint, so it is issued for no name.

    It carries no name at all rather than a wildcard one, which matches nothing.
    """
    _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=str(tmp_path),
        certificate_name="node",
        component_id=_NODE_A,
        purpose=CERT_PURPOSE_CLIENT,
    )
    certificate = _load(pem_file)

    with pytest.raises(x509.ExtensionNotFound):
        _san(certificate)
    # Its component id still identifies it, which is how peers register it
    assert (
        certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        == _NODE_A
    )


def test_certificates_are_end_entity_not_ca(tmp_path):
    _, _, basic = _extensions(_load(_self_signed(str(tmp_path), _NODE_A)))
    assert not basic.ca


def test_server_purpose_gets_server_auth_only(tmp_path):
    certificate = _self_signed(str(tmp_path), _RESEARCHER_A, CERT_PURPOSE_SERVER)
    eku, key_usage, _ = _extensions(_load(certificate))
    assert ExtendedKeyUsageOID.SERVER_AUTH in eku
    assert ExtendedKeyUsageOID.CLIENT_AUTH not in eku
    assert key_usage.digital_signature
    assert key_usage.key_encipherment


def test_client_purpose_gets_client_auth_only(tmp_path):
    certificate = _self_signed(str(tmp_path), _NODE_A, CERT_PURPOSE_CLIENT)
    eku, key_usage, _ = _extensions(_load(certificate))
    assert ExtendedKeyUsageOID.CLIENT_AUTH in eku
    assert ExtendedKeyUsageOID.SERVER_AUTH not in eku
    assert key_usage.digital_signature
    assert not key_usage.key_encipherment


def test_unknown_purpose_is_rejected(tmp_path):
    """A certificate is only ever issued for a role the caller names."""
    with pytest.raises(FedbiomedCertificateError):
        _self_signed(str(tmp_path), _NODE_A, "both")


# -----------------------------------------------------------------------------
# What a component's own certificate declares, which follows from its type alone
# -----------------------------------------------------------------------------


def test_researcher_certificate_is_issued_for_its_host(tmp_path):
    """A researcher is verified by name, so it is issued for the host it serves on."""
    _, pem_file = generate_component_certificate(
        component_type=ComponentType.RESEARCHER.name,
        component_id=_RESEARCHER_A,
        folder=str(tmp_path),
        name="server",
        host="fbm.example.org",
        extra_san=["10.0.0.9"],
    )

    assert certificate_san_names(_pem(pem_file)) == ["fbm.example.org", "10.0.0.9"]
    assert _load(pem_file).extensions.get_extension_for_class(
        x509.ExtendedKeyUsage
    ).value == x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH])


def test_node_certificate_is_issued_for_no_host(tmp_path):
    """A node is resolved by fingerprint, so its certificate names nothing."""
    _, pem_file = generate_component_certificate(
        component_type=ComponentType.NODE.name,
        component_id=_NODE_A,
        folder=str(tmp_path),
        name="node",
    )

    assert certificate_san_names(_pem(pem_file)) == []
    assert _load(pem_file).extensions.get_extension_for_class(
        x509.ExtendedKeyUsage
    ).value == x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH])


@pytest.mark.parametrize(
    "host,extra_san", [("fbm.example.org", None), (None, ["fbm.example.org"])]
)
def test_naming_a_host_for_a_node_certificate_refused(tmp_path, host, extra_san):
    """Nothing verifies a node by name, so a name asked for would go unused."""
    with pytest.raises(FedbiomedCertificateError, match="never verified by name"):
        generate_component_certificate(
            component_type=ComponentType.NODE.name,
            component_id=_NODE_A,
            folder=str(tmp_path),
            name="node",
            host=host,
            extra_san=extra_san,
        )


def test_researcher_certificate_without_a_host_refused(tmp_path):
    """Issued for no host, it is a certificate no node would build a channel on."""
    with pytest.raises(FedbiomedCertificateError, match="has to be given"):
        generate_component_certificate(
            component_type=ComponentType.RESEARCHER.name,
            component_id=_RESEARCHER_A,
            folder=str(tmp_path),
            name="server",
        )


def test_certificate_for_something_that_is_not_a_component_type_refused(tmp_path):
    """Issuing goes through the same check opening a certificate manager does."""
    with pytest.raises(FedbiomedCertificateError, match="not a component type"):
        generate_component_certificate(
            component_type="GUI",
            component_id=_NODE_A,
            folder=str(tmp_path),
            name="node",
        )


# -----------------------------------------------------------------------------
# The module-level `generate_certificate` wrapper
# -----------------------------------------------------------------------------


def test_generate_certificate_writes_files_under_root(tmp_path):
    key_file, pem_file = generate_certificate(
        root=str(tmp_path), component_id=_NODE_A, component_type=ComponentType.NODE.name
    )
    certs_dir = os.path.join(str(tmp_path), CERTS_FOLDER_NAME)
    assert os.path.isfile(key_file)
    assert os.path.isfile(pem_file)
    assert os.path.dirname(pem_file) == certs_dir


def test_generate_certificate_aborts_when_certificates_already_exist(tmp_path):
    certs_dir = os.path.join(str(tmp_path), CERTS_FOLDER_NAME)
    os.makedirs(certs_dir)
    with open(os.path.join(certs_dir, "certificate.pem"), "w"):
        pass
    with pytest.raises(FedbiomedCertificateError):
        generate_certificate(
            root=str(tmp_path),
            component_id=_NODE_A,
            component_type=ComponentType.NODE.name,
        )


# -----------------------------------------------------------------------------
# Registration against a real database
# -----------------------------------------------------------------------------


@pytest.fixture
def cert_db(tmp_path):
    """A researcher's and a node's certificate manager, each over its own database."""
    researcher_cm = CertificateManager(
        str(tmp_path / "researcher.json"), ComponentType.RESEARCHER.name
    )
    node_cm = CertificateManager(str(tmp_path / "node.json"), ComponentType.NODE.name)

    yield SimpleNamespace(
        researcher_cm=researcher_cm, node_cm=node_cm, tmp=str(tmp_path)
    )
    researcher_cm.close()
    node_cm.close()


@pytest.mark.parametrize(
    "certificate",
    [
        "",
        "not a certificate",
        "-----BEGIN CERTIFICATE-----\nnope\n-----END CERTIFICATE-----\n",
    ],
)
def test_material_that_is_not_a_certificate_is_rejected(cert_db, certificate):
    """Every other rule passes on what it cannot read, so nothing else would stop it."""
    with pytest.raises(FedbiomedCertificateError):
        cert_db.researcher_cm.register(
            certificate=certificate,
            component_id=_NODE_A,
        )

    assert cert_db.researcher_cm.get(_NODE_A) is None


# `component_id` reconciliation against the certificate identity (`CN=`), which
# counts only on a certificate carrying `O=Fed-BioMed`. The id itself is taken as
# given, whatever its shape.


def test_recovers_component_id_from_certificate(cert_db):
    cert_db.researcher_cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _NODE_A),
    )
    assert cert_db.researcher_cm.get(_NODE_A) is not None


@pytest.mark.parametrize("component_id", ["some-other-party", "NODE_not-a-uuid"])
def test_free_form_certificate_identity_is_recovered(cert_db, component_id):
    """A `CN=` Fed-BioMed issued names a component whatever shape it has."""
    cert_db.researcher_cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, component_id),
    )
    assert cert_db.researcher_cm.get(component_id) is not None


def test_identity_ignored_when_another_issuer_signed_it(cert_db):
    """Another issuer's `CN=` names no component, even shaped like a component id.

    What stops a certificate issued elsewhere from claiming an identity: it is
    registered under the id the operator gives it, and under no other.
    """
    certificate = _third_party(cert_db.tmp, "Hospital", common_name=_NODE_A)

    with pytest.raises(FedbiomedCertificateError):
        cert_db.researcher_cm.register_certificate(
            certificate_path=certificate,
        )

    cert_db.researcher_cm.register_certificate(
        certificate_path=certificate,
        component_id=_NODE_B,
    )

    assert cert_db.researcher_cm.get(_NODE_A) is None
    assert cert_db.researcher_cm.get(_NODE_B) is not None


def test_matching_component_id_is_accepted(cert_db):
    cert_db.researcher_cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _NODE_A),
        component_id=_NODE_A,
    )
    assert cert_db.researcher_cm.get(_NODE_A) is not None


def test_conflicting_component_id_raises(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.researcher_cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, _NODE_A),
            component_id=_NODE_B,
        )


def test_component_id_required_without_usable_identity(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.researcher_cm.register_certificate(
            certificate_path=_third_party(cert_db.tmp, "Hospital A"),
        )


def test_certificate_already_registered_under_another_party_is_rejected(cert_db):
    """A certificate identifies one component, so a second cannot claim it.

    Only reachable with a third-party certificate: one embedding an identity can
    only be registered under that identity.
    """
    certificate = _third_party(cert_db.tmp, "Hospital A")
    cert_db.researcher_cm.register_certificate(
        certificate_path=certificate,
        component_id=_NODE_A,
    )

    with pytest.raises(FedbiomedCertificateError, match=_NODE_A):
        cert_db.researcher_cm.register_certificate(
            certificate_path=certificate,
            component_id=_NODE_B,
        )

    assert cert_db.researcher_cm.get(_NODE_B) is None


def test_reregistering_a_party_own_certificate_is_allowed(cert_db):
    """Renewal keeps working: the conflict is with another component, not itself."""
    certificate = _third_party(cert_db.tmp, "Hospital A")
    cert_db.researcher_cm.register_certificate(
        certificate_path=certificate,
        component_id=_NODE_A,
    )
    cert_db.researcher_cm.register_certificate(
        certificate_path=certificate,
        component_id=_NODE_A,
        upsert=True,
    )

    assert len(cert_db.researcher_cm.list()) == 1


def test_given_component_id_used_without_usable_identity(cert_db):
    cert_db.researcher_cm.register_certificate(
        certificate_path=_third_party(cert_db.tmp, "Hospital A"),
        component_id=_NODE_A,
    )
    assert cert_db.researcher_cm.get(_NODE_A) is not None


# The rules the registering component's own type carries: a node keeps a single
# registered certificate, and requires it to state a host. A researcher has none of
# its own.


def test_node_registering_researcher_certificate_accepted(cert_db):
    cert_db.node_cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER),
    )
    assert cert_db.node_cm.get(_RESEARCHER_A) is not None


@pytest.mark.parametrize(
    "component_type,component_id,purpose",
    [
        (ComponentType.RESEARCHER.name, _RESEARCHER_A, CERT_PURPOSE_SERVER),
        (ComponentType.NODE.name, _NODE_A, CERT_PURPOSE_CLIENT),
    ],
)
def test_certificate_role_is_not_read_at_registration(
    cert_db, component_type, component_id, purpose
):
    """The Extended Key Usage is descriptive: a component registers a certificate
    declaring its own role as readily as the other's."""
    cm = (
        cert_db.node_cm
        if component_type == ComponentType.NODE.name
        else cert_db.researcher_cm
    )
    cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, component_id, purpose),
    )
    assert cm.get(component_id) is not None


def test_node_registering_a_certificate_stating_no_host_rejected(cert_db):
    """gRPC would verify it against its Common Name, which holds a component id."""
    with pytest.raises(FedbiomedCertificateError, match="states no host"):
        cert_db.node_cm.register_certificate(
            certificate_path=_self_signed(
                cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER, san=()
            ),
        )

    assert cert_db.node_cm.list() == []


def test_researcher_registering_a_certificate_stating_no_host_accepted(cert_db):
    """A node is authenticated by the certificate registered for it, never by name."""
    cert_db.researcher_cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _NODE_A, san=()),
    )

    assert cert_db.researcher_cm.get(_NODE_A) is not None


def test_node_registering_second_certificate_rejected(cert_db):
    # A node communicates with a single researcher: once a certificate is
    # registered, one for another component is rejected and the database keeps
    # holding exactly one.
    cert_db.node_cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER),
    )
    with pytest.raises(FedbiomedCertificateError):
        cert_db.node_cm.register_certificate(
            certificate_path=_self_signed(
                cert_db.tmp, _RESEARCHER_B, CERT_PURPOSE_SERVER
            ),
        )
    assert len(cert_db.node_cm.list()) == 1


def test_registration_is_audited_with_the_certificate_it_trusts(cert_db):
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        cert_db.node_cm.register_certificate(
            certificate_path=_self_signed(
                cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER
            ),
        )

    events = _events(security_event, "certificate_registered")
    assert len(events) == 1
    assert events[0].kwargs["status"] == "success"
    assert events[0].kwargs["component_id"] == _RESEARCHER_A
    assert events[0].kwargs["replaced"] is False
    # The certificate is identified, never emitted.
    assert _RESEARCHER_A in events[0].kwargs["cert_subject"]
    assert "certificate" not in events[0].kwargs


def test_replacing_a_registered_certificate_is_marked_as_such(cert_db):
    cert_db.node_cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER),
    )
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        cert_db.node_cm.register_certificate(
            certificate_path=_self_signed(
                cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER, san=("other",)
            ),
            upsert=True,
        )

    events = _events(security_event, "certificate_registered")
    assert len(events) == 1
    assert events[0].kwargs["replaced"] is True


# The rejection a node can hit: a second certificate once one is registered. It
# leaves the database as it was, so it is not audited.
def test_rejected_registration_is_not_audited(cert_db):
    cert_db.node_cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER),
    )
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        with pytest.raises(FedbiomedCertificateError):
            cert_db.node_cm.register_certificate(
                certificate_path=_self_signed(
                    cert_db.tmp, _RESEARCHER_B, CERT_PURPOSE_SERVER
                ),
            )

    assert _events(security_event, "certificate_registered") == []


def test_deletion_is_audited_with_the_certificate_it_revokes(cert_db):
    cert_db.node_cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER),
    )
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        cert_db.node_cm.delete(component_id=_RESEARCHER_A)

    events = _events(security_event, "certificate_deleted")
    assert len(events) == 1
    assert events[0].kwargs["status"] == "success"
    assert events[0].kwargs["component_id"] == _RESEARCHER_A
    assert _RESEARCHER_A in events[0].kwargs["cert_subject"]


def test_deleting_an_absent_component_is_not_audited(cert_db):
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        cert_db.researcher_cm.delete(component_id=_RESEARCHER_A)

    assert _events(security_event, "certificate_deleted") == []


def test_node_reregistering_same_party_upserts(cert_db):
    # Same component id is not a second certificate: the usual upsert flow applies.
    certificate = _self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER)
    cert_db.node_cm.register_certificate(
        certificate_path=certificate,
    )
    cert_db.node_cm.register_certificate(
        certificate_path=certificate,
        upsert=True,
    )
    assert len(cert_db.node_cm.list()) == 1


def test_researcher_registering_multiple_node_certificates_accepted(cert_db):
    # The single-certificate constraint is the node's; a researcher registers
    # a certificate per node.
    for node in (_NODE_A, _NODE_C):
        cert_db.researcher_cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, node),
        )
    assert len(cert_db.researcher_cm.list()) == 2


# Validity dates, read when trust is granted rather than when a connection uses it.


def test_expired_certificate_is_rejected(cert_db):
    with pytest.raises(FedbiomedCertificateError, match="expired on"):
        cert_db.researcher_cm.register(
            certificate=_certificate(valid_days=-1).decode(),
            component_id=_NODE_A,
        )

    assert cert_db.researcher_cm.list() == []


def test_node_registering_an_expired_certificate_rejected(cert_db):
    with pytest.raises(FedbiomedCertificateError, match="expired on"):
        cert_db.node_cm.register(
            certificate=_certificate(
                san=[x509.DNSName("localhost")], valid_days=-1
            ).decode(),
            component_id=_RESEARCHER_A,
        )

    assert cert_db.node_cm.list() == []


def test_upsert_does_not_replace_a_registration_with_an_expired_certificate(cert_db):
    """What stays registered is the certificate a peer can still authenticate with."""
    registered = _certificate(valid_days=365).decode()
    cert_db.researcher_cm.register(certificate=registered, component_id=_NODE_A)

    with pytest.raises(FedbiomedCertificateError, match="expired on"):
        cert_db.researcher_cm.register(
            certificate=_certificate(valid_days=-1).decode(),
            component_id=_NODE_A,
            upsert=True,
        )

    assert cert_db.researcher_cm.get(_NODE_A)["certificate"] == registered


def test_certificate_expiring_soon_is_registered_with_a_warning(cert_db):
    with patch("fedbiomed.common.certificate_manager.logger.warning") as warning:
        cert_db.researcher_cm.register(
            certificate=_certificate(
                valid_days=CERTIFICATE_EXPIRY_WARNING_DAYS - 1
            ).decode(),
            component_id=_NODE_A,
        )

    assert cert_db.researcher_cm.get(_NODE_A) is not None
    warning.assert_called_once()
    assert _NODE_A in warning.call_args.args[0]


def test_certificate_expiring_beyond_the_window_is_registered_silently(cert_db):
    with patch("fedbiomed.common.certificate_manager.logger.warning") as warning:
        cert_db.researcher_cm.register(
            certificate=_certificate(
                valid_days=CERTIFICATE_EXPIRY_WARNING_DAYS + 1
            ).decode(),
            component_id=_NODE_A,
        )

    assert cert_db.researcher_cm.get(_NODE_A) is not None
    warning.assert_not_called()


# -----------------------------------------------------------------------------
# Mutual authentication trusted-certificate provider
# -----------------------------------------------------------------------------


@pytest.fixture
def bundle_env(tmp_path, issued_certificate):
    """Certificate database for the trusted-certificate provider tests.

    The bundle serves the researcher, which is the component registering here.
    """
    db_path = str(tmp_path / "certs.json")
    cm = CertificateManager(
        db_path=db_path, component_type=ComponentType.RESEARCHER.name
    )

    def register(component_id, pem=None, upsert=False):
        """Registers a certificate, generated when the test does not supply one."""
        pem = issued_certificate(component_id) if pem is None else pem
        cm.register(
            certificate=pem,
            component_id=component_id,
            upsert=upsert,
        )
        return pem

    yield SimpleNamespace(
        cm=cm,
        db_path=db_path,
        register=register,
        # A real (~5 year) certificate, so expiry parsing has something to read.
        real_certificate=issued_certificate,
    )
    cm.close()


def test_bundle_picks_up_hot_added_certificate(bundle_env):
    provider = TrustedCertificateBundle(bundle_env.db_path)

    pem_a = bundle_env.register(_NODE_A)
    first = provider()
    assert pem_a.encode() in first
    assert first.count(b"BEGIN CERTIFICATE") == 1

    pem_b = bundle_env.register(_NODE_B)
    second = provider()
    assert pem_a.encode() in second
    assert pem_b.encode() in second
    assert second.count(b"BEGIN CERTIFICATE") == 2


def test_bundle_kept_while_database_is_partially_written(bundle_env):
    pem_a = bundle_env.register(_NODE_A)
    provider = TrustedCertificateBundle(bundle_env.db_path)
    assert pem_a.encode() in provider()

    # TinyDB writes in place, so a read concurrent with another process
    # registering a certificate can observe a truncated file.
    with open(bundle_env.db_path) as file:
        content = file.read()
    with open(bundle_env.db_path, "w") as file:
        file.write(content[: len(content) // 2])

    assert pem_a.encode() in provider()

    with open(bundle_env.db_path, "w") as file:
        file.write(content)
    pem_b = bundle_env.register(_NODE_B)
    assert pem_b.encode() in provider()


@pytest.fixture
def bundle_expiry_env(bundle_env):
    """bundle_env with a wide expiry window and the logger captured."""
    with (
        # Generated certificates last ~5 years; widen the window so they
        # register as expiring without having to forge an expiry date.
        patch(
            "fedbiomed.common.certificate_manager.CERTIFICATE_EXPIRY_WARNING_DAYS",
            10000,
        ),
        patch("fedbiomed.common.certificate_manager.logger") as logger,
    ):
        bundle_register = bundle_env.register

        def register(*args, **kwargs):
            """Registers under the real expiry window.

            The widened one would have registration report every certificate as
            expiring, and these tests are about what a bundle read reports.
            """
            with patch(
                "fedbiomed.common.certificate_manager.CERTIFICATE_EXPIRY_WARNING_DAYS",
                CERTIFICATE_EXPIRY_WARNING_DAYS,
            ):
                return bundle_register(*args, **kwargs)

        bundle_env.register = register
        bundle_env.logger = logger
        yield bundle_env


def _warned_parties(logger):
    return [
        call.args[0]
        for call in logger.warning.call_args_list
        if "expires on" in call.args[0]
    ]


def test_expiring_certificate_is_reported_on_first_read(bundle_expiry_env):
    env = bundle_expiry_env
    env.register(_NODE_A, env.real_certificate(_NODE_A))
    provider = TrustedCertificateBundle(env.db_path)
    provider()

    warned = _warned_parties(env.logger)
    assert len(warned) == 1
    assert f"Certificate `{_NODE_A}`" in warned[0]


def test_expiring_certificate_is_registered_as_event(bundle_expiry_env):
    env = bundle_expiry_env
    env.register(_NODE_A, env.real_certificate(_NODE_A))
    provider = TrustedCertificateBundle(env.db_path)
    provider()

    events = _events(env.logger.security_event, "certificate_expiring")
    assert len(events) == 1
    assert events[0].kwargs["status"] == "warning"
    assert events[0].kwargs["component_id"] == _NODE_A


def test_unreadable_certificate_store_is_registered_as_event(bundle_expiry_env):
    """A trust store that cannot be read leaves a stale bundle in use: audited."""
    env = bundle_expiry_env
    pem_a = env.register(_NODE_A)
    provider = TrustedCertificateBundle(env.db_path)
    provider()

    with patch.object(
        CertificateManager, "list", side_effect=OSError("database is locked")
    ):
        # The previously loaded bundle is kept
        assert provider() == pem_a.encode()

    events = _events(env.logger.security_event, "certificate_store_unreadable")
    assert len(events) == 1
    assert events[0].kwargs["status"] == "warning"
    assert events[0].kwargs["db_path"] == env.db_path


def test_absent_certificate_store_reads_as_empty(bundle_expiry_env):
    """A database that was never created reads as an empty bundle, not an error:
    TinyDB creates it on read, so there is nothing to 'keep'."""
    env = bundle_expiry_env
    provider = TrustedCertificateBundle(f"{env.db_path}.missing")

    assert provider() == b""
    assert not _events(env.logger.security_event, "certificate_store_unreadable")


def test_hot_added_certificate_is_reported_on_refresh(bundle_expiry_env):
    """The gap this closes: a certificate registered after startup."""
    env = bundle_expiry_env
    env.register(_NODE_A, env.real_certificate(_NODE_A))
    provider = TrustedCertificateBundle(env.db_path)
    provider()

    env.register(_NODE_B, env.real_certificate(_NODE_B))
    provider()

    assert len(_warned_parties(env.logger)) == 2


def test_certificate_is_not_reported_twice(bundle_expiry_env):
    env = bundle_expiry_env
    env.register(_NODE_A, env.real_certificate(_NODE_A))
    provider = TrustedCertificateBundle(env.db_path)
    provider()

    # A refresh triggered by an unrelated registration must not re-report node A
    env.register(_NODE_B)
    provider()
    provider()

    warned = _warned_parties(env.logger)
    assert len([message for message in warned if _NODE_A in message]) == 1


def test_renewed_certificate_is_reported_again(bundle_expiry_env):
    """A renewal has a new expiry, so it is reported while still expiring."""
    env = bundle_expiry_env
    env.register(_NODE_A, env.real_certificate(_NODE_A))
    provider = TrustedCertificateBundle(env.db_path)

    # Certificates are generated with a fixed ~5 year validity, so a renewal
    # cannot be given a distinct expiry date here; script the dates instead.
    renewed = datetime.now(timezone.utc) + timedelta(days=20)
    with patch.object(
        CertificateManager,
        "expiring_certificates",
        side_effect=[
            [(_NODE_A, datetime.now(timezone.utc) + timedelta(days=10))],
            [(_NODE_A, renewed)],
        ],
    ):
        provider()
        env.register(_NODE_A, env.real_certificate(_NODE_A), upsert=True)
        provider()

    warned = _warned_parties(env.logger)
    assert len(warned) == 2
    assert f"{renewed:%Y-%m-%d}" in warned[1]


def _mode(path):
    """Permission bits of a file, without the file-type bits."""
    return stat.S_IMODE(os.stat(path).st_mode)


def _pair_config(folder):
    """A configuration naming a pair under `folder`, as a component's does."""
    paths = {
        ("certificate", "public_key"): os.path.join(folder, "etc", "certs", "cert.pem"),
        ("certificate", "private_key"): os.path.join(
            folder, "etc", "certs", "cert.key"
        ),
    }
    return SimpleNamespace(getpath=lambda section, key: paths[(section, key)])


def test_generated_private_key_is_readable_only_by_the_component(tmp_path):
    """The private key is the component's identity, so nobody else reads it."""
    key_file, pem_file = _generate_in(str(tmp_path))

    assert _mode(key_file) == 0o600
    # The certificate is handed to every other party, so it stays readable.
    assert _mode(pem_file) & 0o044


def test_written_pair_restricts_the_private_key(tmp_path):
    """A pair installed from elsewhere is stored under the component's own mode."""
    key_file, pem_file = _generate_in(str(tmp_path))
    os.chmod(key_file, 0o644)  # as a key supplied by a user may well arrive
    config = _pair_config(str(tmp_path))

    write_certificate_pair(config, read_file(pem_file), read_file(key_file))

    assert _mode(config.getpath("certificate", "private_key")) == 0o600
    # The supplied file is the user's own; it is read, not modified.
    assert _mode(key_file) == 0o644


def test_written_pair_restricts_the_key_it_backs_up(tmp_path):
    """A retired key is still key material, whatever mode it was written under."""
    key_file, pem_file = _generate_in(str(tmp_path))
    certificate, private_key = read_file(pem_file), read_file(key_file)
    config = _pair_config(str(tmp_path))

    write_certificate_pair(config, certificate, private_key)
    # A key issued before the mode was restricted, as an existing component holds
    os.chmod(config.getpath("certificate", "private_key"), 0o644)
    backups = write_certificate_pair(config, certificate, private_key)

    assert _mode(backups["private_key"]) == 0o600

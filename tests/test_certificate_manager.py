import ipaddress
import os
from datetime import datetime, timedelta, timezone
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
    CertificateManager,
    TrustedCertificateBundle,
    certificate_audit_fields,
    certificate_component_id,
    certificate_expiry,
    certificate_san_names,
    generate_certificate,
)
from fedbiomed.common.constants import CERTS_FOLDER_NAME
from fedbiomed.common.exceptions import FedbiomedCertificateError

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


def _self_signed(folder, component_id, purpose=CERT_PURPOSE_CLIENT, san=("localhost",)):
    """Generates a self-signed certificate, returns its PEM file path."""
    _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=folder,
        certificate_name=component_id.replace(" ", "_"),
        component_id=component_id,
        purpose=purpose,
        san=list(san),
    )
    return pem_file


def _certificate(org="Hospital", common_name=None, san=None, extended_key_usages=None):
    """A certificate not issued by Fed-BioMed, as PEM bytes.

    Subject fields, names and TLS roles are chosen freely, which is what a
    certificate issued elsewhere may combine in ways Fed-BioMed never generates.
    """
    pkey = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    attributes = [x509.NameAttribute(NameOID.ORGANIZATION_NAME, org)]
    if common_name is not None:
        attributes.append(x509.NameAttribute(NameOID.COMMON_NAME, common_name))

    name = x509.Name(attributes)
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(pkey.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
    )
    if extended_key_usages is not None:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage(extended_key_usages), critical=False
        )
    if san is not None:
        builder = builder.add_extension(
            x509.SubjectAlternativeName(san), critical=False
        )

    return builder.sign(private_key=pkey, algorithm=hashes.SHA256()).public_bytes(
        serialization.Encoding.PEM
    )


def _third_party(folder, org, extended_key_usages=None, common_name=None):
    """A certificate not issued by Fed-BioMed: arbitrary subject, chosen TLS roles.

    Carrying no `O=Fed-BioMed`, it embeds no identity, so it is the only kind a
    component id can be chosen for at registration.
    """
    roles = "_".join(oid.dotted_string for oid in extended_key_usages or []) or "no-eku"
    pem_file = os.path.join(folder, f"{org}_{roles}_{common_name}.pem")
    with open(pem_file, "wb") as file:
        file.write(
            _certificate(
                org=org,
                common_name=common_name,
                extended_key_usages=extended_key_usages,
            )
        )
    return pem_file


def _pem(pem_file):
    with open(pem_file, "rb") as f:
        return f.read()


def _load(pem_file):
    return x509.load_pem_x509_certificate(_pem(pem_file))


# -----------------------------------------------------------------------------
# CertificateManager over a real TinyDB
# -----------------------------------------------------------------------------


def test_certificate_manager_initialization(tmp_path):
    """A manager opened on a path reads and writes that database."""
    db_path = str(tmp_path / "certs.json")
    cm = CertificateManager(db_path=db_path)
    try:
        cm.register(certificate="cert", component_id=_NODE_A)
    finally:
        cm.close()

    reopened = CertificateManager(db_path=db_path)
    try:
        assert reopened.get(component_id=_NODE_A)["certificate"] == "cert"
    finally:
        reopened.close()


def test_certificate_manager_set_db_switches_database(tmp_path):
    """`set_db` moves the manager to another database, releasing the first."""
    first, second = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    cm = CertificateManager(db_path=first)
    try:
        cm.register(certificate="cert", component_id=_NODE_A)

        cm.set_db(db_path=second)
        assert cm.list() == []

        cm.register(certificate="other", component_id=_NODE_B)
        assert [d["component_id"] for d in cm.list()] == [_NODE_B]
    finally:
        cm.close()


def test_certificate_manager_get(cert_db):
    """Only the requested component is returned; an unknown one yields nothing."""
    cert_db.cm.register(certificate="cert-a", component_id=_NODE_A)
    cert_db.cm.register(certificate="cert-b", component_id=_NODE_B)

    assert cert_db.cm.get(component_id=_NODE_A)["certificate"] == "cert-a"
    assert cert_db.cm.get(component_id=_NODE_C) is None


def test_certificate_manager_registering_twice_requires_upsert(cert_db):
    """A component can be registered once; registering again needs `upsert`."""
    entry = dict(certificate="first", component_id=_NODE_A)

    cert_db.cm.register(**entry)
    assert cert_db.cm.get(component_id=_NODE_A)["certificate"] == "first"

    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register(**{**entry, "certificate": "second"})
    assert cert_db.cm.get(component_id=_NODE_A)["certificate"] == "first"

    cert_db.cm.register(**{**entry, "certificate": "second"}, upsert=True)
    assert cert_db.cm.get(component_id=_NODE_A)["certificate"] == "second"
    # Updating a component replaces its entry rather than adding one
    assert len(cert_db.cm.list()) == 1


def test_certificate_manager_delete(cert_db):
    """Deleting removes only the named component."""
    cert_db.cm.register(certificate="cert-a", component_id=_NODE_A)
    cert_db.cm.register(certificate="cert-b", component_id=_NODE_B)

    cert_db.cm.delete(component_id=_NODE_A)

    assert [d["component_id"] for d in cert_db.cm.list()] == [_NODE_B]


def test_certificate_manager_list(cert_db):
    """Tests list method of certificate manager"""
    cert_db.cm.register(certificate="cert-a", component_id=_NODE_A)

    assert [d["component_id"] for d in cert_db.cm.list()] == [_NODE_A]

    with patch("builtins.print") as mock_print:
        result = cert_db.cm.list(verbose=True)
        mock_print.assert_called_once()
    # Printing must not strip the certificate from what the caller receives
    assert result[0]["certificate"] == "cert-a"


def test_certificate_manager_register_certificate(cert_db):
    """`register_certificate` stores what the file at the given path holds."""

    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=os.path.join(cert_db.tmp, "missing.pem"),
            component_id=_NODE_A,
        )

    pem_file = _third_party(cert_db.tmp, "Hospital")
    registered = cert_db.cm.register_certificate(
        certificate_path=pem_file, component_id=_NODE_A
    )

    assert registered == _NODE_A
    with open(pem_file, encoding="UTF-8") as f:
        assert cert_db.cm.get(component_id=_NODE_A)["certificate"] == f.read()


def test_register_certificate_returns_the_recovered_component_id(cert_db):
    """The caller learns who was registered even when it supplied no component id.

    The identity normally comes from the certificate, so the return value is the
    only way to report which component a registration applied to.
    """
    registered = cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER)
    )

    assert registered == _RESEARCHER_A


def test_operations_require_initialized_database():
    """Using the manager before `set_db` is a clear error, not an AttributeError."""
    with pytest.raises(FedbiomedCertificateError):
        CertificateManager().get(_NODE_A)


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
    assert certificate_san_names(real_cert) == ["localhost", "127.0.0.1"]


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
    assert fields["cert_san"] == "localhost,127.0.0.1"
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
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, component_id)
        )

    # Generated cert lasts ~5 years: a wide window catches it, a tight one doesn't
    assert {c for c, _ in cert_db.cm.expiring_certificates(within_days=10000)} == {
        _NODE_A,
        _RESEARCHER_A,
    }
    assert cert_db.cm.expiring_certificates(within_days=1) == []


def test_list_verbose_adds_expires_column(cert_db):
    cert_db.cm.register_certificate(certificate_path=_self_signed(cert_db.tmp, _NODE_A))

    with patch("fedbiomed.common.certificate_manager.tabulate") as tabulate:
        cert_db.cm.list(verbose=True)

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


def test_ip_produces_ip_san(tmp_path):
    """An address is an `iPAddress` entry, the only place TLS reads one from."""
    certificate = _load(_self_signed(str(tmp_path), _NODE_A, san=["10.0.0.5"]))
    assert _san(certificate).get_values_for_type(x509.IPAddress) == [
        ipaddress.ip_address("10.0.0.5"),
        ipaddress.ip_address("127.0.0.1"),
    ]
    # The address is nowhere in the subject, which states who the component is
    assert certificate.subject.rfc4514_string() == f"CN={_NODE_A},O={CERT_ORGANIZATION}"


def test_named_certificate_also_carries_the_loopback_names(tmp_path):
    """A peer on the machine the component runs on dials it by a loopback name."""
    san = _san(_load(_self_signed(str(tmp_path), _NODE_A, san=["fbm-researcher"])))
    assert san.get_values_for_type(x509.DNSName) == ["fbm-researcher", "localhost"]
    assert san.get_values_for_type(x509.IPAddress) == [
        ipaddress.ip_address("127.0.0.1")
    ]


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
        "localhost",
        "127.0.0.1",
    ]


def test_names_are_not_repeated(tmp_path):
    _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=str(tmp_path),
        certificate_name="dedup",
        component_id=_RESEARCHER_A,
        purpose=CERT_PURPOSE_SERVER,
        san=["localhost", "127.0.0.1"],
    )
    assert certificate_san_names(_pem(pem_file)) == ["localhost", "127.0.0.1"]


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
# The module-level `generate_certificate` wrapper
# -----------------------------------------------------------------------------


def test_generate_certificate_writes_files_under_root(tmp_path):
    key_file, pem_file = generate_certificate(
        root=str(tmp_path), component_id=_NODE_A, purpose=CERT_PURPOSE_CLIENT
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
            root=str(tmp_path), component_id=_NODE_A, purpose=CERT_PURPOSE_CLIENT
        )


# -----------------------------------------------------------------------------
# Registration against a real database
# -----------------------------------------------------------------------------


@pytest.fixture
def cert_db(tmp_path):
    """Real CertificateManager over a temporary database."""
    cm = CertificateManager(db_path=str(tmp_path / "certs.json"))
    yield SimpleNamespace(cm=cm, tmp=str(tmp_path))
    cm.close()


# `component_id` reconciliation against the certificate identity (`CN=`), which
# counts only on a certificate carrying `O=Fed-BioMed`. The id itself is taken as
# given, whatever its shape.


def test_recovers_component_id_from_certificate(cert_db):
    cert_db.cm.register_certificate(certificate_path=_self_signed(cert_db.tmp, _NODE_A))
    assert cert_db.cm.get(_NODE_A) is not None


@pytest.mark.parametrize("component_id", ["some-other-party", "NODE_not-a-uuid"])
def test_free_form_certificate_identity_is_recovered(cert_db, component_id):
    """A `CN=` Fed-BioMed issued names a component whatever shape it has."""
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, component_id)
    )
    assert cert_db.cm.get(component_id) is not None


def test_identity_ignored_when_another_issuer_signed_it(cert_db):
    """Another issuer's `CN=` names no component, even shaped like a component id.

    What stops a certificate issued elsewhere from claiming an identity: it is
    registered under the id the operator gives it, and under no other.
    """
    certificate = _third_party(cert_db.tmp, "Hospital", common_name=_NODE_A)

    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(certificate_path=certificate)

    cert_db.cm.register_certificate(certificate_path=certificate, component_id=_NODE_B)

    assert cert_db.cm.get(_NODE_A) is None
    assert cert_db.cm.get(_NODE_B) is not None


def test_matching_component_id_is_accepted(cert_db):
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _NODE_A), component_id=_NODE_A
    )
    assert cert_db.cm.get(_NODE_A) is not None


def test_conflicting_component_id_raises(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, _NODE_A), component_id=_NODE_B
        )


def test_component_id_required_without_usable_identity(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_third_party(cert_db.tmp, "Hospital A")
        )


def test_certificate_already_registered_under_another_party_is_rejected(cert_db):
    """A certificate identifies one component, so a second cannot claim it.

    Only reachable with a third-party certificate: one embedding an identity can
    only be registered under that identity.
    """
    certificate = _third_party(cert_db.tmp, "Hospital A")
    cert_db.cm.register_certificate(certificate_path=certificate, component_id=_NODE_A)

    with pytest.raises(FedbiomedCertificateError, match=_NODE_A):
        cert_db.cm.register_certificate(
            certificate_path=certificate, component_id=_NODE_B
        )

    assert cert_db.cm.get(_NODE_B) is None


def test_reregistering_a_party_own_certificate_is_allowed(cert_db):
    """Renewal keeps working: the conflict is with another component, not itself."""
    certificate = _third_party(cert_db.tmp, "Hospital A")
    cert_db.cm.register_certificate(certificate_path=certificate, component_id=_NODE_A)
    cert_db.cm.register_certificate(
        certificate_path=certificate, component_id=_NODE_A, upsert=True
    )

    assert len(cert_db.cm.list()) == 1


def test_given_component_id_used_without_usable_identity(cert_db):
    cert_db.cm.register_certificate(
        certificate_path=_third_party(cert_db.tmp, "Hospital A"), component_id=_NODE_A
    )
    assert cert_db.cm.get(_NODE_A) is not None


# The TLS role a registered certificate must carry: the one the registering
# component does not act in. A certificate restricted to the registrar's own role
# is rejected; one leaving the role open is registered. A TLS client additionally
# keeps a single registered certificate. Omitting the registering purpose skips
# both checks.


def test_node_registering_researcher_certificate_accepted(cert_db):
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER),
        registering_purpose=CERT_PURPOSE_CLIENT,
    )
    assert cert_db.cm.get(_RESEARCHER_A) is not None


def test_researcher_registering_node_certificate_accepted(cert_db):
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _NODE_A),
        registering_purpose=CERT_PURPOSE_SERVER,
    )
    assert cert_db.cm.get(_NODE_A) is not None


def test_node_registering_node_certificate_rejected(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, _NODE_A),
            registering_purpose=CERT_PURPOSE_CLIENT,
        )


def test_researcher_registering_researcher_certificate_rejected(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(
                cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER
            ),
            registering_purpose=CERT_PURPOSE_SERVER,
        )


def test_unknown_registering_purpose_is_rejected(cert_db):
    """A purpose that names no TLS role checks nothing, so it is refused.

    Only `client` and `server` name a role; anything else would pass every
    certificate through unchecked.
    """
    with pytest.raises(FedbiomedCertificateError, match="Unknown registering purpose"):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, _NODE_A),
            registering_purpose="NODE",
        )

    assert cert_db.cm.list() == []


def test_component_id_does_not_exempt_an_own_role_certificate(cert_db):
    # Only the EKU decides: a node component id does not make a server-only
    # certificate registrable on a researcher.
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_third_party(
                cert_db.tmp, "Hospital_x", [ExtendedKeyUsageOID.SERVER_AUTH]
            ),
            component_id=_NODE_A,
            registering_purpose=CERT_PURPOSE_SERVER,
        )


@pytest.mark.parametrize(
    "extended_key_usages",
    [
        None,  # no Extended Key Usage at all
        [ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH],
    ],
)
def test_open_role_third_party_is_registered(cert_db, extended_key_usages):
    """Only a certificate restricted to the registrar's own role is refused."""
    cert_db.cm.register_certificate(
        certificate_path=_third_party(cert_db.tmp, "Hospital_x", extended_key_usages),
        component_id=_NODE_A,
        registering_purpose=CERT_PURPOSE_SERVER,
    )

    assert cert_db.cm.get(_NODE_A) is not None


def test_node_registering_second_certificate_rejected(cert_db):
    # A node communicates with a single researcher: once a certificate is
    # registered, one for another component is rejected and the database keeps
    # holding exactly one.
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER),
        registering_purpose=CERT_PURPOSE_CLIENT,
    )
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(
                cert_db.tmp, _RESEARCHER_B, CERT_PURPOSE_SERVER
            ),
            registering_purpose=CERT_PURPOSE_CLIENT,
        )
    assert len(cert_db.cm.list()) == 1


def test_registration_is_audited_with_the_certificate_it_trusts(cert_db):
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(
                cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER
            ),
            registering_purpose=CERT_PURPOSE_CLIENT,
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
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER),
        registering_purpose=CERT_PURPOSE_CLIENT,
    )
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(
                cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER, san=("other",)
            ),
            registering_purpose=CERT_PURPOSE_CLIENT,
            upsert=True,
        )

    events = _events(security_event, "certificate_registered")
    assert len(events) == 1
    assert events[0].kwargs["replaced"] is True


# Both rejections a node can hit: a certificate restricted to its own TLS role,
# and a second certificate once one is registered. Neither changes the database,
# so neither is audited.
@pytest.mark.parametrize(
    "preregister,component_id,purpose",
    [
        (None, _NODE_A, CERT_PURPOSE_CLIENT),
        (_RESEARCHER_A, _RESEARCHER_B, CERT_PURPOSE_SERVER),
    ],
)
def test_rejected_registration_is_not_audited(
    cert_db, preregister, component_id, purpose
):
    if preregister:
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, preregister, purpose),
            registering_purpose=CERT_PURPOSE_CLIENT,
        )
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        with pytest.raises(FedbiomedCertificateError):
            cert_db.cm.register_certificate(
                certificate_path=_self_signed(cert_db.tmp, component_id, purpose),
                registering_purpose=CERT_PURPOSE_CLIENT,
            )

    assert _events(security_event, "certificate_registered") == []


def test_deletion_is_audited_with_the_certificate_it_revokes(cert_db):
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER),
        registering_purpose=CERT_PURPOSE_CLIENT,
    )
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        cert_db.cm.delete(component_id=_RESEARCHER_A)

    events = _events(security_event, "certificate_deleted")
    assert len(events) == 1
    assert events[0].kwargs["status"] == "success"
    assert events[0].kwargs["component_id"] == _RESEARCHER_A
    assert _RESEARCHER_A in events[0].kwargs["cert_subject"]


def test_deleting_an_absent_component_is_not_audited(cert_db):
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        cert_db.cm.delete(component_id=_RESEARCHER_A)

    assert _events(security_event, "certificate_deleted") == []


def test_node_reregistering_same_party_upserts(cert_db):
    # Same component id is not a second certificate: the usual upsert flow applies.
    certificate = _self_signed(cert_db.tmp, _RESEARCHER_A, CERT_PURPOSE_SERVER)
    cert_db.cm.register_certificate(
        certificate_path=certificate,
        registering_purpose=CERT_PURPOSE_CLIENT,
    )
    cert_db.cm.register_certificate(
        certificate_path=certificate,
        registering_purpose=CERT_PURPOSE_CLIENT,
        upsert=True,
    )
    assert len(cert_db.cm.list()) == 1


def test_researcher_registering_multiple_node_certificates_accepted(cert_db):
    # The single-certificate constraint is the node's; a researcher registers
    # a certificate per node.
    for node in (_NODE_A, _NODE_C):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, node),
            registering_purpose=CERT_PURPOSE_SERVER,
        )
    assert len(cert_db.cm.list()) == 2


def test_omitted_registering_purpose_skips_the_role_checks(cert_db):
    # Naming no role states none to check against: a node certificate registers.
    cert_db.cm.register_certificate(certificate_path=_self_signed(cert_db.tmp, _NODE_A))
    assert cert_db.cm.get(_NODE_A) is not None


# -----------------------------------------------------------------------------
# Mutual authentication trusted-certificate provider
# -----------------------------------------------------------------------------


@pytest.fixture
def bundle_env(tmp_path):
    """Certificate database for the trusted-certificate provider tests."""
    db_path = str(tmp_path / "certs.json")
    cm = CertificateManager(db_path=db_path)

    def register(component_id, pem, upsert=False):
        cm.register(certificate=pem, component_id=component_id, upsert=upsert)

    def real_certificate(component_id):
        """A real (~5 year) certificate, so expiry parsing has something to read."""
        pem_file = _self_signed(str(tmp_path), component_id)
        with open(pem_file) as file:
            return file.read()

    yield SimpleNamespace(
        cm=cm,
        db_path=db_path,
        register=register,
        real_certificate=real_certificate,
    )
    cm.close()


def test_bundle_picks_up_hot_added_certificate(bundle_env):
    provider = TrustedCertificateBundle(bundle_env.db_path)

    bundle_env.register(_NODE_A, "PEM-1")
    first = provider()
    assert b"PEM-1" in first
    assert first.count(b"PEM") == 1

    bundle_env.register(_NODE_B, "PEM-2")
    second = provider()
    assert b"PEM-1" in second
    assert b"PEM-2" in second
    assert second.count(b"PEM") == 2


def test_bundle_does_not_reread_when_unchanged(bundle_env):
    bundle_env.register(_NODE_A, "PEM-1")
    provider = TrustedCertificateBundle(bundle_env.db_path)
    provider()

    with patch("fedbiomed.common.certificate_manager.CertificateManager") as cm_cls:
        provider()
        cm_cls.assert_not_called()


def test_bundle_kept_while_database_is_partially_written(bundle_env):
    bundle_env.register(_NODE_A, "PEM-1")
    provider = TrustedCertificateBundle(bundle_env.db_path)
    assert b"PEM-1" in provider()

    # TinyDB writes in place, so a read concurrent with another process
    # registering a certificate can observe a truncated file.
    with open(bundle_env.db_path) as file:
        content = file.read()
    with open(bundle_env.db_path, "w") as file:
        file.write(content[: len(content) // 2])

    assert b"PEM-1" in provider()

    with open(bundle_env.db_path, "w") as file:
        file.write(content)
    bundle_env.register(_NODE_B, "PEM-2")
    assert b"PEM-2" in provider()


def test_bundle_kept_when_database_is_missing(bundle_env):
    bundle_env.register(_NODE_A, "PEM-1")
    provider = TrustedCertificateBundle(bundle_env.db_path)
    assert b"PEM-1" in provider()

    os.remove(bundle_env.db_path)
    assert b"PEM-1" in provider()


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
    env.register(_NODE_A, "PEM-1")
    provider = TrustedCertificateBundle(env.db_path)
    provider()

    with patch(
        "fedbiomed.common.certificate_manager.os.stat",
        side_effect=OSError("database is locked"),
    ):
        # The previously loaded bundle is kept
        assert provider() == b"PEM-1"

    events = _events(env.logger.security_event, "certificate_store_unreadable")
    assert len(events) == 1
    assert events[0].kwargs["status"] == "warning"
    assert events[0].kwargs["db_path"] == env.db_path


def test_never_read_certificate_store_reports_no_certificate_available(
    bundle_expiry_env,
):
    """A database never read holds nothing to keep, and must not claim it does."""
    env = bundle_expiry_env
    provider = TrustedCertificateBundle(f"{env.db_path}.missing")

    assert provider() == b""
    assert not provider.loaded

    warning = env.logger.warning.call_args.args[0]
    assert "No certificate is available" in warning
    assert "Keeping" not in warning


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
    env.register(_NODE_B, "PEM-2")
    provider()
    provider()

    warned = _warned_parties(env.logger)
    assert len(warned) == 1
    assert _NODE_A in warned[0]


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

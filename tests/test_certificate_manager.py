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
    CertificateManager,
    TrustedCertificateBundle,
    certificate_audit_fields,
    certificate_expiry,
    generate_certificate,
)
from fedbiomed.common.constants import CERTS_FOLDER_NAME, ComponentType
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


def _self_signed(folder, org, cn="localhost", with_org_subject=True):
    """Generates a self-signed certificate, returns its PEM file path."""
    subject = {"CommonName": cn}
    if with_org_subject:
        subject["OrganizationName"] = org
    _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=folder,
        certificate_name=org.replace(" ", "_"),
        component_id=org,
        subject=subject,
    )
    return pem_file


def _third_party(folder, org, extended_key_usages=None):
    """A certificate not issued by Fed-BioMed: arbitrary `O=`, chosen TLS roles.

    Fed-BioMed derives the role from the component id, so a certificate whose
    role contradicts the party it is registered as can only come from outside.
    """
    pkey = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, org)])
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

    pem_file = os.path.join(folder, f"{org}_{extended_key_usages}.pem")
    with open(pem_file, "wb") as file:
        file.write(
            builder.sign(private_key=pkey, algorithm=hashes.SHA256()).public_bytes(
                serialization.Encoding.PEM
            )
        )
    return pem_file


def _load(pem_file):
    with open(pem_file, "rb") as f:
        return x509.load_pem_x509_certificate(f.read())


# -----------------------------------------------------------------------------
# CertificateManager over a real TinyDB
# -----------------------------------------------------------------------------


def test_certificate_manager_initialization(tmp_path):
    """A manager opened on a path reads and writes that database."""
    db_path = str(tmp_path / "certs.json")
    cm = CertificateManager(db_path=db_path)
    try:
        cm.insert(certificate="cert", party_id="NODE_a", component="NODE")
    finally:
        cm.close()

    reopened = CertificateManager(db_path=db_path)
    try:
        assert reopened.get(party_id="NODE_a")["certificate"] == "cert"
    finally:
        reopened.close()


def test_certificate_manager_set_db_switches_database(tmp_path):
    """`set_db` moves the manager to another database, releasing the first."""
    first, second = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    cm = CertificateManager(db_path=first)
    try:
        cm.insert(certificate="cert", party_id="NODE_a", component="NODE")

        cm.set_db(db_path=second)
        assert cm.list() == []

        cm.insert(certificate="other", party_id="NODE_b", component="NODE")
        assert [d["party_id"] for d in cm.list()] == ["NODE_b"]
    finally:
        cm.close()


def test_certificate_manager_get(cert_db):
    """Only the requested party is returned; an unknown one yields nothing."""
    cert_db.cm.insert(certificate="cert-a", party_id="NODE_a", component="NODE")
    cert_db.cm.insert(certificate="cert-b", party_id="NODE_b", component="NODE")

    assert cert_db.cm.get(party_id="NODE_a")["certificate"] == "cert-a"
    assert cert_db.cm.get(party_id="NODE_missing") is None


def test_certificate_manager_get_by_component(cert_db):
    """Only certificates of the requested component type are returned."""
    cert_db.cm.insert(certificate="node-cert", party_id="NODE_a", component="NODE")
    cert_db.cm.insert(
        certificate="researcher-cert", party_id="RESEARCHER_a", component="RESEARCHER"
    )

    assert cert_db.cm.get_by_component("NODE") == ["node-cert"]
    assert cert_db.cm.get_by_component("RESEARCHER") == ["researcher-cert"]


def test_certificate_manager_get_by_component_empty(cert_db):
    """Tests component lookup with no registered certificates"""
    assert cert_db.cm.get_by_component("NODE") == []


def test_certificate_manager_insert(cert_db):
    """A party can be registered once; registering again needs `upsert`."""
    entry = dict(certificate="first", party_id="NODE_a", component="NODE")

    cert_db.cm.insert(**entry)
    assert cert_db.cm.get(party_id="NODE_a")["certificate"] == "first"

    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.insert(**{**entry, "certificate": "second"})
    assert cert_db.cm.get(party_id="NODE_a")["certificate"] == "first"

    cert_db.cm.insert(**{**entry, "certificate": "second"}, upsert=True)
    assert cert_db.cm.get(party_id="NODE_a")["certificate"] == "second"
    # Updating a party replaces its entry rather than adding one
    assert len(cert_db.cm.list()) == 1


def test_certificate_manager_delete(cert_db):
    """Deleting removes only the named party."""
    cert_db.cm.insert(certificate="cert-a", party_id="NODE_a", component="NODE")
    cert_db.cm.insert(certificate="cert-b", party_id="NODE_b", component="NODE")

    cert_db.cm.delete(party_id="NODE_a")

    assert [d["party_id"] for d in cert_db.cm.list()] == ["NODE_b"]


def test_certificate_manager_list(cert_db):
    """Tests list method of certificate manager"""
    cert_db.cm.insert(certificate="cert-a", party_id="NODE_a", component="NODE")

    assert [d["party_id"] for d in cert_db.cm.list()] == ["NODE_a"]

    with patch("builtins.print") as mock_print:
        result = cert_db.cm.list(verbose=True)
        mock_print.assert_called_once()
    # Printing must not strip the certificate from what the caller receives
    assert result[0]["certificate"] == "cert-a"


@pytest.mark.parametrize(
    "party_id,component",
    [
        ("node_4f2c8a10-0e7d-4a11-9c33-8b7f0a1d2e44", "NODE"),
        ("researcher_9c2b1d70-1111-2222-3333-444455556666", "RESEARCHER"),
    ],
)
def test_certificate_manager_register_certificate(cert_db, party_id, component):
    """A registered certificate is stored under its inferred component"""

    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=os.path.join(cert_db.tmp, "missing.pem"),
            party_id=party_id,
        )

    # No `O=` identity, so the given party id decides how it is classified
    pem_file = _self_signed(cert_db.tmp, "Hospital", with_org_subject=False)
    registered = cert_db.cm.register_certificate(
        certificate_path=pem_file, party_id=party_id
    )

    assert registered == party_id
    entry = cert_db.cm.get(party_id=party_id)
    assert entry["component"] == component
    with open(pem_file, encoding="UTF-8") as f:
        assert entry["certificate"] == f.read()


def test_register_certificate_returns_the_recovered_party_id(cert_db):
    """The caller learns who was registered even when it supplied no party id.

    The identity normally comes from the certificate, so the return value is the
    only way to report which party a registration applied to.
    """
    registered = cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A)
    )

    assert registered == _RESEARCHER_A


def test_certificate_manager_write_certificate_file(cert_db):
    path = os.path.join(cert_db.tmp, "written.pem")
    CertificateManager._write_certificate_file(path, "Certificate")

    with open(path, encoding="UTF-8") as f:
        assert f.read() == "Certificate"


def test_certificate_manager_write_certificate_file_unwritable(cert_db):
    """A path that cannot be written is reported as a certificate error."""
    with pytest.raises(FedbiomedCertificateError):
        CertificateManager._write_certificate_file(
            os.path.join(cert_db.tmp, "no-such-dir", "written.pem"), "Certificate"
        )


def test_operations_require_initialized_database():
    """Using the manager before `set_db` is a clear error, not an AttributeError."""
    with pytest.raises(FedbiomedCertificateError):
        CertificateManager().get("NODE_1")


def _generate_in(certificate_folder):
    return CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=certificate_folder,
        certificate_name="certificate",
        component_id="component-id",
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
    pem_file = _self_signed(str(tmp_path), "node_1")
    with open(pem_file, "rb") as f:
        return f.read()


def test_certificate_expiry_returns_future_date(real_cert):
    expiry = certificate_expiry(real_cert)
    assert isinstance(expiry, datetime)
    assert expiry > datetime.now(timezone.utc)


def test_certificate_expiry_none_for_unparsable():
    assert certificate_expiry(b"not a certificate") is None


def test_certificate_audit_fields_identify_the_certificate(real_cert):
    fields = certificate_audit_fields(real_cert)
    assert fields["cert_subject"] == "O=node_1,CN=localhost"
    assert fields["cert_issuer"] == "O=node_1,CN=localhost"
    assert fields["cert_san"] == "localhost"
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


def test_expiring_certificates_filters_by_threshold_and_component(real_cert):
    cm = CertificateManager()
    docs = [
        {"certificate": real_cert.decode(), "party_id": "node_1", "component": "NODE"},
        {
            "certificate": real_cert.decode(),
            "party_id": "res_1",
            "component": "RESEARCHER",
        },
    ]
    cm._db = MagicMock()
    cm._db.all.return_value = docs

    # Generated cert lasts ~5 years: a wide window catches it, a tight one doesn't
    wide = cm.expiring_certificates(within_days=10000, component="NODE")
    assert [p for p, _ in wide] == ["node_1"]
    assert cm.expiring_certificates(within_days=1, component="NODE") == []
    # Component filter excludes the researcher entry
    assert [p for p, _ in cm.expiring_certificates(within_days=10000)] == [
        "node_1",
        "res_1",
    ]


def test_list_verbose_adds_expires_column(real_cert):
    cm = CertificateManager()
    cm._db = MagicMock()
    cm._db.all.return_value = [
        {"certificate": real_cert.decode(), "party_id": "node_1", "component": "NODE"}
    ]
    with patch("fedbiomed.common.certificate_manager.tabulate") as tabulate:
        cm.list(verbose=True)
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


def test_subject_carries_common_and_organization_name(tmp_path):
    cert = _load(_self_signed(str(tmp_path), "node_1"))
    assert (
        cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
        == "localhost"
    )
    assert (
        cert.subject.get_attributes_for_oid(x509.oid.NameOID.ORGANIZATION_NAME)[0].value
        == "node_1"
    )


def test_hostname_common_name_produces_dns_san(tmp_path):
    san = _san(_load(_self_signed(str(tmp_path), "node_1", cn="localhost")))
    assert san.get_values_for_type(x509.DNSName) == ["localhost"]


def test_ip_common_name_produces_ip_san(tmp_path):
    san = _san(_load(_self_signed(str(tmp_path), "node_1", cn="10.0.0.5")))
    assert san.get_values_for_type(x509.IPAddress) == [ipaddress.ip_address("10.0.0.5")]


def test_wildcard_common_name_has_no_san(tmp_path):
    # `*` is neither a resolvable host nor an IP -> no SubjectAlternativeName
    with pytest.raises(x509.ExtensionNotFound):
        _san(_load(_self_signed(str(tmp_path), "node_1", cn="*")))


def test_certificates_are_end_entity_not_ca(tmp_path):
    _, _, basic = _extensions(_load(_self_signed(str(tmp_path), "node_1")))
    assert not basic.ca


def test_researcher_id_gets_server_auth_only(tmp_path):
    eku, key_usage, _ = _extensions(_load(_self_signed(str(tmp_path), "RESEARCHER_1")))
    assert ExtendedKeyUsageOID.SERVER_AUTH in eku
    assert ExtendedKeyUsageOID.CLIENT_AUTH not in eku
    assert key_usage.digital_signature
    assert key_usage.key_encipherment


def test_node_id_gets_client_auth_only(tmp_path):
    eku, key_usage, _ = _extensions(_load(_self_signed(str(tmp_path), "NODE_1")))
    assert ExtendedKeyUsageOID.CLIENT_AUTH in eku
    assert ExtendedKeyUsageOID.SERVER_AUTH not in eku
    assert key_usage.digital_signature
    assert not key_usage.key_encipherment


def test_unrecognized_id_gets_both_roles(tmp_path):
    eku, _, _ = _extensions(_load(_self_signed(str(tmp_path), "some-other-party")))
    assert ExtendedKeyUsageOID.SERVER_AUTH in eku
    assert ExtendedKeyUsageOID.CLIENT_AUTH in eku


# -----------------------------------------------------------------------------
# The module-level `generate_certificate` wrapper
# -----------------------------------------------------------------------------


def test_generate_certificate_writes_files_under_root(tmp_path):
    key_file, pem_file = generate_certificate(root=str(tmp_path), component_id=_NODE_A)
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
        generate_certificate(root=str(tmp_path), component_id=_NODE_A)


# -----------------------------------------------------------------------------
# Registration against a real database
# -----------------------------------------------------------------------------


@pytest.fixture
def cert_db(tmp_path):
    """Real CertificateManager over a temporary database."""
    cm = CertificateManager(db_path=str(tmp_path / "certs.json"))
    yield SimpleNamespace(cm=cm, tmp=str(tmp_path))
    cm.close()


def _register_own(env, party_id):
    """Registers a certificate whose `O=` is its own party id."""
    env.cm.register_certificate(
        certificate_path=_self_signed(env.tmp, party_id), party_id=party_id
    )


# Component classification of registered certificates. Component ids are
# `<COMPONENT_TYPE>_<uuid>` (see `Config.generate`), so these use that real
# shape: classifying a node certificate as `RESEARCHER` leaves the researcher's
# mutual-TLS trust bundle empty.


def test_node_id_registers_as_node_component(cert_db):
    _register_own(cert_db, _NODE_A)
    assert len(cert_db.cm.get_by_component(ComponentType.NODE.name)) == 1
    assert len(cert_db.cm.get_by_component(ComponentType.RESEARCHER.name)) == 0


def test_researcher_id_registers_as_researcher_component(cert_db):
    _register_own(cert_db, _RESEARCHER_A)
    assert len(cert_db.cm.get_by_component(ComponentType.RESEARCHER.name)) == 1
    assert len(cert_db.cm.get_by_component(ComponentType.NODE.name)) == 0


def test_lowercase_node_id_registers_as_node_component(cert_db):
    """Ids from older lowercase-prefixed deployments keep classifying as NODE."""
    _register_own(cert_db, _NODE_A.lower())
    assert len(cert_db.cm.get_by_component(ComponentType.NODE.name)) == 1


@pytest.mark.parametrize(
    "party_id",
    [
        "some-other-party",  # unprefixed
        "NODE_not-a-uuid",  # non-uuid
        "ADMIN_4f2c8a10-0e7d-4a11-9c33-8b7f0a1d2e44",  # unknown component prefix
    ],
)
def test_invalid_party_id_is_rejected(cert_db, party_id):
    with pytest.raises(FedbiomedCertificateError):
        _register_own(cert_db, party_id)


# `party_id` reconciliation against the certificate identity (`O=`). A `O=`
# that is not a valid party id is treated as no usable identity, like an absent
# one (the absent case is covered by the register test above).


def test_recovers_party_id_from_certificate(cert_db):
    cert_db.cm.register_certificate(certificate_path=_self_signed(cert_db.tmp, _NODE_A))
    assert cert_db.cm.get(_NODE_A) is not None
    assert len(cert_db.cm.get_by_component(ComponentType.NODE.name)) == 1


def test_matching_party_id_is_accepted(cert_db):
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _NODE_A), party_id=_NODE_A
    )
    assert cert_db.cm.get(_NODE_A) is not None


def test_conflicting_party_id_raises(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, _NODE_A), party_id=_NODE_B
        )


def test_party_id_required_without_usable_identity(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, "Hospital A")
        )


def test_certificate_already_registered_under_another_party_is_rejected(cert_db):
    """A certificate identifies one party, so a second party cannot claim it.

    Only reachable with a third-party certificate: one embedding a valid identity
    can only be registered under that identity.
    """
    certificate = _self_signed(cert_db.tmp, "Hospital A")
    cert_db.cm.register_certificate(certificate_path=certificate, party_id=_NODE_A)

    with pytest.raises(FedbiomedCertificateError, match=_NODE_A):
        cert_db.cm.register_certificate(certificate_path=certificate, party_id=_NODE_B)

    assert cert_db.cm.get(_NODE_B) is None


def test_reregistering_a_party_own_certificate_is_allowed(cert_db):
    """Renewal keeps working: the conflict is with another party, not itself."""
    certificate = _self_signed(cert_db.tmp, "Hospital A")
    cert_db.cm.register_certificate(certificate_path=certificate, party_id=_NODE_A)
    cert_db.cm.register_certificate(
        certificate_path=certificate, party_id=_NODE_A, upsert=True
    )

    assert len(cert_db.cm.list()) == 1


def test_given_party_id_used_without_usable_identity(cert_db):
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, "Hospital A"), party_id=_NODE_A
    )
    assert cert_db.cm.get(_NODE_A) is not None


def test_near_miss_identity_treated_as_third_party(cert_db):
    # An `O=` resembling a party id but failing the pattern is no usable
    # identity: the certificate registers as third-party under the given
    # party id instead of being rejected for a mismatch.
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, "NODE_not-a-uuid"),
        party_id=_NODE_A,
    )
    assert cert_db.cm.get(_NODE_A)["component"] == ComponentType.NODE.name


def test_given_party_id_must_follow_pattern(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, "Hospital A"),
            party_id="not-a-valid-id",
        )


def test_malformed_party_id_rejected_even_with_certificate_identity(cert_db):
    # The certificate embeds a valid identity, but a provided party id must
    # still follow the expected pattern rather than pass unchecked.
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, _NODE_A),
            party_id="NODE_garbage",
        )


def test_wrong_component_party_id_rejected(cert_db):
    # Valid pattern but the wrong component: a researcher party id given for a
    # node certificate must not be reconciled.
    researcher = "RESEARCHER_" + _NODE_A.split("_", 1)[1]
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, _NODE_A), party_id=researcher
        )


# Rejection of certificates of the registering component's own kind. A node
# registers researcher certificates and a researcher node ones. A certificate
# is rejected when the party id it registers under or a single-role EKU
# identifies it as the registrar's own type; a missing identity or EKU
# constrains nothing, as does omitting the registering component. A node
# additionally keeps a single registered certificate — its researcher's.


def test_node_registering_researcher_certificate_accepted(cert_db):
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A),
        registering_component=ComponentType.NODE.name,
    )
    assert cert_db.cm.get(_RESEARCHER_A) is not None


def test_researcher_registering_node_certificate_accepted(cert_db):
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _NODE_A),
        registering_component=ComponentType.RESEARCHER.name,
    )
    assert cert_db.cm.get(_NODE_A) is not None


def test_node_registering_node_certificate_rejected(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, _NODE_A),
            registering_component=ComponentType.NODE.name,
        )


def test_researcher_registering_researcher_certificate_rejected(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A),
            registering_component=ComponentType.RESEARCHER.name,
        )


def test_given_party_id_of_own_type_rejected(cert_db):
    # The party id is user-given for a third-party certificate (arbitrary
    # `O=`, dual-role EKU): it goes through the same protection.
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, "Hospital A"),
            party_id=_NODE_A,
            registering_component=ComponentType.NODE.name,
        )


def test_researcher_registering_server_only_third_party_rejected(cert_db):
    # EKU restricts the certificate to the researcher's own role (server),
    # even though `O=` carries no party id.
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_third_party(
                cert_db.tmp, "Hospital_x", [ExtendedKeyUsageOID.SERVER_AUTH]
            ),
            party_id=_NODE_A,
            registering_component=ComponentType.RESEARCHER.name,
        )


def test_node_registering_client_only_third_party_rejected(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_third_party(
                cert_db.tmp, "Hospital_x", [ExtendedKeyUsageOID.CLIENT_AUTH]
            ),
            party_id=_RESEARCHER_A,
            registering_component=ComponentType.NODE.name,
        )


def test_dual_role_third_party_accepted(cert_db):
    # A dual-role EKU does not identify a component, so it constrains nothing.
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, "Hospital A"),
        party_id=_NODE_A,
        registering_component=ComponentType.RESEARCHER.name,
    )
    assert cert_db.cm.get(_NODE_A) is not None


def test_missing_eku_constrains_nothing(cert_db):
    # A certificate without any EKU carries no role to check against.
    cert_db.cm.register_certificate(
        certificate_path=_third_party(cert_db.tmp, "Hospital_x"),
        party_id=_NODE_A,
        registering_component=ComponentType.RESEARCHER.name,
    )
    assert cert_db.cm.get(_NODE_A) is not None


def test_node_registering_second_certificate_rejected(cert_db):
    # A node communicates with a single researcher: once a certificate is
    # registered, one for another party is rejected and the database keeps
    # holding exactly one.
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A),
        registering_component=ComponentType.NODE.name,
    )
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_B),
            registering_component=ComponentType.NODE.name,
        )
    assert len(cert_db.cm.list()) == 1


# Both rejections a node can hit: a certificate of its own type, and a second
# certificate once one is registered.
@pytest.mark.parametrize(
    "preregister,party_id", [(None, _NODE_A), (_RESEARCHER_A, _RESEARCHER_B)]
)
def test_registration_rejection_is_registered_as_event(cert_db, preregister, party_id):
    if preregister:
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, preregister),
            registering_component=ComponentType.NODE.name,
        )
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        with pytest.raises(FedbiomedCertificateError):
            cert_db.cm.register_certificate(
                certificate_path=_self_signed(cert_db.tmp, party_id),
                registering_component=ComponentType.NODE.name,
            )

    events = _events(security_event, "certificate_registration_rejected")
    assert len(events) == 1
    assert events[0].kwargs["status"] == "failure"
    assert events[0].kwargs["party_id"] == party_id
    assert events[0].kwargs["registering_component"] == ComponentType.NODE.name


def test_accepted_registration_is_not_rejected_event(cert_db):
    """Successful inserts are audited by the DBTable wrapper, not by this path."""
    with patch(
        "fedbiomed.common.certificate_manager.logger.security_event"
    ) as security_event:
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A),
            registering_component=ComponentType.NODE.name,
        )

    assert _events(security_event, "certificate_registration_rejected") == []


def test_node_reregistering_same_party_upserts(cert_db):
    # Same party id is not a second certificate: the usual upsert flow applies.
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A),
        registering_component=ComponentType.NODE.name,
    )
    cert_db.cm.register_certificate(
        certificate_path=_self_signed(cert_db.tmp, _RESEARCHER_A),
        registering_component=ComponentType.NODE.name,
        upsert=True,
    )
    assert len(cert_db.cm.list()) == 1


def test_researcher_registering_multiple_node_certificates_accepted(cert_db):
    # The single-certificate constraint is the node's; a researcher registers
    # a certificate per node.
    for node in (_NODE_A, _NODE_C):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(cert_db.tmp, node),
            registering_component=ComponentType.RESEARCHER.name,
        )
    assert len(cert_db.cm.list()) == 2


def test_omitted_registering_component_skips_checks(cert_db):
    # Direct API use without a registering component keeps the permissive
    # behavior: a node certificate registers fine.
    cert_db.cm.register_certificate(certificate_path=_self_signed(cert_db.tmp, _NODE_A))
    assert cert_db.cm.get(_NODE_A) is not None


# -----------------------------------------------------------------------------
# Mutual-TLS trusted-certificate provider
# -----------------------------------------------------------------------------


@pytest.fixture
def bundle_env(tmp_path):
    """Certificate database for the trusted-certificate provider tests."""
    db_path = str(tmp_path / "certs.json")
    cm = CertificateManager(db_path=db_path)

    def register(party_id, pem, upsert=False):
        cm.insert(
            certificate=pem,
            party_id=party_id,
            component=ComponentType.NODE.name,
            upsert=upsert,
        )

    def real_certificate(party_id):
        """A real (~5 year) certificate, so expiry parsing has something to read."""
        pem_file = _self_signed(str(tmp_path), party_id)
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
    provider = TrustedCertificateBundle(bundle_env.db_path, ComponentType.NODE.name)

    bundle_env.register("node-1", "PEM-1")
    first = provider()
    assert b"PEM-1" in first
    assert first.count(b"PEM") == 1

    bundle_env.register("node-2", "PEM-2")
    second = provider()
    assert b"PEM-1" in second
    assert b"PEM-2" in second
    assert second.count(b"PEM") == 2


def test_bundle_does_not_reread_when_unchanged(bundle_env):
    bundle_env.register("node-1", "PEM-1")
    provider = TrustedCertificateBundle(bundle_env.db_path, ComponentType.NODE.name)
    provider()

    with patch("fedbiomed.common.certificate_manager.CertificateManager") as cm_cls:
        provider()
        cm_cls.assert_not_called()


def test_bundle_kept_while_database_is_partially_written(bundle_env):
    bundle_env.register("node-1", "PEM-1")
    provider = TrustedCertificateBundle(bundle_env.db_path, ComponentType.NODE.name)
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
    bundle_env.register("node-2", "PEM-2")
    assert b"PEM-2" in provider()


def test_bundle_kept_when_database_is_missing(bundle_env):
    bundle_env.register("node-1", "PEM-1")
    provider = TrustedCertificateBundle(bundle_env.db_path, ComponentType.NODE.name)
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
    env.register("node-1", env.real_certificate("node-1"))
    provider = TrustedCertificateBundle(env.db_path, ComponentType.NODE.name)
    provider()

    warned = _warned_parties(env.logger)
    assert len(warned) == 1
    assert "NODE certificate `node-1`" in warned[0]


def test_expiring_certificate_is_registered_as_event(bundle_expiry_env):
    env = bundle_expiry_env
    env.register("node-1", env.real_certificate("node-1"))
    provider = TrustedCertificateBundle(env.db_path, ComponentType.NODE.name)
    provider()

    events = _events(env.logger.security_event, "certificate_expiring")
    assert len(events) == 1
    assert events[0].kwargs["status"] == "warning"
    assert events[0].kwargs["party_id"] == "node-1"
    assert events[0].kwargs["component"] == ComponentType.NODE.name


def test_unreadable_certificate_store_is_registered_as_event(bundle_expiry_env):
    """A trust store that cannot be read leaves a stale bundle in use: audited."""
    env = bundle_expiry_env
    env.register("node-1", "PEM-1")
    provider = TrustedCertificateBundle(env.db_path, ComponentType.NODE.name)
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
    assert events[0].kwargs["component"] == ComponentType.NODE.name
    assert events[0].kwargs["db_path"] == env.db_path


def test_hot_added_certificate_is_reported_on_refresh(bundle_expiry_env):
    """The gap this closes: a certificate registered after startup."""
    env = bundle_expiry_env
    env.register("node-1", env.real_certificate("node-1"))
    provider = TrustedCertificateBundle(env.db_path, ComponentType.NODE.name)
    provider()

    env.register("node-2", env.real_certificate("node-2"))
    provider()

    assert len(_warned_parties(env.logger)) == 2


def test_certificate_is_not_reported_twice(bundle_expiry_env):
    env = bundle_expiry_env
    env.register("node-1", env.real_certificate("node-1"))
    provider = TrustedCertificateBundle(env.db_path, ComponentType.NODE.name)
    provider()

    # A refresh triggered by an unrelated registration must not re-report node-1
    env.register("node-2", "PEM-2")
    provider()
    provider()

    warned = _warned_parties(env.logger)
    assert len(warned) == 1
    assert "node-1" in warned[0]


def test_renewed_certificate_is_reported_again(bundle_expiry_env):
    """A renewal has a new expiry, so it is reported while still expiring."""
    env = bundle_expiry_env
    env.register("node-1", env.real_certificate("node-1"))
    provider = TrustedCertificateBundle(env.db_path, ComponentType.NODE.name)

    # Certificates are generated with a fixed ~5 year validity, so a renewal
    # cannot be given a distinct expiry date here; script the dates instead.
    renewed = datetime.now(timezone.utc) + timedelta(days=20)
    with patch.object(
        CertificateManager,
        "expiring_certificates",
        side_effect=[
            [("node-1", datetime.now(timezone.utc) + timedelta(days=10))],
            [("node-1", renewed)],
        ],
    ):
        provider()
        env.register("node-1", env.real_certificate("node-1"), upsert=True)
        provider()

    warned = _warned_parties(env.logger)
    assert len(warned) == 2
    assert f"{renewed:%Y-%m-%d}" in warned[1]

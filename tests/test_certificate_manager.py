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
    CERT_PURPOSE_CLIENT,
    CERT_PURPOSE_SERVER,
    CertificateManager,
    TrustedCertificateBundle,
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


def _self_signed(folder, org, cn="localhost", purpose=None, with_org_subject=True):
    """Generates a self-signed certificate, returns its PEM file path."""
    subject = {"CommonName": cn}
    if with_org_subject:
        subject["OrganizationName"] = org
    _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=folder,
        certificate_name=org.replace(" ", "_"),
        component_id=org,
        subject=subject,
        purpose=purpose,
    )
    return pem_file


def _load(pem_file):
    with open(pem_file, "rb") as f:
        return x509.load_pem_x509_certificate(f.read())


# -----------------------------------------------------------------------------
# CertificateManager over a mocked TinyDB
# -----------------------------------------------------------------------------


@pytest.fixture
def mocked_cm():
    """CertificateManager with the TinyDB layer fully mocked."""
    with (
        patch("os.path.isdir") as mock_isdir,
        patch("os.path.isfile") as mock_isfile,
        patch(
            "fedbiomed.common.certificate_manager.TinyDB.__init__",
            MagicMock(return_value=None),
        ) as tiny_db,
        patch("fedbiomed.common.certificate_manager.Query") as query,
        patch("fedbiomed.common.certificate_manager.TinyDB.table") as table,
        patch("fedbiomed.common.certificate_manager.TinyDB.close"),
    ):
        cm = CertificateManager(db_path="test-db")
        tiny_db.reset_mock()
        query.reset_mock()
        table.reset_mock()
        yield SimpleNamespace(
            cm=cm,
            tiny_db=tiny_db,
            query=query,
            table=table,
            isdir=mock_isdir,
            isfile=mock_isfile,
        )


def test_certificate_manager_initialization(mocked_cm):
    CertificateManager(db_path="test-db")

    # Check tiny db called
    mocked_cm.tiny_db.assert_called_once_with("test-db")
    mocked_cm.query.assert_called_once()
    mocked_cm.table.assert_called_once_with("Certificates")


def test_certificate_manager_set_db(mocked_cm):
    db_path = "test-set-db"
    mocked_cm.cm.set_db(db_path=db_path)
    mocked_cm.tiny_db.assert_called_once_with(db_path)
    mocked_cm.table.assert_called_once_with("Certificates")


def test_certificate_manager_get(mocked_cm):
    """Tests get method of certificate manager"""
    mocked_cm.cm.get("Test-ID")
    mocked_cm.table.return_value.get.assert_called_once()
    mocked_cm.query.return_value.party_id.__eq__.assert_called_once_with("Test-ID")


def test_certificate_manager_get_by_component(mocked_cm):
    """Tests retrieving all certificates of a given component type"""
    mocked_cm.table.return_value.search.return_value = [
        {"certificate": "cert-1"},
        {"certificate": "cert-2"},
    ]

    result = mocked_cm.cm.get_by_component("NODE")

    assert result == ["cert-1", "cert-2"]
    mocked_cm.table.return_value.search.assert_called_once()
    mocked_cm.query.return_value.component.__eq__.assert_called_once_with("NODE")


def test_certificate_manager_get_by_component_empty(mocked_cm):
    """Tests component lookup with no registered certificates"""
    mocked_cm.table.return_value.search.return_value = []
    assert mocked_cm.cm.get_by_component("NODE") == []


def test_certificate_manager_insert(mocked_cm):
    """Tests insert method of certificate manager"""

    certificate = "Dummy certificate"
    party_id = "test-id"
    component = "researcher"
    entry = dict(certificate=certificate, party_id=party_id, component=component)

    # Assume that get will return empty dict
    mocked_cm.table.return_value.get.return_value = {}
    mocked_cm.cm.insert(**entry)
    mocked_cm.table.return_value.insert.assert_called_once_with(entry)

    # A non-empty dict means that party is already registered
    mocked_cm.table.return_value.get.return_value = {"certificate": "xxx"}
    with pytest.raises(FedbiomedCertificateError):
        mocked_cm.cm.insert(**entry)

    # Already registered, and force to upsert/update with new data
    mocked_cm.table.reset_mock()
    mocked_cm.query.return_value.party_id.__eq__.reset_mock()
    mocked_cm.table.return_value.get.return_value = {"certificate": "xxx"}
    mocked_cm.cm.insert(**entry, upsert=True)
    assert mocked_cm.query.return_value.party_id.__eq__.call_count == 2
    mocked_cm.query.return_value.party_id.__eq__.assert_called_with(party_id)
    mocked_cm.table.return_value.upsert.assert_called_once_with(entry, False)


def test_certificate_manager_delete(mocked_cm):
    """Tests delete method of certificate manager"""

    mocked_cm.cm.delete("Test-ID")
    mocked_cm.table.return_value.remove.assert_called_once()
    mocked_cm.query.return_value.party_id.__eq__.assert_called_once_with("Test-ID")


def test_certificate_manager_list(mocked_cm):
    """Tests list method of certificate manager"""
    dummy_result = [{"certificate": "xxxx", "party_id": "xxxx"}]
    mocked_cm.table.return_value.all.return_value = dummy_result

    result = mocked_cm.cm.list()

    mocked_cm.table.return_value.all.assert_called_once()
    assert result == dummy_result

    with patch("builtins.print") as mock_print:
        result = mocked_cm.cm.list(verbose=True)
        mock_print.assert_called_once()
        assert result == dummy_result


@pytest.mark.parametrize(
    "party_id,component",
    [
        ("node_4f2c8a10-0e7d-4a11-9c33-8b7f0a1d2e44", "NODE"),
        ("researcher_9c2b1d70-1111-2222-3333-444455556666", "RESEARCHER"),
    ],
)
def test_certificate_manager_register_certificate(mocked_cm, party_id, component):
    """A registered certificate is inserted with its inferred component"""

    # Missing certificate file
    mocked_cm.isfile.return_value = False
    with pytest.raises(FedbiomedCertificateError):
        mocked_cm.cm.register_certificate(
            certificate_path="dummy/path", party_id=party_id
        )

    with (
        patch("builtins.open") as mock_open,
        patch(
            "fedbiomed.common.certificate_manager.CertificateManager.insert"
        ) as cm_insert,
    ):
        mocked_cm.isfile.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "Test certificate"
        )
        mocked_cm.table.return_value.get.return_value = {}

        mocked_cm.cm.register_certificate(
            certificate_path="dummy/path", party_id=party_id
        )
        cm_insert.assert_called_once_with(
            certificate="Test certificate",
            party_id=party_id,
            component=component,
            upsert=False,
        )


def test_certificate_manager_write_certificate_file(mocked_cm):
    with patch("builtins.open") as mock_open:
        mocked_cm.cm._write_certificate_file("dummy/path", "Certificate")
        mock_open.assert_called_once_with("dummy/path", "w", encoding="UTF-8")
        mock_open.return_value.__enter__.return_value.write.assert_called_once_with(
            "Certificate"
        )

        mock_open.side_effect = Exception
        with pytest.raises(FedbiomedCertificateError):
            mocked_cm.cm._write_certificate_file("dummy/path", "Certificate")


def test_operations_require_initialized_database():
    """Using the manager before `set_db` is a clear error, not an AttributeError."""
    with pytest.raises(FedbiomedCertificateError):
        CertificateManager().get("NODE_1")


def _generate_in(cm, certificate_folder):
    return cm.generate_self_signed_ssl_certificate(
        certificate_folder=certificate_folder,
        certificate_name="certificate",
        component_id="component-id",
    )


def test_generate_writes_key_and_certificate_files(mocked_cm):
    # Production always passes an absolute path (component roots are
    # absolutized before reaching certificate generation).
    folder = os.path.abspath("test-dir")
    mocked_cm.isdir.return_value = True
    with patch("fedbiomed.common.certificate_manager.open") as mock_open:
        _generate_in(mocked_cm.cm, folder)
        assert mock_open.call_args_list[0][0] == (
            os.path.join(folder, "certificate.key"),
            "wb",
        )
        assert mock_open.call_args_list[1][0] == (
            os.path.join(folder, "certificate.pem"),
            "wb",
        )
        assert mock_open.return_value.__enter__.return_value.write.call_count == 2


# Failing on the key file write, then on the certificate file write.
@pytest.mark.parametrize("side_effect", [Exception, [MagicMock(), Exception]])
def test_generate_raises_when_a_file_cannot_be_written(mocked_cm, side_effect):
    mocked_cm.isdir.return_value = True
    with patch("fedbiomed.common.certificate_manager.open", side_effect=side_effect):
        with pytest.raises(FedbiomedCertificateError):
            _generate_in(mocked_cm.cm, os.path.abspath("test-dir"))


def test_generate_raises_for_non_existing_folder(mocked_cm):
    mocked_cm.isdir.return_value = False
    with pytest.raises(FedbiomedCertificateError):
        _generate_in(mocked_cm.cm, os.path.abspath("test-dir"))


def test_generate_rejects_relative_path(mocked_cm):
    # Absoluteness is checked before the folder is used, so no mocking needed.
    with pytest.raises(FedbiomedCertificateError):
        _generate_in(mocked_cm.cm, "relative-dir")


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


def test_explicit_purpose_overrides_component_id(tmp_path):
    eku, _, _ = _extensions(
        _load(
            _self_signed(
                str(tmp_path),
                "NODE_1",
                purpose=CERT_PURPOSE_SERVER,
                with_org_subject=False,
            )
        )
    )
    assert ExtendedKeyUsageOID.SERVER_AUTH in eku
    assert ExtendedKeyUsageOID.CLIENT_AUTH not in eku


def test_unknown_purpose_raises(tmp_path):
    with pytest.raises(FedbiomedCertificateError):
        _self_signed(str(tmp_path), "NODE_1", purpose="bogus", with_org_subject=False)


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
            certificate_path=_self_signed(
                cert_db.tmp, "Hospital_x", purpose=CERT_PURPOSE_SERVER
            ),
            party_id=_NODE_A,
            registering_component=ComponentType.RESEARCHER.name,
        )


def test_node_registering_client_only_third_party_rejected(cert_db):
    with pytest.raises(FedbiomedCertificateError):
        cert_db.cm.register_certificate(
            certificate_path=_self_signed(
                cert_db.tmp, "Hospital_x", purpose=CERT_PURPOSE_CLIENT
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
    pkey = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Hospital_x")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(pkey.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
        .sign(private_key=pkey, algorithm=hashes.SHA256())
    )
    pem_file = os.path.join(cert_db.tmp, "no_eku.pem")
    with open(pem_file, "wb") as file:
        file.write(certificate.public_bytes(serialization.Encoding.PEM))
    cert_db.cm.register_certificate(
        certificate_path=pem_file,
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

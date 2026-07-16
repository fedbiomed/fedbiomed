import copy
import ipaddress
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch

from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID

from fedbiomed.common.certificate_manager import (
    CERT_PURPOSE_SERVER,
    CertificateManager,
    TrustedCertificateBundle,
    certificate_expiry,
    generate_certificate,
)
from fedbiomed.common.constants import CERTS_FOLDER_NAME, ComponentType
from fedbiomed.common.exceptions import FedbiomedCertificateError


class TestCertificateManager(unittest.TestCase):
    """
    Unit Tests for DatasetManager class.
    """

    def setUp(self) -> None:
        self.table_mock = Mock()
        self.table_mock.get.return_value = MagicMock()

        self.patch_isdir = patch("os.path.isdir")
        self.mock_isdir = self.patch_isdir.start()

        self.patch_isfile = patch("os.path.isfile")
        self.mock_isfile = self.patch_isfile.start()

        self.tiny_db_patch = patch(
            "fedbiomed.common.certificate_manager.TinyDB.__init__",
            MagicMock(return_value=None),
        )
        self.tiny_db_query_patch = patch("fedbiomed.common.certificate_manager.Query")
        self.tiny_db_table_patch = patch(
            "fedbiomed.common.certificate_manager.TinyDB.table"
        )

        self.tiny_db_mock = self.tiny_db_patch.start()
        self.tiny_db_table_mock = self.tiny_db_table_patch.start()
        self.tiny_db_query_mock = self.tiny_db_query_patch.start()

        self.tiny_db_table_patch.return_value = None
        self.cm = CertificateManager(db_path="test-db")

        self.tiny_db_mock.reset_mock()
        self.tiny_db_query_mock.reset_mock()
        self.tiny_db_table_mock.reset_mock()

    def tearDown(self) -> None:
        self.tiny_db_mock.reset_mock()
        self.tiny_db_query_mock.reset_mock()

        self.patch_isfile.stop()
        self.patch_isdir.stop()
        self.tiny_db_patch.stop()
        self.tiny_db_table_patch.stop()
        self.tiny_db_query_patch.stop()

        pass

    def test_01_certificate_manager_initialization(self):
        self.cm = CertificateManager(db_path="test-db")

        # Check tiny db called
        self.tiny_db_mock.assert_called_once_with("test-db")
        self.tiny_db_query_mock.assert_called_once()
        self.tiny_db_table_mock.assert_called_once_with("Certificates")

    def test_02_set_db(self):
        db_path = "test-set-db"
        self.cm.set_db(db_path=db_path)
        self.tiny_db_mock.assert_called_once_with(db_path)
        self.tiny_db_table_mock.assert_called_once_with("Certificates")

    def test_02_certificate_manager_get(self):
        """Tests get method of certificate manager"""
        self.cm.get("Test-ID")
        self.tiny_db_table_mock.return_value.get.assert_called_once()
        self.tiny_db_query_mock.return_value.party_id.__eq__.assert_called_once_with(
            "Test-ID"
        )

    def test_02_certificate_manager_get_by_component(self):
        """Tests retrieving all certificates of a given component type"""
        self.tiny_db_table_mock.return_value.search.return_value = [
            {"certificate": "cert-1"},
            {"certificate": "cert-2"},
        ]

        result = self.cm.get_by_component("NODE")

        self.assertEqual(result, ["cert-1", "cert-2"])
        self.tiny_db_table_mock.return_value.search.assert_called_once()
        self.tiny_db_query_mock.return_value.component.__eq__.assert_called_once_with(
            "NODE"
        )

    def test_02_certificate_manager_get_by_component_empty(self):
        """Tests component lookup with no registered certificates"""
        self.tiny_db_table_mock.return_value.search.return_value = []
        self.assertEqual(self.cm.get_by_component("NODE"), [])

    def test_03_certificate_manager_insert(self):
        """Tests get method of certificate manager"""

        certificate = "Dummy certificate"
        party_id = "test-id"
        component = "researcher"

        # Assume that get will return empty dict
        self.tiny_db_table_mock.return_value.get.return_value = {}
        self.cm.insert(
            certificate=certificate,
            party_id=party_id,
            component=component,
        )
        self.tiny_db_table_mock.return_value.insert.assert_called_once_with(
            dict(
                certificate=certificate,
                party_id=party_id,
                component=component,
            )
        )

        # Assume that get will return empty non-empty dict means that party is already registered
        self.tiny_db_table_mock.return_value.get.return_value = {"certificate": "xxx"}
        with self.assertRaises(FedbiomedCertificateError):
            self.cm.insert(
                certificate=certificate,
                party_id=party_id,
                component=component,
            )

        # Assume that get will return empty non-empty dict means that party is already registered
        # and force to upsert/update with new data
        self.tiny_db_table_mock.reset_mock()
        self.tiny_db_query_mock.return_value.party_id.__eq__.reset_mock()
        self.tiny_db_table_mock.return_value.get.return_value = {"certificate": "xxx"}
        self.cm.insert(
            certificate=certificate,
            party_id=party_id,
            component=component,
            upsert=True,
        )
        self.assertEqual(
            self.tiny_db_query_mock.return_value.party_id.__eq__.call_count, 2
        )
        self.tiny_db_query_mock.return_value.party_id.__eq__.assert_called_with(
            party_id
        )
        self.tiny_db_table_mock.return_value.upsert.assert_called_once_with(
            dict(
                certificate=certificate,
                party_id=party_id,
                component=component,
            ),
            False,
        )

    def test_04_certificate_manager_delete(self):
        """Tests delete method of certificate manager"""

        self.cm.delete("Test-ID")
        self.tiny_db_table_mock.return_value.remove.assert_called_once()
        self.tiny_db_query_mock.return_value.party_id.__eq__.assert_called_once_with(
            "Test-ID"
        )

    def test_05_certificate_manager_list(self):
        """Tests list method of certificate manager"""
        dummy_result = [{"certificate": "xxxx", "party_id": "xxxx"}]
        self.tiny_db_table_mock.return_value.all.return_value = dummy_result

        result = self.cm.list()

        self.tiny_db_table_mock.return_value.all.assert_called_once()
        self.assertListEqual(result, dummy_result)

        with patch("builtins.print") as mock_print:
            result = self.cm.list(verbose=True)
            mock_print.assert_called_once()
            self.assertListEqual(result, dummy_result)

    def test_06_certificate_manager_register_certificate(self):
        """Test certificate registration"""

        arguments = {
            "certificate_path": "dummy/path",
            "party_id": "node_4f2c8a10-0e7d-4a11-9c33-8b7f0a1d2e44",
        }

        with patch("os.path.isfile") as mock_isfile:
            mock_isfile.return_value = False

            with self.assertRaises(FedbiomedCertificateError):
                self.cm.register_certificate(**arguments)

            with (
                patch("builtins.open") as mock_open,
                patch(
                    "fedbiomed.common.certificate_manager.CertificateManager.insert"
                ) as cm_mock,
            ):
                mock_isfile.return_value = True
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    "Test certificate"
                )
                self.tiny_db_table_mock.return_value.get.return_value = {}

                args = copy.deepcopy(arguments)
                args.pop("certificate_path")
                self.cm.register_certificate(**arguments)
                cm_mock.assert_called_once_with(
                    **{
                        "certificate": "Test certificate",
                        "component": "NODE",
                        "upsert": False,
                        **args,
                    }
                )
                cm_mock.reset_mock()

                arguments_2 = {
                    **arguments,
                    "party_id": "researcher_9c2b1d70-1111-2222-3333-444455556666",
                }
                args = copy.deepcopy(arguments_2)
                args.pop("certificate_path")
                self.cm.register_certificate(**arguments_2)
                cm_mock.assert_called_once_with(
                    **{
                        "certificate": "Test certificate",
                        "component": "RESEARCHER",
                        "upsert": False,
                        **args,
                    }
                )

    def test_08_certificate_manager_write_certificate_file(self):
        with patch("builtins.open") as mock_open:
            self.cm._write_certificate_file("dummy/path", "Certificate")
            mock_open.assert_called_once_with("dummy/path", "w", encoding="UTF-8")
            mock_open.return_value.__enter__.return_value.write.assert_called_once_with(
                "Certificate"
            )

            mock_open.side_effect = Exception
            with self.assertRaises(FedbiomedCertificateError):
                self.cm._write_certificate_file("dummy/path", "Certificate")

    def test_09_operations_require_initialized_database(self):
        """Using the manager before `set_db` is a clear error, not an AttributeError."""
        with self.assertRaises(FedbiomedCertificateError):
            CertificateManager().get("NODE_1")

    def _generate(self, certificate_folder):
        return self.cm.generate_self_signed_ssl_certificate(
            certificate_folder=certificate_folder,
            certificate_name="certificate",
            component_id="component-id",
        )

    def test_10_generate_writes_key_and_certificate_files(self):
        # Production always passes an absolute path (component roots are
        # absolutized before reaching certificate generation).
        folder = os.path.abspath("test-dir")
        self.mock_isdir.return_value = True
        with patch("fedbiomed.common.certificate_manager.open") as mock_open:
            self._generate(folder)
            self.assertEqual(
                mock_open.call_args_list[0][0],
                (os.path.join(folder, "certificate.key"), "wb"),
            )
            self.assertEqual(
                mock_open.call_args_list[1][0],
                (os.path.join(folder, "certificate.pem"), "wb"),
            )
            self.assertEqual(
                mock_open.return_value.__enter__.return_value.write.call_count, 2
            )

    def test_11_generate_raises_when_a_file_cannot_be_written(self):
        self.mock_isdir.return_value = True
        # Failing on the key file write, then on the certificate file write.
        for side_effect in (Exception, [MagicMock(), Exception]):
            with self.subTest(side_effect=side_effect):
                with patch(
                    "fedbiomed.common.certificate_manager.open",
                    side_effect=side_effect,
                ):
                    with self.assertRaises(FedbiomedCertificateError):
                        self._generate(os.path.abspath("test-dir"))

    def test_12_generate_raises_for_non_existing_folder(self):
        self.mock_isdir.return_value = False
        with self.assertRaises(FedbiomedCertificateError):
            self._generate(os.path.abspath("test-dir"))

    def test_13_generate_rejects_relative_path(self):
        # Absoluteness is checked before the folder is used, so no mocking needed.
        with self.assertRaises(FedbiomedCertificateError):
            self._generate("relative-dir")


class TestCertificateExpiry(unittest.TestCase):
    """Tests certificate expiry helpers (`notAfter` parsing + reporting)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
            certificate_folder=self._tmp.name,
            certificate_name="cert",
            component_id="node_1",
            subject={"CommonName": "localhost", "OrganizationName": "node_1"},
        )
        with open(pem_file, "rb") as f:
            self.cert = f.read()

    def tearDown(self):
        self._tmp.cleanup()

    def test_certificate_expiry_returns_future_date(self):
        expiry = certificate_expiry(self.cert)
        self.assertIsInstance(expiry, datetime)
        self.assertGreater(expiry, datetime.now(timezone.utc))

    def test_certificate_expiry_none_for_unparsable(self):
        self.assertIsNone(certificate_expiry(b"not a certificate"))

    def test_expiring_certificates_filters_by_threshold_and_component(self):
        cm = CertificateManager()
        docs = [
            {
                "certificate": self.cert.decode(),
                "party_id": "node_1",
                "component": "NODE",
            },
            {
                "certificate": self.cert.decode(),
                "party_id": "res_1",
                "component": "RESEARCHER",
            },
        ]
        cm._db = MagicMock()
        cm._db.all.return_value = docs

        # Generated cert lasts ~5 years: a wide window catches it, a tight one doesn't
        wide = cm.expiring_certificates(within_days=10000, component="NODE")
        self.assertEqual([p for p, _ in wide], ["node_1"])
        self.assertEqual(cm.expiring_certificates(within_days=1, component="NODE"), [])
        # Component filter excludes the researcher entry
        self.assertEqual(
            [p for p, _ in cm.expiring_certificates(within_days=10000)],
            ["node_1", "res_1"],
        )

    def test_list_verbose_adds_expires_column(self):
        cm = CertificateManager()
        cm._db = MagicMock()
        cm._db.all.return_value = [
            {
                "certificate": self.cert.decode(),
                "party_id": "node_1",
                "component": "NODE",
            }
        ]
        with patch("fedbiomed.common.certificate_manager.tabulate") as tabulate:
            cm.list(verbose=True)
        rows = tabulate.call_args.args[0]
        self.assertIn("expires", rows[0])
        self.assertNotIn("certificate", rows[0])


class TestGenerateSelfSignedCertificate(unittest.TestCase):
    """Tests the `cryptography`-based self-signed certificate generation."""

    def _generate(self, cn, org="node_1"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
            certificate_folder=tmp.name,
            certificate_name="cert",
            component_id=org,
            subject={"CommonName": cn, "OrganizationName": org},
        )
        with open(pem_file, "rb") as f:
            return x509.load_pem_x509_certificate(f.read())

    def _san(self, cert):
        return cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value

    def test_subject_carries_common_and_organization_name(self):
        cert = self._generate("localhost", org="node_1")
        self.assertEqual(
            cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value,
            "localhost",
        )
        self.assertEqual(
            cert.subject.get_attributes_for_oid(x509.oid.NameOID.ORGANIZATION_NAME)[
                0
            ].value,
            "node_1",
        )

    def test_hostname_common_name_produces_dns_san(self):
        san = self._san(self._generate("localhost"))
        self.assertEqual(san.get_values_for_type(x509.DNSName), ["localhost"])

    def test_ip_common_name_produces_ip_san(self):
        san = self._san(self._generate("10.0.0.5"))
        self.assertEqual(
            san.get_values_for_type(x509.IPAddress),
            [ipaddress.ip_address("10.0.0.5")],
        )

    def test_wildcard_common_name_has_no_san(self):
        # `*` is neither a resolvable host nor an IP -> no SubjectAlternativeName
        with self.assertRaises(x509.ExtensionNotFound):
            self._san(self._generate("*"))

    def _extensions(self, cert):
        eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        key_usage = cert.extensions.get_extension_for_class(x509.KeyUsage).value
        basic = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
        return eku, key_usage, basic

    def _generate_with_purpose(self, component_id, purpose):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
            certificate_folder=tmp.name,
            certificate_name="cert",
            component_id=component_id,
            subject={"CommonName": "localhost"},
            purpose=purpose,
        )
        with open(pem_file, "rb") as f:
            return x509.load_pem_x509_certificate(f.read())

    def test_certificates_are_end_entity_not_ca(self):
        _, _, basic = self._extensions(self._generate("localhost", org="node_1"))
        self.assertFalse(basic.ca)

    def test_researcher_id_gets_server_auth_only(self):
        eku, key_usage, _ = self._extensions(
            self._generate("localhost", org="RESEARCHER_1")
        )
        self.assertIn(ExtendedKeyUsageOID.SERVER_AUTH, eku)
        self.assertNotIn(ExtendedKeyUsageOID.CLIENT_AUTH, eku)
        self.assertTrue(key_usage.digital_signature)
        self.assertTrue(key_usage.key_encipherment)

    def test_node_id_gets_client_auth_only(self):
        eku, key_usage, _ = self._extensions(self._generate("localhost", org="NODE_1"))
        self.assertIn(ExtendedKeyUsageOID.CLIENT_AUTH, eku)
        self.assertNotIn(ExtendedKeyUsageOID.SERVER_AUTH, eku)
        self.assertTrue(key_usage.digital_signature)
        self.assertFalse(key_usage.key_encipherment)

    def test_unrecognized_id_gets_both_roles(self):
        eku, _, _ = self._extensions(
            self._generate("localhost", org="some-other-party")
        )
        self.assertIn(ExtendedKeyUsageOID.SERVER_AUTH, eku)
        self.assertIn(ExtendedKeyUsageOID.CLIENT_AUTH, eku)

    def test_explicit_purpose_overrides_component_id(self):
        eku, _, _ = self._extensions(
            self._generate_with_purpose("NODE_1", CERT_PURPOSE_SERVER)
        )
        self.assertIn(ExtendedKeyUsageOID.SERVER_AUTH, eku)
        self.assertNotIn(ExtendedKeyUsageOID.CLIENT_AUTH, eku)

    def test_unknown_purpose_raises(self):
        with self.assertRaises(FedbiomedCertificateError):
            self._generate_with_purpose("NODE_1", "bogus")


class TestGenerateCertificate(unittest.TestCase):
    """The module-level `generate_certificate` wrapper."""

    _NODE_ID = "NODE_4f2c8a10-0e7d-4a11-9c33-8b7f0a1d2e44"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_writes_certificate_files_under_root(self):
        key_file, pem_file = generate_certificate(
            root=self._tmp.name, component_id=self._NODE_ID
        )
        certs_dir = os.path.join(self._tmp.name, CERTS_FOLDER_NAME)
        self.assertTrue(os.path.isfile(key_file))
        self.assertTrue(os.path.isfile(pem_file))
        self.assertEqual(os.path.dirname(pem_file), certs_dir)

    def test_aborts_when_certificates_already_exist(self):
        certs_dir = os.path.join(self._tmp.name, CERTS_FOLDER_NAME)
        os.makedirs(certs_dir)
        with open(os.path.join(certs_dir, "certificate.pem"), "w"):
            pass
        with self.assertRaises(FedbiomedCertificateError):
            generate_certificate(root=self._tmp.name, component_id=self._NODE_ID)


class TestRegisterCertificateComponent(unittest.TestCase):
    """Component classification of registered certificates.

    Component ids are `<COMPONENT_TYPE>_<uuid>` (see `Config.generate`), so these
    use that real shape: classifying a node certificate as `RESEARCHER` leaves the
    researcher's mutual-TLS trust bundle empty.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = CertificateManager(db_path=os.path.join(self._tmp.name, "certs.json"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _register(self, party_id):
        _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
            certificate_folder=self._tmp.name,
            certificate_name=party_id,
            component_id=party_id,
            subject={"CommonName": "localhost", "OrganizationName": party_id},
        )
        self.cm.register_certificate(certificate_path=pem_file, party_id=party_id)

    def test_node_id_registers_as_node_component(self):
        self._register(
            f"{ComponentType.NODE.name}_4f2c8a10-0e7d-4a11-9c33-8b7f0a1d2e44"
        )
        self.assertEqual(len(self.cm.get_by_component(ComponentType.NODE.name)), 1)
        self.assertEqual(
            len(self.cm.get_by_component(ComponentType.RESEARCHER.name)), 0
        )

    def test_researcher_id_registers_as_researcher_component(self):
        self._register(
            f"{ComponentType.RESEARCHER.name}_9c2b1d70-1111-2222-3333-444455556666"
        )
        self.assertEqual(
            len(self.cm.get_by_component(ComponentType.RESEARCHER.name)), 1
        )
        self.assertEqual(len(self.cm.get_by_component(ComponentType.NODE.name)), 0)

    def test_lowercase_node_id_registers_as_node_component(self):
        """Ids from older lowercase-prefixed deployments keep classifying as NODE."""
        self._register("node_4f2c8a10-0e7d-4a11-9c33-8b7f0a1d2e44")
        self.assertEqual(len(self.cm.get_by_component(ComponentType.NODE.name)), 1)

    def test_unprefixed_id_is_rejected(self):
        with self.assertRaises(FedbiomedCertificateError):
            self._register("some-other-party")

    def test_non_uuid_id_is_rejected(self):
        with self.assertRaises(FedbiomedCertificateError):
            self._register("NODE_not-a-uuid")


class TestRegisterCertificatePartyId(unittest.TestCase):
    """`party_id` reconciliation against the certificate identity (`O=`).

    A `O=` that is not a valid party id is treated as no usable identity, like an
    absent one (the absent case is covered by `test_06`).
    """

    _NODE_A = "NODE_4f2c8a10-0e7d-4a11-9c33-8b7f0a1d2e44"
    _NODE_B = "NODE_9c2b1d70-1111-2222-3333-444455556666"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cm = CertificateManager(db_path=os.path.join(self._tmp.name, "certs.json"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _cert(self, org):
        _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
            certificate_folder=self._tmp.name,
            certificate_name=org.replace(" ", "_"),
            component_id=org,
            subject={"CommonName": "localhost", "OrganizationName": org},
        )
        return pem_file

    def test_recovers_party_id_from_certificate(self):
        self.cm.register_certificate(certificate_path=self._cert(self._NODE_A))
        self.assertIsNotNone(self.cm.get(self._NODE_A))
        self.assertEqual(len(self.cm.get_by_component(ComponentType.NODE.name)), 1)

    def test_matching_party_id_is_accepted(self):
        self.cm.register_certificate(
            certificate_path=self._cert(self._NODE_A), party_id=self._NODE_A
        )
        self.assertIsNotNone(self.cm.get(self._NODE_A))

    def test_conflicting_party_id_raises(self):
        with self.assertRaises(FedbiomedCertificateError):
            self.cm.register_certificate(
                certificate_path=self._cert(self._NODE_A), party_id=self._NODE_B
            )

    def test_party_id_required_without_usable_identity(self):
        with self.assertRaises(FedbiomedCertificateError):
            self.cm.register_certificate(certificate_path=self._cert("Hospital A"))

    def test_given_party_id_used_without_usable_identity(self):
        self.cm.register_certificate(
            certificate_path=self._cert("Hospital A"), party_id=self._NODE_A
        )
        self.assertIsNotNone(self.cm.get(self._NODE_A))

    def test_given_party_id_must_follow_pattern(self):
        with self.assertRaises(FedbiomedCertificateError):
            self.cm.register_certificate(
                certificate_path=self._cert("Hospital A"), party_id="not-a-valid-id"
            )


class _TrustedCertificateBundleFixture:
    """Certificate database fixture for the trusted-certificate provider tests."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "certs.json")
        self.cm = CertificateManager(db_path=self.db_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _register(self, party_id, pem, upsert=False):
        self.cm.insert(
            certificate=pem,
            party_id=party_id,
            component=ComponentType.NODE.name,
            upsert=upsert,
        )

    def _real_certificate(self, party_id):
        """A real (~5 year) certificate, so expiry parsing has something to read."""
        _, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
            certificate_folder=self._tmp.name,
            certificate_name=party_id,
            component_id=party_id,
            subject={"CommonName": "localhost", "OrganizationName": party_id},
        )
        with open(pem_file) as file:
            return file.read()


class TestTrustedCertificateBundle(_TrustedCertificateBundleFixture, unittest.TestCase):
    """Hot-add behaviour of the mutual-TLS trusted-certificate provider."""

    def test_bundle_picks_up_hot_added_certificate(self):
        provider = TrustedCertificateBundle(self.db_path, ComponentType.NODE.name)

        self._register("node-1", "PEM-1")
        first = provider()
        self.assertIn(b"PEM-1", first)
        self.assertEqual(first.count(b"PEM"), 1)

        self._register("node-2", "PEM-2")
        second = provider()
        self.assertIn(b"PEM-1", second)
        self.assertIn(b"PEM-2", second)
        self.assertEqual(second.count(b"PEM"), 2)

    def test_bundle_does_not_reread_when_unchanged(self):
        self._register("node-1", "PEM-1")
        provider = TrustedCertificateBundle(self.db_path, ComponentType.NODE.name)
        provider()

        with patch("fedbiomed.common.certificate_manager.CertificateManager") as cm_cls:
            provider()
            cm_cls.assert_not_called()

    def test_bundle_kept_while_database_is_partially_written(self):
        self._register("node-1", "PEM-1")
        provider = TrustedCertificateBundle(self.db_path, ComponentType.NODE.name)
        self.assertIn(b"PEM-1", provider())

        # TinyDB writes in place, so a read concurrent with another process
        # registering a certificate can observe a truncated file.
        with open(self.db_path) as file:
            content = file.read()
        with open(self.db_path, "w") as file:
            file.write(content[: len(content) // 2])

        self.assertIn(b"PEM-1", provider())

        with open(self.db_path, "w") as file:
            file.write(content)
        self._register("node-2", "PEM-2")
        self.assertIn(b"PEM-2", provider())

    def test_bundle_kept_when_database_is_missing(self):
        self._register("node-1", "PEM-1")
        provider = TrustedCertificateBundle(self.db_path, ComponentType.NODE.name)
        self.assertIn(b"PEM-1", provider())

        os.remove(self.db_path)
        self.assertIn(b"PEM-1", provider())


class TestTrustedCertificateBundleExpiry(
    _TrustedCertificateBundleFixture, unittest.TestCase
):
    """Expiry reporting of the mutual-TLS trusted-certificate provider."""

    def setUp(self) -> None:
        super().setUp()
        # Generated certificates last ~5 years; widen the window so they register
        # as expiring without having to forge an expiry date.
        self.days_patch = patch(
            "fedbiomed.common.certificate_manager.CERTIFICATE_EXPIRY_WARNING_DAYS",
            10000,
        )
        self.days_patch.start()
        self.addCleanup(self.days_patch.stop)

        self.logger_patch = patch("fedbiomed.common.certificate_manager.logger")
        self.logger = self.logger_patch.start()
        self.addCleanup(self.logger_patch.stop)

    def _warned_parties(self):
        return [
            call.args[0]
            for call in self.logger.warning.call_args_list
            if "expires on" in call.args[0]
        ]

    def test_expiring_certificate_is_reported_on_first_read(self):
        self._register("node-1", self._real_certificate("node-1"))
        provider = TrustedCertificateBundle(self.db_path, ComponentType.NODE.name)
        provider()

        warned = self._warned_parties()
        self.assertEqual(len(warned), 1)
        self.assertIn("NODE certificate `node-1`", warned[0])

    def test_hot_added_certificate_is_reported_on_refresh(self):
        """The gap this closes: a certificate registered after startup."""
        self._register("node-1", self._real_certificate("node-1"))
        provider = TrustedCertificateBundle(self.db_path, ComponentType.NODE.name)
        provider()

        self._register("node-2", self._real_certificate("node-2"))
        provider()

        self.assertEqual(len(self._warned_parties()), 2)

    def test_certificate_is_not_reported_twice(self):
        self._register("node-1", self._real_certificate("node-1"))
        provider = TrustedCertificateBundle(self.db_path, ComponentType.NODE.name)
        provider()

        # A refresh triggered by an unrelated registration must not re-report node-1
        self._register("node-2", "PEM-2")
        provider()
        provider()

        warned = self._warned_parties()
        self.assertEqual(len(warned), 1)
        self.assertIn("node-1", warned[0])

    def test_renewed_certificate_is_reported_again(self):
        """A renewal has a new expiry, so it is reported while still expiring."""
        self._register("node-1", self._real_certificate("node-1"))
        provider = TrustedCertificateBundle(self.db_path, ComponentType.NODE.name)

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
            self._register("node-1", self._real_certificate("node-1"), upsert=True)
            provider()

        warned = self._warned_parties()
        self.assertEqual(len(warned), 2)
        self.assertIn(f"{renewed:%Y-%m-%d}", warned[1])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

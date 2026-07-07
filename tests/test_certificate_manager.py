import copy
import ipaddress
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

from cryptography import x509

from fedbiomed.common.certificate_manager import (
    CertificateManager,
    certificate_expiry,
)
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

    def test_05_certificate_manager_delete(self):
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
            "party_id": "node_party-id",
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

                arguments_2 = {**arguments, "party_id": "xxx"}
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

    @patch("os.path.abspath")
    def test_10_certificate_generate_self_signed_ssl_certificate(self, mock_abspath):
        """Tests method to generate self-signed ssl certificate"""

        certificate_folder = "test-dir"
        with patch("fedbiomed.common.certificate_manager.open") as mock_open:
            self.mock_isdir.return_value = True
            self.cm.generate_self_signed_ssl_certificate(
                certificate_folder=certificate_folder,
                certificate_name="certificate",
                component_id="component-id",
            )
            self.assertEqual(
                mock_open.call_args_list[0][0],
                (os.path.join(certificate_folder, "certificate.key"), "wb"),
            )
            self.assertEqual(
                mock_open.call_args_list[1][0],
                (os.path.join(certificate_folder, "certificate.pem"), "wb"),
            )

            self.assertEqual(
                mock_open.return_value.__enter__.return_value.write.call_count, 2
            )

            mock_open.side_effect = Exception
            with self.assertRaises(FedbiomedCertificateError):
                self.cm.generate_self_signed_ssl_certificate(
                    certificate_folder=certificate_folder,
                    certificate_name="certificate",
                    component_id="component-id",
                )

            # It didn't work: raising exception for the second call of `open`
            mock_open.side_effect = [MagicMock(), Exception]
            mock_abspath.return_value = True
            self.mock_isdir.return_value = True
            with self.assertRaises(FedbiomedCertificateError):
                self.cm.generate_self_signed_ssl_certificate(
                    certificate_folder=certificate_folder,
                    certificate_name="certificate",
                    component_id="component-id",
                )

            mock_abspath.return_value = False
            with self.assertRaises(FedbiomedCertificateError):
                self.cm.generate_self_signed_ssl_certificate(
                    certificate_folder=certificate_folder,
                    certificate_name="certificate",
                    component_id="component-id",
                )

            mock_abspath.return_value = True
            self.mock_isdir.return_value = False
            with self.assertRaises(FedbiomedCertificateError):
                self.cm.generate_self_signed_ssl_certificate(
                    certificate_folder=os.path.join(certificate_folder, "no-exisintg"),
                    certificate_name="certificate",
                    component_id="component-id",
                )


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

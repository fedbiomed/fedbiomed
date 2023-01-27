import copy
import unittest
from unittest.mock import patch, MagicMock, Mock

from fedbiomed.common.certificate_manager import CertificateManager
from fedbiomed.common.exceptions import FedbiomedCertificateError, FedbiomedError


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

        self.tiny_db_patch = patch("fedbiomed.common.certificate_manager.TinyDB.__init__",
                                   MagicMock(return_value=None))
        self.tiny_db_query_patch = patch("fedbiomed.common.certificate_manager.Query")
        self.tiny_db_table_patch = patch("fedbiomed.common.certificate_manager.TinyDB.table")

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
        """Tests get method of certificate manager """
        self.cm.get("Test-ID")
        self.tiny_db_table_mock.return_value.get.assert_called_once()
        self.tiny_db_query_mock.return_value.party_id.__eq__.assert_called_once_with("Test-ID")

    def test_03_certificate_manager_insert(self):
        """Tests get method of certificate manager """

        certificate = "Dummy certificate"
        party_id = "test-id"
        component = "researcher"
        ip = "1.1.1.1"
        port = 1234

        # Assume that get will return empty dict
        self.tiny_db_table_mock.return_value.get.return_value = {}
        self.cm.insert(certificate=certificate,
                       party_id=party_id,
                       component=component,
                       ip=ip,
                       port=port)

        self.tiny_db_table_mock.return_value.insert.assert_called_once_with(dict(certificate=certificate,
                                                                                 party_id=party_id,
                                                                                 component=component,
                                                                                 ip=ip,
                                                                                 port=port))

        # Assume that get will return empty non-empty dict means that party is already registered
        self.tiny_db_table_mock.return_value.get.return_value = {"certificate": "xxx"}
        with self.assertRaises(FedbiomedCertificateError):
            self.cm.insert(
                certificate=certificate,
                party_id=party_id,
                component=component,
                ip=ip,
                port=port
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
            ip=ip,
            port=port,
            upsert=True
        )
        self.assertEqual(self.tiny_db_query_mock.return_value.party_id.__eq__.call_count, 2)
        self.tiny_db_query_mock.return_value.party_id.__eq__.assert_called_with(party_id)
        self.tiny_db_table_mock.return_value.upsert.assert_called_once_with(dict(certificate=certificate,
                                                                                 party_id=party_id,
                                                                                 component=component,
                                                                                 ip=ip,
                                                                                 port=port), False)

    def test_04_certificate_manager_delete(self):
        """Tests delete method of certificate manager """

        self.cm.delete("Test-ID")
        self.tiny_db_table_mock.return_value.remove.assert_called_once()
        self.tiny_db_query_mock.return_value.party_id.__eq__.assert_called_once_with("Test-ID")

    def test_05_certificate_manager_delete(self):
        dummy_result = [
            {
                "certificate": "xxxx",
                "party_id": "xxxx",
                "ip": "xxxx",
                "port": "yyyy"
            }
        ]
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
            "ip": "1.1.1.1",
            "port": 1000
        }

        with patch("os.path.isfile") as mock_isfile:
            mock_isfile.return_value = False

            with self.assertRaises(FedbiomedCertificateError):
                self.cm.register_certificate(**arguments)

            with patch("builtins.open") as mock_open, \
                    patch("fedbiomed.common.certificate_manager.CertificateManager.insert") as cm_mock:
                mock_isfile.return_value = True
                mock_open.return_value.__enter__.return_value.read.return_value = "Test certificate"
                self.tiny_db_table_mock.return_value.get.return_value = {}

                args = copy.deepcopy(arguments)
                args.pop("certificate_path")
                self.cm.register_certificate(**arguments)
                cm_mock.assert_called_once_with(
                    **{'certificate': 'Test certificate',
                       'component': "NODE",
                       'upsert': False,
                       **args}
                )
                cm_mock.reset_mock()

                arguments_2 = {**arguments, "party_id": "xxx"}
                args = copy.deepcopy(arguments_2)
                args.pop("certificate_path")
                self.cm.register_certificate(**arguments_2)
                cm_mock.assert_called_once_with(
                    **{'certificate': 'Test certificate',
                       'component': "RESEARCHER",
                       'upsert': False,
                       **args}
                )

    @patch("fedbiomed.common.certificate_manager.CertificateManager.get")
    @patch("os.path.abspath")
    @patch("os.remove")
    @patch("fedbiomed.common.certificate_manager.read_file")
    @patch("fedbiomed.common.certificate_manager.CertificateManager._write_certificate_file")
    @patch("fedbiomed.common.certificate_manager.CertificateManager._append_new_ip_address")
    def test_07_certificate_manager_write_mpc_certificates_for_experiment(
            self,
            mock_append_new_ip_address,
            mock_write_certificate_file,
            mock_read_file,
            mock_remove,
            mock_abspath,
            mock_cm_get
    ):
        """Tests writing MPC certificate files into given directory"""

        arguments = {
            "parties": ["id-1", "id-2", "id-3"],
            "path": "/dummy/path",
            "self_id": "id-2",
            "self_ip": "1.1.1.1",
            "self_port": 12345,
            "self_private_key": "MY-PRIVATE-KEY",
            "self_public_key": "MY-PUBLIC-KEY"
        }

        self.mock_isdir.return_value = False
        with self.assertRaises(FedbiomedCertificateError):
            self.cm.write_mpc_certificates_for_experiment(**arguments)

        self.mock_isdir.return_value = True
        self.mock_isfile.return_value = True

        def get_side_effect(party):

            return {
                "certificate": f"cert-{party}",
                "ip": "1.1.1.1",
                "port": 12,
                "party_id": party
            }

        mock_abspath.side_effect = lambda x: x
        mock_remove.return_value = True
        mock_cm_get.side_effect = get_side_effect
        mock_read_file.return_value = "Hello I am self certificate"

        result = self.cm.write_mpc_certificates_for_experiment(**arguments)

        # Test party query call
        self.assertEqual(mock_cm_get.call_count, 2)

        # Test append ip address call count and call arguments
        self.assertEqual(mock_append_new_ip_address.call_count, 3)
        self.assertEqual(mock_append_new_ip_address.call_args_list[0][0],
                         ('/dummy/path/ip_addresses', '1.1.1.1', 12, 'id-1'))

        self.assertEqual(mock_append_new_ip_address.call_args_list[1][0],
                         ('/dummy/path/ip_addresses', '1.1.1.1', 12345, 'id-2')) # Self as port 12345
        self.assertEqual(mock_append_new_ip_address.call_args_list[2][0],
                         ('/dummy/path/ip_addresses', '1.1.1.1', 12, 'id-3'))

        # Test write certificate calls
        self.assertEqual(mock_write_certificate_file.call_count, 4)
        # 4 because for Self component private key will be written too

        self.assertEqual(mock_write_certificate_file.call_args_list[0][0], ('/dummy/path/P0.pem', 'cert-id-1'))
        self.assertEqual(mock_write_certificate_file.call_args_list[1][0],
                         ('/dummy/path/P1.key', "Hello I am self certificate"))
        self.assertEqual(mock_write_certificate_file.call_args_list[2][0],
                         ('/dummy/path/P1.pem', "Hello I am self certificate"))
        self.assertEqual(mock_write_certificate_file.call_args_list[3][0], ('/dummy/path/P2.pem', 'cert-id-3'))

        # Check if the result is as expected
        self.assertListEqual(
            result,
            ['/dummy/path/P0.pem', '/dummy/path/P1.key', '/dummy/path/P1.pem', '/dummy/path/P2.pem'])

        mock_cm_get.side_effect = None
        mock_cm_get.return_value = {}
        with self.assertRaises(FedbiomedCertificateError):
            result = self.cm.write_mpc_certificates_for_experiment(**arguments)

        mock_cm_get.side_effect = get_side_effect
        mock_append_new_ip_address.side_effect = FedbiomedCertificateError
        with self.assertRaises(FedbiomedCertificateError):
            result = self.cm.write_mpc_certificates_for_experiment(**arguments)

        mock_append_new_ip_address.side_effect = FedbiomedError
        with self.assertRaises(FedbiomedCertificateError):
            result = self.cm.write_mpc_certificates_for_experiment(**arguments)

        mock_append_new_ip_address.side_effect = Exception
        with self.assertRaises(FedbiomedCertificateError):
            result = self.cm.write_mpc_certificates_for_experiment(**arguments)

    def test_08_certificate_manager_write_certificate_file(self):
        """ """

        with patch("builtins.open") as mock_open:
            self.cm._write_certificate_file("dummy/path", "Certificate")
            mock_open.assert_called_once_with("dummy/path", "w")
            mock_open.return_value.__enter__.return_value.write.assert_called_once_with("Certificate")
            mock_open.return_value.__enter__.return_value.close.assert_called_once()

            mock_open.side_effect = Exception
            with self.assertRaises(FedbiomedCertificateError):
                self.cm._write_certificate_file("dummy/path", "Certificate")

    def test_09_certificate_manager_append_new_ip_address(self):
        """Test method for adding new ip address into given path"""

        with patch("builtins.open") as mock_open:
            self.cm._append_new_ip_address("dummy/path", "localhost", 1234, "party-id")
            mock_open.assert_called_once_with("dummy/path", "a")
            mock_open.return_value.__enter__.return_value.write.assert_called_once_with("localhost:1234\n")
            mock_open.return_value.__enter__.return_value.close.assert_called_once()

            mock_open.side_effect = Exception
            with self.assertRaises(FedbiomedCertificateError):
                self.cm._append_new_ip_address("dummy/path", "localhost", 1234, "party-id")

    @patch("os.path.abspath")
    def test_10_certificate_generate_self_signed_ssl_certificate(self, mock_abspath):
        """Tests method to generate self-signed ssl certificate"""

        certificate_folder = "/dummy/folder"

        with patch("builtins.open") as mock_open:

            self.mock_isdir.return_value = True
            self.cm.generate_self_signed_ssl_certificate(
                certificate_folder=certificate_folder,
                certificate_name="certificate",
                component_id="component-id"
            )
            self.assertEqual(mock_open.call_args_list[0][0], ('/dummy/folder/certificate.key', 'wb'))
            self.assertEqual(mock_open.call_args_list[1][0], ('/dummy/folder/certificate.pem', 'wb'))

            self.assertEqual(mock_open.return_value.__enter__.return_value.write.call_count, 2)
            self.assertEqual(mock_open.return_value.__enter__.return_value.close.call_count, 2)

            mock_open.side_effect = Exception
            with self.assertRaises(FedbiomedCertificateError):
                self.cm.generate_self_signed_ssl_certificate(
                    certificate_folder=certificate_folder,
                    certificate_name="certificate",
                    component_id="component-id"
                )

            # It didn't work: raising exception for the second call of `open`
            mock_open.side_effect = [MagicMock(), Exception]
            mock_abspath.return_value = True
            self.mock_isdir.return_value = True
            with self.assertRaises(FedbiomedCertificateError):
                self.cm.generate_self_signed_ssl_certificate(
                    certificate_folder=certificate_folder,
                    certificate_name="certificate",
                    component_id="component-id"
                )

            mock_abspath.return_value = False
            with self.assertRaises(FedbiomedCertificateError):
                self.cm.generate_self_signed_ssl_certificate(
                    certificate_folder=certificate_folder,
                    certificate_name="certificate",
                    component_id="component-id"
                )

            mock_abspath.return_value = True
            self.mock_isdir.return_value = False
            with self.assertRaises(FedbiomedCertificateError):
                self.cm.generate_self_signed_ssl_certificate(
                    certificate_folder=certificate_folder,
                    certificate_name="certificate",
                    component_id="component-id"
                )


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock, Mock

from fedbiomed.common.certificate_manager import CertificateManager
from fedbiomed.common.exceptions import FedbiomedCertificateError


class TestDatasetManager(unittest.TestCase):
    """
    Unit Tests for DatasetManager class.
    """

    def setUp(self) -> None:
        self.table_mock = Mock()
        self.table_mock.get.return_value = MagicMock()

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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

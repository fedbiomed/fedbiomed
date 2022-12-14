
import unittest
from unittest.mock import patch, MagicMock

from fedbiomed.common.certificate_manager import CertificateManager


class TestDatasetManager(unittest.TestCase):
    """
    Unit Tests for DatasetManager class.
    """


    def setUp(self) -> None:



        self.tiny_db_patch = patch("fedbiomed.common.certificate_manager.TinyDB.__init__",
                                   MagicMock(return_value=None))
        self.tiny_db_query_patch = patch("fedbiomed.common.certificate_manager.Query")
        self.tiny_db_table_patch = patch("fedbiomed.common.certificate_manager.TinyDB.table")
        self.tiny_db_insert_patch = patch("tinydb.table.Table.insert")

        self.tiny_db_mock = self.tiny_db_patch.start()
        self.tiny_db_table_mock = self.tiny_db_table_patch.start()
        self.tiny_db_query_mock = self.tiny_db_query_patch.start()
        self.tiny_db_insert_mock = self.tiny_db_insert_patch.start()

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
        self.tiny_db_insert_patch.stop()
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

    def test_02_insert(self):
        """"""

        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
import unittest
from unittest.mock import patch, MagicMock
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.researcher.secagg import SecaggServkeyContext

import testsupport.mock_node_environ  ## noqa (remove flake8 false warning)

from fedbiomed.node.secagg_manager import SecaggServkeyManager, SecaggBiprimeManager


class FakeTinyDB:
    def __init__(self, path):
        pass

    def table(self, *args, **kwargs):
        return FakeTable()

class FakeQuery:
    def __init__(self):
        pass

class FakeTable:
    def __init__(self):
        pass

class TestSecaggManager(unittest.TestCase):
    """Test for SecaggManager node side module"""

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def setUp(self):
        self.patcher_db = patch('fedbiomed.node.secagg_manager.TinyDB', FakeTinyDB)
        self.patcher_query = patch('fedbiomed.node.secagg_manager.Query', FakeQuery)

        self.patcher_db.start()
        self.patcher_query.start()

    def tearDown(self) -> None:
        self.patcher_query.stop()
        self.patcher_db.stop()

    def test_secagg_manager_01_init_ok(self):
        """Instantiate SecaggManager normal successful case"""
        # prepare
        managers = [SecaggServkeyManager, SecaggBiprimeManager]

        # action
        for manager in managers:
            manager()

        # test
        # nothing to test at this point ...

    @patch('fedbiomed.node.secagg_manager.TinyDB.__init__')
    def test_secagg_manager_02_init_error(
            self,
            patch_tinydb_init):
        """Instantiate SecaggManager fails with exception"""
        # prepare
        managers = [SecaggServkeyManager, SecaggBiprimeManager]

        patch_tinydb_init.side_effect = FedbiomedSecaggError

        # action + check
        for manager in managers:
            with self.assertRaises(FedbiomedSecaggError):
                manager()

if __name__ == '__main__':  # pragma: no cover
    unittest.main()

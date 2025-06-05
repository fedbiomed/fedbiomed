from unittest.mock import patch, MagicMock
import unittest

from testsupport.fake_tiny_db import FakeTinyDB, FakeQuery

from fedbiomed.common.channel_manager import ChannelManager
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError


class TestCommonChannelManager(unittest.TestCase):
    """Test for common channel manager module"""

    def setUp(self):
        self.patcher_db = patch('fedbiomed.common.channel_manager.TinyDB', FakeTinyDB)
        self.patcher_query = patch('fedbiomed.common.channel_manager.Query', FakeQuery)

        self.mock_db = self.patcher_db.start()
        self.mock_query = self.patcher_query.start()

        #self.dh_key_1_in_bytes = b'DH_KEY_1'
        #self.dh_key_1_in_str = str(base64.b64encode(self.dh_key_1_in_bytes), 'utf-8')  # value = 'REhfS0VZXzE='
        #self.dh_key_2_in_bytes = b'DH_KEY_2'
        #self.dh_key_2_in_str = str(base64.b64encode(self.dh_key_2_in_bytes), 'utf-8')  # value = 'REhfS0VZXzI='
        #self.equivalences = {SecaggServkeyManager: SecaggElementTypes.SERVER_KEY,
        #                     SecaggDhManager: SecaggElementTypes.DIFFIE_HELLMAN}

    def tearDown(self) -> None:
        self.patcher_query.stop()
        self.patcher_db.stop()

    def test_channel_manager_01_init(self):
        """Instantiate a channel manager DB table"""

        # 1. successful
        ChannelManager('/path/to/dummy/file')

        # 2. failed
        patcher_db = patch('fedbiomed.common.channel_manager.TinyDB.__init__')
        mock_db = patcher_db.start()
        mock_db.side_effect = Exception

        with self.assertRaises(FedbiomedNodeToNodeError):
            ChannelManager('/path/to/dummy/file')

        # clean
        patcher_db.stop()

    def test_channel_manager_02_list_add_get(self):
        """List elements from table"""

        # 1. successful

        # prepare
        node_id = 'node1'
        dummy_key = b'12345678'

        # action
        cm = ChannelManager('/path/to/dummy/file')
        list1 = cm.list()

        cm.add(node_id, dummy_key)
        list2 = cm.list()
        element2 = cm.get(node_id)

        # check
        self.assertEqual(list1, [])
        self.assertEqual(tuple(list2), (node_id,))
        self.assertTrue(element2)
        self.assertEqual(element2['distant_node_id'], node_id)
        self.assertEqual(element2['local_key'], dummy_key)

        # 2. failed
        cm._table.exception_list = True
        with self.assertRaises(Exception):
            cm.list()
        cm._table.exception_list = False

        cm._table.exception_get = True
        with self.assertRaises(Exception):
            cm.get(node_id)
        cm._table.exception_get = False

        cm._table.exception_upsert = True
        with self.assertRaises(Exception):
            cm.add(node_id, dummy_key)
        cm._table.exception_upsert = False


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

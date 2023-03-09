import unittest
from unittest.mock import patch
import copy
import os

import inspect

from fedbiomed.common.constants import BiprimeType 
from fedbiomed.common.exceptions import FedbiomedSecaggError
from fedbiomed.common.secagg_manager import SecaggServkeyManager, SecaggBiprimeManager


class FakeTinyDB:
    def __init__(self, path):
        self.db_table = None

    def table(self, *args, **kwargs):
        self.db_table = FakeTable()
        return self.db_table


class FakeQuery:
    def __init__(self):
        class FakeSecaggId:
            def __init__(self):
                self.exists_value = True
            def exists(self):
                return self.exists_value
            def one_of(self, id):
                return True
        self.secagg_id = FakeSecaggId()
        class Type():
            def exists(self):
                return True
        self.type = Type()


class FakeTable:
    def __init__(self):
        self.entries = []
        self.exception_insert = False
        self.exception_upsert = False
        self.exception_search = False
        self.exception_remove = False

    def insert(self, entry):
        if self.exception_insert:
            raise FedbiomedSecaggError('mocked exception')
        else:
            self.entries.append(entry)

    def upsert(self, entry, condition):
        if self.exception_upsert:
            raise FedbiomedSecaggError('mocked exception')
        else:
            self.entries.append(entry)


    def search(self, *args, **kwargs):
        if self.exception_search:
            raise FedbiomedSecaggError('mocked exception')
        else:
            return self.entries

    def remove(self, *args, **kwargs):
        if self.exception_remove:
            raise FedbiomedSecaggError('mocked exception')
        else:
            self.entries = []
            return True


class TestBaseSecaggManager(unittest.TestCase):
    """Test for common secagg_manager module"""

    def setUp(self):
        self.patcher_db = patch('fedbiomed.common.secagg_manager.TinyDB', FakeTinyDB)
        self.patcher_query = patch('fedbiomed.common.secagg_manager.Query', FakeQuery)

        self.patcher_db.start()
        self.patcher_query.start()

        self.biprime_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            ),
            "test-data",
            "default_biprimes"
        )

    def tearDown(self) -> None:
        self.patcher_query.stop()
        self.patcher_db.stop()

    def test_secagg_manager_01_init_ok(self):
        """Instantiate SecaggServkeyManager / SecaggBiprimeManager normal successful case"""
        # prepare
        managers = [SecaggServkeyManager, SecaggBiprimeManager]

        # action
        for manager in managers:
            manager('/path/to/dummy/file')

        # test
        # nothing to test at this point ...

    @patch('fedbiomed.common.secagg_manager.TinyDB.__init__')
    def test_secagg_manager_02_init_error(
            self,
            patch_tinydb_init):
        """Instantiate SecaggServkeyManager / SecaggBiprimeManager fails with exception"""
        # prepare
        managers = [SecaggServkeyManager, SecaggBiprimeManager]

        patch_tinydb_init.side_effect = FedbiomedSecaggError

        # action + check
        for manager in managers:
            with self.assertRaises(FedbiomedSecaggError):
                manager('/path/to/dummy/file')

    def test_secagg_manager_03_get_ok(self):
        """Using `get()` method from SecaggServkeyManager / SecaggBiprimeManager successfully"""
        # preparation
        managers = [SecaggServkeyManager, SecaggBiprimeManager]
        entries_list = [
            [[], 'my_dummy_job_id'],
            [[{'job_id': 'my_dummy_job_id'}], 'my_dummy_job_id'],
            [[{'job_id': 33, 'some_more_field': 3}], 33],
        ]

        # action
        for m in managers:
            for entries, job_id in entries_list:
                # preparation (continued)
                manager = m('/path/to/dummy/file')
                # should not be accessing private variable, but got no getter + avoid writing a specific fake class
                # for each test
                manager._db.db_table.entries = entries

                if entries:
                    expected_entries = entries[0]
                else:
                    expected_entries = None
                if m == SecaggServkeyManager:
                    kwargs = {'job_id': job_id}
                else:
                    kwargs = {}

                # action
                get_entries = manager.get('my_secagg_id', **kwargs)

                # check
                self.assertEqual(expected_entries, get_entries)

    def test_secagg_manager_04_get_remove_error_bad_input(self):
        """Using `get()` and `remove()` methods from SecaggManager fails with exception error because of bad arguments
        """
        # 1. common to servkey and biprime

        # preparation
        managers = [SecaggServkeyManager, SecaggBiprimeManager]
        entries_list = [
            [True, [{'job_id': 'my_dummy_job_id'}, {'job_id': 'my_dummy_job_id'} ], 'my_dummy_job_id'],
            [False, [{'job_id': 'my_dummy_job_id'}], 'another_job_id'],
        ]

        # action
        for m in managers:
            for test_for_biprime, entries, job_id in entries_list:
                # preparation (continued)
                if not test_for_biprime and m == SecaggBiprimeManager:
                    continue

                manager = m('/path/to/dummy/file')
                # should not be accessing private variable, but got no getter + avoid writing a specific fake class
                # for each test
                manager._db.db_table.entries = entries

                if m == SecaggServkeyManager:
                    kwargs = {'job_id': job_id}
                else:
                    kwargs = {}

                # action + check
                with self.assertRaises(FedbiomedSecaggError):
                    manager.get('my_secagg_id', **kwargs)

                # action + check
                with self.assertRaises(FedbiomedSecaggError):
                    manager.remove('my_secagg_id', **kwargs)

        # 2. specific remove test for biprime: can't remove default biprimes

        # preparation
        entries_list = [
            [{}],
            [{'type': BiprimeType.DEFAULT.value}],
            [{'other': BiprimeType.DYNAMIC.value}],
        ]

        # action
        for entries in entries_list:
            # preparation (continued)
            manager = SecaggBiprimeManager('/path/to/dummy/file')
            # should not be accessing private variable, but got no getter + avoid writing a specific fake class
            # for each test
            manager._db.db_table.entries = entries

            # action + check
            with self.assertRaises(FedbiomedSecaggError):
                manager.remove('my_secagg_id')

    def test_secagg_manager_05_get_error_table_access_error(self):
        """Using `get()` method from SecaggManager fails with exception error because of table access error"""
        # preparation
        managers = [SecaggServkeyManager, SecaggBiprimeManager]
        entries_list = [
            [[{'job_id': 'my_dummy_job_id'}], 'my_dummy_job_id'],
            [[{'job_id': 245, 'another': 'field'}], 245],
        ]

        # action
        for m in managers:
            for entries, job_id in entries_list:
                # preparation (continued)
                manager = m('/path/to/dummy/file')
                # should not be accessing private variable, but avoids writing a specific fake class
                # for each test
                manager._db.db_table.exception_search = True

                if m == SecaggServkeyManager:
                    kwargs = {'job_id': job_id}
                else:
                    kwargs = {}

                # action + check
                with self.assertRaises(FedbiomedSecaggError):
                    manager.get('my_secagg_id', **kwargs)

    def test_secagg_manager_06_add_ok_remove_ok(self):
        """Using `add()` and `remove()` methods from SecaggManager successfully"""
        # preparation
        # nota: type of inputs not checked by this class
        secagg_id_list = ['one', 'another', 222]
        parties_list = [['r', 'n1', 'n2'], ['r', 'n1', 'n2', 'n3', 'n4', 'n5'], 111, []]

        specific_list = [
            [SecaggServkeyManager, {'job_id': 'my_job_id_dummy', 'context': '123456789'}, {'job_id': 'my_job_id_dummy'}],
            [SecaggBiprimeManager, {'context': 'a_long_dummy_biprime'}, {}],
        ]

        # action
        for secagg_id in secagg_id_list:
            for parties in parties_list:
                for m, specific, kwargs in specific_list:
                    # preparation (continued)
                    manager = m('/path/to/dummy/file')
                    expected_entries = copy.deepcopy(specific)
                    expected_entries.update({'secagg_id': secagg_id, 'parties': parties})

                    # action
                    manager.add(secagg_id, parties, **specific)
                    get_entry = manager.get(secagg_id, **kwargs)

                    # check
                    self.assertEqual(expected_entries, get_entry)

                    # action
                    removed = manager.remove(secagg_id, **kwargs)
                    get_entry = manager.get(secagg_id, **kwargs)

                    # check
                    self.assertEqual(removed, True)
                    self.assertEqual(None, get_entry)

                    # action
                    removed = manager.remove(secagg_id, **kwargs)
                    get_entry = manager.get(secagg_id, **kwargs)

                    # check
                    self.assertEqual(removed, False)
                    self.assertEqual(None, get_entry)

    def test_secagg_manager_07_add_error_re_inserting(self):
        """Using `add()` method from SecaggManager with error re-inserting entry"""
        # preparation
        # nota: type of inputs not checked by this class
        secagg_id_list = ['one', 'another', 222]
        parties_list = [['r', 'n1', 'n2'], ['r', 'n1', 'n2', 'n3', 'n4', 'n5'], 111, []]

        specific_list = [
            [
                SecaggServkeyManager,
                {'job_id': 'my_job_id_dummy', 'context': '123456789'},
                {'job_id': 'my_job__alternate_id_dummy', 'context': '987654321'},
                {'job_id': 'my_job_id_dummy'}
            ],
            [
                SecaggBiprimeManager,
                {'context': 'a_long_dummy_biprime'},
                {'context': 'a_long_dummy_alternate_biprime'},
                {}
            ],
        ]

        # action
        for secagg_id in secagg_id_list:
            for parties in parties_list:
                for m, specific, alt_specific, kwargs in specific_list:
                    # preparation (continued)
                    manager = m('/path/to/dummy/file')
                    manager.add(secagg_id, parties, **specific)

                    # action + check
                    with self.assertRaises(FedbiomedSecaggError):
                        manager.add(secagg_id, parties, **specific)
                    with self.assertRaises(FedbiomedSecaggError):
                        manager.add(secagg_id, ['some', 'other', 'parties'], **specific)
                    with self.assertRaises(FedbiomedSecaggError):
                        manager.add(secagg_id, ['some', 'other', 'parties'], **alt_specific)

    def test_secagg_manager_08_add_table_access_error(self):
        """Using `add()` method from SecaggManager fails with exception in table access"""
        # preparation
        # nota: type of inputs not checked by this class
        secagg_id_list = ['one', 'another', 222]
        parties_list = [['r', 'n1', 'n2'], ['r', 'n1', 'n2', 'n3', 'n4', 'n5'], 111, []]

        specific_list = [
            [SecaggServkeyManager, {'job_id': 'my_job_id_dummy', 'context': '123456789'}, {'job_id': 'my_job_id_dummy'}],
            [SecaggBiprimeManager, {'context': 'a_long_dummy_biprime'}, {}],
        ]

        # action
        for secagg_id in secagg_id_list:
            for parties in parties_list:
                for m, specific, kwargs in specific_list:
                    # preparation (continued)
                    manager = m('/path/to/dummy/file')
                    # should not be accessing private variable, but avoids writing a specific fake class
                    # for each test
                    manager._db.db_table.exception_insert = True

                    # action + check
                    with self.assertRaises(FedbiomedSecaggError):
                        manager.add(secagg_id, parties, **specific)

                    # preparation (continued)
                    # should not be accessing private variable, but avoids writing a specific fake class
                    # for each test
                    manager._db.db_table.exception_insert = False
                    manager._db.db_table.exception_remove = True
                    manager.add(secagg_id, parties, **specific)

                    # action + check
                    with self.assertRaises(FedbiomedSecaggError):
                        manager.remove('my_secagg_id', **kwargs)

    def test_secagg_manager_08_update_default_biprimes_ok(self):
        """Testing successful update of the default biprimes in database
        """
        # 1. use default biprimes

        # prepare
        bpm = SecaggBiprimeManager('/path/to/dummy/file')
        bpm._table.insert({'secagg_id': 'ANOTHER'})

        # test
        bpm.update_default_biprimes(True, self.biprime_dir)

        # check
        #
        # checks are done accordingly to default_biprimes dir content
        self.assertEqual(
            bpm._table.entries,
            [{'secagg_id': 'dummy_biprime', 'parties': None, 'type': 'default',
                'context': {'biprime': 12345678, 'max_keysize': 33}}]
        )

        # 2. don't use default biprimes

        # prepare
        bpm._query.secagg_id.exists = False

        # test
        bpm.update_default_biprimes(False, self.biprime_dir)

        # check
        #
        # checks default biprimes from previous test were removed as we don't use default biprimes
        self.assertEqual(bpm._table.entries, [])

    def test_secagg_manager_08_update_default_biprimes_fail(self):
        """Testing failed update of the default biprimes in database
        """
        # prepare
        bpm = SecaggBiprimeManager('/path/to/dummy/file')

        for exception in ['bpm._table.exception_search', 'bpm._table.exception_remove', 'bpm._table.exception_upsert']:
            exec(f'{exception} = True')

            # test
            with self.assertRaises(FedbiomedSecaggError):
                bpm.update_default_biprimes(True, self.biprime_dir)

            # clean
            exec(f'{exception} = False')

    def test_secagg_manager_08_read_default_biprimes_fail(self):
        """Testing failed reading of the default biprimes files
        """
        # prepare
        bpm = SecaggBiprimeManager('/path/to/dummy/file')

        with patch('builtins.open', return_value=Exception):
            # test
            with self.assertRaises(FedbiomedSecaggError):
                bpm._read_default_biprimes(self.biprime_dir)

        for bad_json in [
                None,
                3,
                [],
                {},
                { 'secagg_id': 1, 'biprime': 2, 'max_keysize': 3},
                { 'secagg_id': '1', 'biprime': '2', 'max_keysize': 3},
                { 'secagg_id': '1', 'biprime': 2, 'max_keysize': '3'},
                { 'secagg_id': '', 'biprime': 2, 'max_keysize': 3},
        ]:
            with patch('json.load', return_value=bad_json):
                # test
                with self.assertRaises(FedbiomedSecaggError):
                    bpm._read_default_biprimes(self.biprime_dir)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

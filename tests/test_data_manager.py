# Managing NODE, RESEARCHER environ mock before running tests

import copy
from re import search
from typing import Dict, List

from unittest import mock

import numpy as np
import pandas as pd
from testsupport.delete_environ import delete_environ
# Detele environ. It is necessary to rebuild environ for required component
delete_environ()
import testsupport.mock_common_environ
# Import environ for node since test will be runing for node component
from fedbiomed.node.environ    import environ


from fedbiomed.node.data_manager import DataManager
import unittest
from unittest.mock import MagicMock, patch
import os

import inspect
import torch
from torch.utils.data import Dataset

from tests.testsupport.fake_uuid import FakeUuid

print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))


class TestDataManager(unittest.TestCase):
    """
    Test dataset loading
    Args:
        unittest ([type]): [description]
    """
    class FakeDataset(Dataset):
            """This class present a very simple custom dataset,
            that should be used within PyTorch framework"""
            def __init__(self, data, labels):
                self._data = data
                self._labels = labels
                
            def __len__(self):
                return len(self._data)
            
            def __getitem__(self, idx):
                return self._data[idx], self._labels[idx]
     
    @classmethod
    def setUpClass(cls) -> None:
        
        # creating a fake pytorch dataset:
        fake_dataset_shape = (12_345, 10, 20, 30)
        fake_data = torch.rand(fake_dataset_shape)
        fake_labels = torch.randint(0,2, (fake_dataset_shape[0],))
        fake_dataset = TestDataManager.FakeDataset(fake_data, fake_labels)
        cls.fake_dataset_shape = fake_dataset_shape
        cls.fake_dataset = fake_dataset  # we might need a fake dataset 
        # for testing
        
        # dummy_data for pandas dataframe stuff
        dummy_data = {
                'integers': [1,2,3,4,5,6,7,8,9,0],
                'floats': [1.1, 1.2, 1.3, 1.4, 1.5, 2.6, 2.7, 1.8, 2.9, 1.0],
                'chars': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                'booleans': [True, False, True, False, True] * 2
                }
        cls.dummy_data = dummy_data

    # Setup data manager
    def setUp(self):

        self.testdir = os.path.join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
                ),
            "test-data"
            )

        self.data_manager = DataManager()
        
        # arguments
        self.fake_database = {"1": {"name": "MNIST",
                               "data_type": "default",
                               "tags": ["#MNIST", "#dataset"],
                               "description": "MNIST database",
                               "shape": [60000, 1, 28, 28],
                               "path": "/path/to/MNIST", 
                               "dataset_id": "dataset_1234",
                               "dtypes": []},
                           "2": {"name": "test",
                                 "data_type": "csv",
                                 "tags": ["some", "tags"],
                                 "description": "test",
                                 "shape": [1000,2],
                                 "path": "/path/to/my/data",
                                 "dataset_id": "dataset_4567",
                                 "dtypes": ["float64", "int64"]}}

    # after the tests
    def tearDown(self):
        self.data_manager.db.close()
        os.remove(environ['DB_PATH'])
        pass

    @patch('tinydb.table.Table.clear_cache')
    def test_data_manager_01_search_by_id(self,
                                          tinydb_cache_patch,):
        
        """tests `search_by_id` method with a non existing id
        """
        tinydb_cache_patch.return_value = None
        # action (should retrun an empty array)
        res = self.data_manager.search_by_id('dataset_id_1234')
        self.assertEqual(res, [])
        
    @patch('tinydb.queries.Query.all')
    @patch('tinydb.table.Table.search')    
    @patch('tinydb.table.Table.clear_cache')
    def test_data_manager_02_search_by_id(self,
                                          tinydb_cache_patch,
                                          tinydb_search_patch,
                                          queries_all_patch):
        """Simulates a query with a correct dataset id by patching Query and 
        Table.search object/methods"""
        # arguments
        search_results = [{'dataset_id':'datset_id_1234'}]
        dataset_id = 'dataset_id_1234'
        # patches
        tinydb_cache_patch.return_value = None
        tinydb_search_patch.return_value = search_results
        
        # action
        res = self.data_manager.search_by_id(dataset_id)
        
        # checks
        self.assertEqual(search_results, res)
        queries_all_patch.assert_called_once_with(dataset_id)
        tinydb_search_patch.assert_called_once()
        
    def test_data_manager_03_search_by_tags(self):
        """tests `search_by_tags` with a non existing tags
        """
        res = self.data_manager.search_by_tags('dataset_id_1234')
        self.assertEqual(res, [])
    
    @patch('tinydb.queries.Query.all')      
    @patch('tinydb.table.Table.search') 
    @patch('tinydb.table.Table.clear_cache')
    def test_data_manager_04_search_by_tags(self,
                                            tinydb_cache_patch,
                                            tinydb_search_patch,
                                            queries_all_patch
                                            ):
        """Simulates a query with correct dataset tags by patching Query and 
        Table.search object/methods"""
        # arguments
        search_results = [{'dataset_id': 'datset_id_1234'}]
        dataset_tags = ['some', 'tags']
        
        # patches
        tinydb_cache_patch.return_value = None
        tinydb_search_patch.return_value = search_results
        
        # action
        res = self.data_manager.search_by_tags(dataset_tags)
        
        # checks
        self.assertEqual(search_results, res)
        queries_all_patch.assert_called_once_with(dataset_tags)
        tinydb_search_patch.assert_called_once()
    
    def test_data_manager_05_read_csv_with_header(self):
        """Tests if `read_csv` method is able to identify and parse 
        the csv file 'tata-header.csv' (containing a header)
        """
        # action
        res = self.data_manager.read_csv(os.path.join(
                                                    self.testdir,
                                                    "csv",
                                                    "tata-header.csv"
                                                    )
                                         )
        # checks if the file has been correctly parsed
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(list(res.columns), ['Titi','Tata','Toto'])
        
    def test_data_manager_06_red_csv_without_header(self):
        """Tests if `read_csv` method is able to identify and parse 
        the csv file 'titi-normal.csv' (don't contain any headers)
        """
        # action
        res = self.data_manager.read_csv(os.path.join(
                                                    self.testdir,
                                                    "csv",
                                                    "titi-normal.csv"
                                                    )
                                         )
        # checks if the file has been correctly parsed
        self.assertIsInstance(res, pd.DataFrame)
        # when there are no headers detected in csv file, pandas automatically
        # creates headers (integers from 0 to the number of columns)
        # in the assertion below, we are testing that headers will be
        # auomatically created and named [0,1,2,3]
        self.assertEqual(list(res.columns), [0, 1, 2, 3])

    def test_data_manager_07_get_torch_dataset_shape(self):
        """Tests if method `get_torch_dataset_shape` works
        on a custom dataset"""
        
        # creating agruments
        
        
        # action
        res = self.data_manager.get_torch_dataset_shape(self.fake_dataset)
        
        #checks
        self.assertEqual(res, list(self.fake_dataset_shape))
    
    def test_data_manager_08_get_csv_data_types(self):
        "Tests `get_csv_data_type` (norma case scenario)"
        # creating argument for unittest
        
        
        fake_csv_dataframe = pd.DataFrame(self.dummy_data)
        
        # action
        data_types = self.data_manager.get_csv_data_types(fake_csv_dataframe)
        
        # checks
        self.assertEqual(data_types, ['int64', 'float64', 'object', 'bool'])
    
    @patch('torchvision.datasets.MNIST')
    def test_data_manager_09_load_default_database_as_dataset(self,
                                                              dataset_mnist_patch):
        """Tests if `load_default_dataset` is loading the default dataset and
        returns it (arg `as_dataset` set to True)"""

        # defining patcher
        dataset_mnist_patch.return_value = self.fake_dataset
        
        # defining arguments
        database_name = 'MNIST'
        database_path = '/path/to/MNIST/dataset'
        # action
        # currently, only MNIST dataset is considered as the default dataset
        
        res_dataset = self.data_manager.load_default_database(database_name,
                                                              database_path,
                                                              as_dataset=True)
        # checks
        self.assertEqual(res_dataset, self.fake_dataset)
        dataset_mnist_patch.assett_called_once_with(root=database_path,
                                                    download=True,
                                                    transform=mock.ANY)
        
    @patch('fedbiomed.node.data_manager.DataManager.get_torch_dataset_shape')
    @patch('torchvision.datasets.MNIST')
    def test_data_manager_10_load_default_database_as_dataset_false(self,
                                                                    dataset_mnist_patch,
                                                                    load_default_database_patch):
        """Tests if `load_default_dataset` is loading the default dataset and
        returns its shape (arg `as_dataset` set to False)"""
        dataset_mnist_patch.return_value = self.fake_dataset
        
        # defining arguments
        database_name = 'MNIST'
        database_path = '/path/to/MNIST/dataset'
        
        # action
        res_dataset = self.data_manager.load_default_database(
                                                            database_name,
                                                            database_path,
                                                            as_dataset=False)
        # checks
        dataset_mnist_patch.assert_called_once_with(root=database_path,
                                                    download=True,
                                                    transform=mock.ANY)
    
        load_default_database_patch.assert_called_once_with(self.fake_dataset)
    
    def test_data_manager_11_load_default_database_exception(self):
        """Tests if exception `NotImplemntedError` is triggered 
        when passing an unknown dataset
        """
        with self.assertRaises(NotImplementedError):
            # action: we are here passing an unknown dataset
            self.data_manager.load_default_database('my_default_dataset',
                                                    '/path/to/my/default/dataset')
    
    @patch('torchvision.datasets.ImageFolder')
    def test_data_manager_12_load_images_dataset_as_dataset_true(self, imgfolder_patch):
        """Tests case where one is loading image dataset with argument
        `as_dataset` is set to True"""
        # defining patcher
        imgfolder_patch.return_value = self.fake_dataset
        
        # arguments
        database_path = '/path/to/MNIST/dataset'
        # action
        dataset = self.data_manager.load_images_dataset(database_path, as_dataset=True)
        
        # checks
        self.assertEqual(dataset, self.fake_dataset)
        imgfolder_patch.assert_called_once_with(database_path,
                                                transform=mock.ANY)
        
    def test_data_manager_13_load_images_dataset_as_dataset_false(self):
        """Tests case where one is loading image dataset with argument 
        `as_dataset` is set to False"""
 
        # arguments
        dataset_path = os.path.join( self.testdir,
                                        "images"
                                    )

        # action
        res_dataset_shape = self.data_manager.load_images_dataset(dataset_path,
                                                                  as_dataset=False)

        # checks
        self.assertEqual(res_dataset_shape, [5, 3, 30, 60])
        
    @patch('fedbiomed.node.data_manager.DataManager.read_csv')    
    def test_data_manager_14_load_csv_dataset(self, read_csv_patch):
        """Tests `load_csv_method` (normal case scenario)"""
        
        # patcher
        dummy_data = pd.DataFrame(self.dummy_data)
        read_csv_patch.return_value = dummy_data
        # arguments
        database_path = '/path/to/MNIST/dataset'
        
        # action
        data = self.data_manager.load_csv_dataset(database_path)
        
        # checks
        read_csv_patch.assert_called_once_with(database_path)
        # if below assertion is True, will return None, ottherwise, 
        # if False, triggers an error
        self.assertTrue(np.testing.assert_array_equal(data.values,
                                                      dummy_data.values) == None)
    
    @patch('tinydb.table.Table.insert')
    @patch('fedbiomed.node.data_manager.DataManager.load_default_database')
    @patch('os.path.isdir')
    def test_data_manager_15_add_database_default_dataset(self,
                                                          os_listdir_patch,
                                                          datamanager_load_default_dataset_patch,
                                                          insert_table_patch):
        # unit test parameters
        fake_dataset_shape = copy.deepcopy(self.fake_dataset_shape)
        fake_dataset_shape = list(fake_dataset_shape)
        fake_dataset_path = '/path/to/some/dataset'
        fake_dataset_id = 'dataset_id_12234'
        fake_dataset_name = 'test'
        # patchers
        os_listdir_patch.return_value = True
        datamanager_load_default_dataset_patch.return_value = fake_dataset_shape
        insert_table_patch.return_value = None
        # action
        dataset_id = self.data_manager.add_database(name=fake_dataset_name,
                                                    data_type='default',
                                                    tags=['unit', 'test'],
                                                    description='some description',
                                                    path= fake_dataset_path,
                                                    dataset_id=fake_dataset_id
                                                    )
        # checks
        self.assertEqual(dataset_id, fake_dataset_id)
        datamanager_load_default_dataset_patch.assert_called_once_with(fake_dataset_name,
                                                                       fake_dataset_path)
        
    def test_data_manager_16_add_database_real_csv_examples_based(self):
        """ Test add_database method for loading real csv datasets """

        fake_dataset_id = 'dataset_id_1232345'
        # Load data with header example
        dataset_id = self.data_manager.add_database(name='test',
                                                    tags=['titi'],
                                                    data_type='csv',
                                                    description='description',
                                                    path=os.path.join( self.testdir,
                                                                    "csv",
                                                                    "tata-header.csv"
                                                                    ),
                                                    dataset_id=fake_dataset_id
                                                    )

        self.assertEqual(dataset_id, fake_dataset_id)
        # Should raise error due to same tag
        with self.assertRaises(Exception):
            self.data_manager.add_database( name='test',
                                            tags=['titi'],
                                            data_type='csv',
                                            description='description',
                                            path=os.path.join( self.testdir,
                                                               "csv",
                                                               "tata-header.csv"
                                                              )
                                           )

        # Load data with normal different types
        self.data_manager.add_database( name='test',
                                        tags=['tata'],
                                        data_type='csv',
                                        description='description',
                                        path=os.path.join( self.testdir,
                                                           "csv",
                                                           "titi-normal.csv"
                                                          )
                                       )

        # Should raise error due to broken csv
        with self.assertRaises(Exception):
            self.data_manager.add_database( name='test',
                                            tags=['tutu'],
                                            data_type='csv',
                                            description='description',
                                            path=os.path.join( self.testdir,
                                                               "csv",
                                                               "toto-error.csv"
                                                              )
                                           )
            
        # should raise n error because it is not a csv file (but data_format passed
        # as 'csv' file)
        with self.assertRaises(AssertionError):
            dataset_id = self.data_manager.add_database(name='test',
                                                        tags=['titi'],
                                                        data_type='csv',
                                                        description='description',
                                                        path=os.path.join( self.testdir,
                                                                        "images"
                                                                        )
                                                    )
    
    def test_data_manager_17_add_database_wrong_datatype(self):
        """Tests if NotImplementedError is raised when specifying
        an unknown data type in add_database method"""
        with self.assertRaises(NotImplementedError):
            self.data_manager.add_database(name='test',
                                           data_type='unknwon format',
                                           tags=['test'],
                                           description='this is a test',
                                           path='/a/path/to/some/data')
        
    @patch('uuid.uuid4')
    def test_data_manager_18_add_database_real_images_example_based(self, uuid_patch):

        """ Test data_manager method for loading real images datasets """
        # patchers:
        uuid_patch.return_value = FakeUuid()
        
        # Load data with header example
        dataset_id = self.data_manager.add_database(name='test',
                                                    tags=['titi'],
                                                    data_type='images',
                                                    description='description',
                                                    path=os.path.join( self.testdir,
                                                                    "images"
                                                                    )
                                                )
        # check if dataset_id is correct when none is passed as argument
        self.assertEqual('dataset_' + str(FakeUuid.VALUE), dataset_id)
        
        # Should raise error due to same tag
        with self.assertRaises(Exception):
            self.data_manager.add_database( name='test',
                                            tags=['titi'],
                                            data_type='images',
                                            description='description',
                                            path=os.path.join( self.testdir,
                                                               "images"
                                                              )
                                           )
            
        # should raise error because a csv dataset is loaded as image
        
        with self.assertRaises(AssertionError):
            self.data_manager.add_database(name='test',
                                            tags=['titi'],
                                            data_type='images',
                                            description='description',
                                            path=os.path.join( self.testdir,
                                                            "csv",
                                                            "tata-header.csv"
                                                            ),
                                                    )
    @patch('tinydb.table.Table.remove')
    @patch('fedbiomed.node.data_manager.DataManager.search_by_tags')
    def test_data_manager_19_remove_database(self,
                                             search_by_tags_patch,
                                             db_remove_patch):
        """Tests `remove_database` an simulates the removal of a database
        through its tags.
        """
        # arguments
        doc1 = MagicMock(doc_id=1)  # adding the attribute `doc_id` to doc
        # (usually, datasets are added with an integer from 1 to the number of datasets 
        # contained in the database)
        
        dataset_tags = ['some', 'tags']
        search_result = [doc1]
        
        fake_database = copy.deepcopy(self.fake_database)
        
        # side effect functions
        
        def db_remove_side_effect(doc_ids: List[int]):
            """Removes from `fake_database entries contained
            in doc_ids 
            Args: 
            doc_ids (List[int]): list of doc ids
            """
            for doc_id in doc_ids:
                fake_database.pop(str(doc_id))
                
        
        # patchers
        search_by_tags_patch.return_value = search_result
        db_remove_patch.side_effect = db_remove_side_effect
        
        # action
        self.data_manager.remove_database(dataset_tags)
        
        # checks
        search_by_tags_patch.assert_called_once_with(dataset_tags)
        db_remove_patch.assert_called_once_with(doc_ids=[1])
        self.assertFalse(fake_database.get('1', False))

    @patch('tinydb.table.Table.update')
    @patch('tinydb.queries.Query.all')
    def test_data_manager_20_modify_database_info(self,
                                                  query_all_patch,
                                                  tinydb_update_patch):
        
        """Tests modify_databae_info (normal case scenario), 
        where one replaces an existing dataset by another one
        """
        fake_database = copy.deepcopy(self.fake_database)
        
        # side effect function
        def tinydb_update_side_effect(new_dataset: dict, existing_dataset: List[int]):
            """side effect function that mimicks the update of the database 
            `fake_database`

            Args:
                new_dataset (dict): the new dataset to update
                existing_dataset (List[int]): unused, but should 
                be a list of doc ids that corresponds to the output 
                `Query.tags.all(tags)` dataset query.
            """
            fake_database['2'] = new_dataset
            
        new_dataset = {"name": "anther_test",
                        "data_type": "csv",
                        "tags": [ "other_tags"],
                        "description": "another_description",
                        "shape": [2000,2],
                        "path": "/path/to/my/other/data",
                        "dataset_id": "dataset_9876",
                        "dtypes": ["int64", "float64"]}
        
        tags = ['some', 'tags']
        query_all_patch.return_value = [fake_database.get('2')]
        tinydb_update_patch.side_effect = tinydb_update_side_effect
        
        # action
        self.data_manager.modify_database_info(tags, new_dataset)
        
        # checks
        # check that correct calls are made
        query_all_patch.assert_called_once_with(tags)
        tinydb_update_patch.assert_called_once_with(new_dataset, 
                                                    [self.fake_database.get('2')])
        # check database status after updating
        # first entry in database should be left unchanged ...
        self.assertEqual(fake_database.get('1'), self.fake_database.get('1'))
        # ... whereas second entry should be updated
        self.assertNotEqual(fake_database.get('2'), self.fake_database.get('2'))
        self.assertEqual(fake_database.get('2'), new_dataset)
        
    @patch('tinydb.table.Table.all')
    @patch('tinydb.table.Table.clear_cache')
    def test_data_manager_21_list_my_data(self,
                                          clear_cache_patch,
                                          query_all_patch):
        # arguments
        
        table_all_query = [{"name": "MNIST",
                               "data_type": "default",
                               "tags": ["#MNIST", "#dataset"],
                               "description": "MNIST database",
                               "shape": [60000, 1, 28, 28],
                               "path": "/path/to/MNIST", 
                               "dataset_id": "dataset_1234",
                               "dtypes": []},
                           {"name": "test",
                                 "data_type": "csv",
                                 "tags": ["some", "tags"],
                                 "description": "test",
                                 "shape": [1000,2],
                                 "path": "/path/to/my/data",
                                 "dataset_id": "dataset_4567",
                                 "dtypes": ["float64", "int64"]}
                           ]
        
        # patchers
        clear_cache_patch.return_value = None
        query_all_patch.return_value = table_all_query
        
        # action
        all_data = self.data_manager.list_my_data(True)
        
        # checks
        query_all_patch.assert_called_once()
        clear_cache_patch.assert_called_once()
        # check that none of the database contained on the node 
        # doesnot contain `dtype`entry
        self.assertNotIn('types', table_all_query[0].keys())
        self.assertNotIn('types', table_all_query[1].keys())
        
if __name__ == '__main__':  # pragma: no cover
    unittest.main()

# Managing NODE, RESEARCHER environ mock before running tests

from re import search

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
from unittest.mock import patch
import os

import inspect
import torch
from torch.utils.data import Dataset

print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))


class TestDataManager(unittest.TestCase):
    """
    Test dataset loading
    Args:
        unittest ([type]): [description]
    """
    class FakeQuery:
        DATASET_ID = 'dataset_id_1234'
        def __init__(self):
            self.dataset_id = self.DATASET_ID
            
    class DataSetID:
        def __init__(self):
            pass
        def all(self, dataet_id:str):
            pass
    # Setup data manager
    def setUp(self):

        self.testdir = os.path.join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
                ),
            "test-data"
            )

        self.data_manager = DataManager()
        pass

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
        search_results = [{'dataset_id':'datset_id_1234'}]
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
        on a handmade dataset"""
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

        # creating agruments
        fake_dataset_shape = (12_345, 10, 20, 30)
        fake_data = torch.rand(fake_dataset_shape)
        fake_labels = torch.randint(0,2, (fake_dataset_shape[0],))
        fake_dataset = FakeDataset(fake_data, fake_labels)
        
        # action
        res = self.data_manager.get_torch_dataset_shape(fake_dataset)
        
        #checks
        self.assertEqual(res, list(fake_dataset_shape))
    
    def test_data_manager_08_get_csv_data_types(self):
        # creating argument for unittest
        
        data = {'integers': [1,2,3,4,5,6,7,8,9,0],
                'floats': [1.1, 1.2, 1.3, 1.4, 1.5, 2.6, 2.7, 1.8, 2.9, 1.0],
                'chars': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                'booleans': [True, False, True, False, True] * 2
                }
        fake_csv_dataframe = pd.DataFrame(data)
        
        # action
        data_types = self.data_manager.get_csv_data_types(fake_csv_dataframe)
        
        # checks
        self.assertEqual(data_types, ['int64', 'float64', 'object', 'bool'])
    
    def test_data_manager_XX_add_database(self):

        """ Test add_database method for loading csv datasets """

        # Load data with header example
        self.data_manager.add_database( name='test',
                                        tags=['titi'],
                                        data_type='csv',
                                        description='description',
                                        path=os.path.join( self.testdir,
                                                           "csv",
                                                           "tata-header.csv"
                                                          )
                                        )

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


    def test_load_image_dataset(self):

        """ Test function for loading image dataset """

        # Load data with header example
        self.data_manager.add_database( name='test',
                                        tags=['titi'],
                                        data_type='images',
                                        description='description',
                                        path=os.path.join( self.testdir,
                                                           "images"
                                                          )
                                       )

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
            pass

if __name__ == '__main__':  # pragma: no cover
    unittest.main()

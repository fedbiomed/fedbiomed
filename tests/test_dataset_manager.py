import copy
import inspect
import numpy as np
import os
import pandas as pd
import shutil
from typing import List
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch
from torch.utils.data import Dataset
import tempfile
import pathlib
from PIL import Image
import torch
from torchvision import transforms, datasets
from tinydb import Query

#############################################################
# Import NodeTestCase before importing any FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################

# Test Support
from testsupport.fake_uuid import FakeUuid
from testsupport.testing_data_loading_block import LoadingBlockTypesForTesting

from fedbiomed.node.environ import environ
from fedbiomed.node.dataset_manager import DatasetManager, DataLoadingPlan
from fedbiomed.common.exceptions import FedbiomedDatasetManagerError


class TestDatasetManager(NodeTestCase):
    """
    Unit Tests for DatasetManager class.
    """
    class FakePytorchDataset(Dataset):
        """
        This class fakes a very simple custom dataset (data
        structure), that should be used within PyTorch framework.
        It can be used to mimic MNIST and images datasets.

        For further information, please visit Pytorch documentation.
        """
        def __init__(self, data, labels):
            self._data = data
            self._labels = labels

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx], self._labels[idx]

    def setUp(self):
        """
        run this at the begining of each test

        get the path of the test data folder (containing real data
        for the test)
        """
        self.testdir = os.path.join(
            os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            ),
            "test-data"
        )

        # create an instance of DatasetManager
        self.patcher_dataset_manager_environ = patch('fedbiomed.node.dataset_manager.environ', environ)
        self.patcher_dataset_manager_environ.start()
        self.dataset_manager = DatasetManager()

        # fake arguments
        # fake_database attribute fakes the resut of query over
        # a database (eg `tinydb.queries.Query.all`)
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
                                    "shape": [1000, 2],
                                    "path": "/path/to/my/data",
                                    "dataset_id": "dataset_4567",
                                    "dtypes": ["float64", "int64"]}
                              }

        # creating a fake pytorch dataset:
        self.fake_dataset_shape = (12_345, 10, 20, 30)
        fake_data = torch.rand(self.fake_dataset_shape)
        fake_labels = torch.randint(0, 2, (self.fake_dataset_shape[0],))
        self.fake_dataset = TestDatasetManager.FakePytorchDataset(fake_data, fake_labels)


        # dummy_data for pandas dataframe stuff
        self.dummy_data = {'integers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                           'floats': [1.1, 1.2, 1.3, 1.4, 1.5, 2.6, 2.7, 1.8, 2.9, 1.0],
                           'chars': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                           'booleans': [True, False, True, False, True] * 2
                           }

        self.tempdir = tempfile.mkdtemp()  # Creates and returns tempdir to save temp images

    def tearDown(self):
        """
        after each test function
        """
        self.patcher_dataset_manager_environ.stop()

        self.dataset_manager._db.close()
        del self.dataset_manager
        if os.path.isdir(environ['DB_PATH']):
            os.remove(environ['DB_PATH'])

        shutil.rmtree(self.tempdir)

    def test_dataset_manager_01_get_by_id_non_existing_dataset_id(self):
        """
        Test `search_by_id` method with a non existing id
        """
        # action (should return an empty array)
        res = self.dataset_manager.get_by_id('dataset_id_1234')
        self.assertEqual(res, None)


    @patch('tinydb.table.Table.get')
    def test_dataset_manager_02_get_by_id(self,
                                          tinydb_get_patch):
        """
        Simulates a query with a correct dataset id by patching Query and
        Table.search object/methods
        """
        # arguments
        search_results = {'dataset_id': 'datset_id_1234'}
        dataset_id = 'dataset_id_1234'
        # patches
        tinydb_get_patch.return_value = search_results

        # action
        res = self.dataset_manager.get_by_id(dataset_id)

        # checks
        self.assertEqual(search_results, res)
        tinydb_get_patch.assert_called_once()

    def test_dataset_manager_03_search_by_tags(self):
        """
        tests `search_by_tags` method with non existing tags
        """
        res = self.dataset_manager.search_by_tags('dataset_id_1234')
        # method `search_by_tags` should return an empty list
        self.assertEqual(res, [])


    @patch('tinydb.queries.Query.all')
    @patch('tinydb.table.Table.search')
    def test_dataset_manager_04_search_by_tags(self,
                                               tinydb_search_patch,
                                               queries_all_patch
                                               ):
        """
        Simulates a query with correct dataset tags by patching Query and
        Table.search object/methods
        """
        # arguments
        search_results = [{'dataset_id': 'datset_id_1234'}]
        dataset_tags = ['some', 'tags']

        # patches
        tinydb_search_patch.return_value = search_results

        # action
        res = self.dataset_manager.search_by_tags(dataset_tags)

        # checks
        self.assertEqual(search_results, res)
        queries_all_patch.assert_called_once_with(dataset_tags)
        tinydb_search_patch.assert_called_once()


    def test_dataset_manager_05_read_csv_with_header(self):
        """
        Tests if `read_csv` method is able to identify and parse
        the csv file 'tata-header.csv' (containing a header)
        """
        # action
        res = self.dataset_manager.read_csv(
            os.path.join(self.testdir,
                         "csv",
                         "tata-header.csv"
                         )
        )
        # checks if the file has been correctly parsed
        self.assertIsInstance(res, pd.DataFrame)
        self.assertListEqual(list(res.columns), ['Titi', 'Tata', 'Toto'])


    def test_dataset_manager_06_red_csv_without_header(self):
        """
        Tests if `read_csv` method is able to identify and parse
        the csv file 'titi-normal.csv' (which does not contain a headers)
        """
        # action
        res = self.dataset_manager.read_csv(os.path.join(self.testdir,
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
        self.assertListEqual(list(res.columns), [0, 1, 2, 3])


    def test_dataset_manager_07_get_torch_dataset_shape(self):
        """
        Tests if method `get_torch_dataset_shape` works
        on a custom dataset
        """

        # action
        res = self.dataset_manager.get_torch_dataset_shape(self.fake_dataset)

        # checks
        self.assertListEqual(res, list(self.fake_dataset_shape))


    def test_dataset_manager_08_get_csv_data_types(self):
        """
        Tests `get_csv_data_type` (normal case scenario)
        """
        # creating argument for unittest
        fake_csv_dataframe = pd.DataFrame(self.dummy_data)

        # action
        data_types = self.dataset_manager.get_csv_data_types(fake_csv_dataframe)

        # checks
        self.assertListEqual(data_types, ['int64', 'float64', 'object', 'bool'])

    @patch('torchvision.datasets.MNIST')
    def test_dataset_manager_09_load_default_database_as_dataset(self,
                                                                 dataset_mnist_patch):
        """
        Tests if `load_default_dataset` is loading the default dataset and
        returns it (arg `as_dataset` set to True)
        """

        # defining patcher
        dataset_mnist_patch.return_value = self.fake_dataset

        # defining arguments
        database_name = 'MNIST'
        database_path = '/path/to/MNIST/dataset'
        # action
        # currently, only MNIST dataset is considered as the default dataset

        res_dataset = self.dataset_manager.load_default_database(database_name,
                                                              database_path,
                                                              as_dataset=True)
        # checks
        self.assertEqual(res_dataset, self.fake_dataset)
        # Below, we are not testing that MNIST patch has been calling
        # with the good argument for 'transform``
        dataset_mnist_patch.assert_called_once_with(root=database_path,
                                                    download=True,
                                                    transform=mock.ANY)

    @patch('fedbiomed.node.dataset_manager.DatasetManager.get_torch_dataset_shape')
    @patch('torchvision.datasets.MNIST')
    def test_dataset_manager_10_load_default_database_as_dataset_false(self,
                                                                    dataset_mnist_patch,
                                                                    get_torch_dataset_shape_patch,
                                                                    ):
        """
        Tests if `load_default_dataset` is loading the default dataset and
        returns its shape (arg `as_dataset` set to False)
        """
        # patches
        dataset_mnist_patch.return_value = self.fake_dataset
        get_torch_dataset_shape_patch.return_value = self.fake_dataset_shape
        # defining arguments
        database_name = 'MNIST'
        database_path = '/path/to/MNIST/dataset'

        # action
        res_dataset = self.dataset_manager.load_default_database(database_name,
                                                              database_path,
                                                              as_dataset=False)
        # checks
        # we are using `mock.ANY` because we cannot get the object
        # 'torchivison.transform.ToTensor' used
        dataset_mnist_patch.assert_called_once_with(root=database_path,
                                                    download=True,
                                                    transform=mock.ANY)

        get_torch_dataset_shape_patch.assert_called_once_with(self.fake_dataset)
        # check that correct dataset shape has been computed
        self.assertEqual(res_dataset, self.fake_dataset_shape)


    def test_dataset_manager_11_load_default_database_exception(self):
        """
        Tests if exception `NotImplemntedError` is triggered
        when passing an unknown dataset
        """
        with self.assertRaises(NotImplementedError):
            # action: we are here passing an unknown dataset
            # we are checking method is raising NotImplementedError
            self.dataset_manager.load_default_database('my_default_dataset',
                                                    '/path/to/my/default/dataset')


    @patch('torchvision.datasets.ImageFolder')
    def test_dataset_manager_12_load_images_dataset_as_dataset_true(self, imgfolder_patch):
        """
        Tests case where one is loading image dataset with argument
        `as_dataset` is set to True
        """

        # defining patcher
        imgfolder_patch.return_value = self.fake_dataset

        # arguments
        database_path = '/path/to/MNIST/dataset'

        # action
        dataset = self.dataset_manager.load_images_dataset(database_path, as_dataset=True)

        # checks
        self.assertEqual(dataset, self.fake_dataset)
        imgfolder_patch.assert_called_once_with(database_path,
                                                transform=mock.ANY)


    def test_dataset_manager_13_load_images_dataset_as_dataset_false(self):
        """
        Tests case where one is loading image dataset with argument
        `as_dataset` is set to False
        """
        # arguments
        dataset_path = os.path.join(self.testdir,
                                    "images")

        # action
        res_dataset_shape = self.dataset_manager.load_images_dataset(dataset_path,
                                                                  as_dataset=False)

        # checks
        self.assertListEqual(res_dataset_shape, [5, 3, 30, 60])


    @patch('fedbiomed.node.dataset_manager.DatasetManager.read_csv')
    def test_dataset_manager_14_load_csv_dataset(self, read_csv_patch):
        """
        Tests `load_csv_method` (normal case scenario)
        """

        # patchers
        dummy_data = pd.DataFrame(self.dummy_data)
        read_csv_patch.return_value = dummy_data

        # arguments
        database_path = '/path/to/csv/dataset'

        # action
        data = self.dataset_manager.load_csv_dataset(database_path)

        # checks
        read_csv_patch.assert_called_once_with(database_path)
        # if below assertion `np.testing.assert_array_equal` is True,
        # it will return None, otherwise, if False, triggers an error
        self.assertIsNone(np.testing.assert_array_equal(data.values,
                                                        dummy_data.values))


    @patch('tinydb.table.Table.insert')
    @patch('fedbiomed.node.dataset_manager.DatasetManager.load_default_database')
    @patch('os.path.isdir')
    def test_dataset_manager_15_add_database_default_dataset(self,
                                                          os_listdir_patch,
                                                          datasetmanager_load_default_dataset_patch,
                                                          insert_table_patch):
        """
        Tests `add_database` method,  where one is submitting a default dataset
        """
        # unit test parameters
        fake_dataset_shape = copy.deepcopy(self.fake_dataset_shape)
        fake_dataset_shape = list(fake_dataset_shape)
        fake_dataset_path = '/path/to/some/dataset'
        fake_dataset_id = 'dataset_id_12234'
        fake_dataset_name = 'test'

        # patchers
        os_listdir_patch.return_value = True
        datasetmanager_load_default_dataset_patch.return_value = fake_dataset_shape
        insert_table_patch.return_value = None

        # action
        dataset_id = self.dataset_manager.add_database(name=fake_dataset_name,
                                                    data_type='default',
                                                    tags=['unit', 'test'],
                                                    description='some description',
                                                    path= fake_dataset_path,
                                                    dataset_id=fake_dataset_id
                                                    )
        # checks
        self.assertEqual(dataset_id, fake_dataset_id)
        datasetmanager_load_default_dataset_patch.assert_called_once_with(fake_dataset_name,
                                                                       fake_dataset_path)


    def test_dataset_manager_16_add_database_real_csv_examples_based(self):
        """
        Test add_database method for loading real csv datasets
        """

        fake_dataset_id = 'dataset_id_1232345'
        # Load data with header example
        dataset_id = self.dataset_manager.add_database(name='test',
                                                    tags=['titi'],
                                                    data_type='csv',
                                                    description='description',
                                                    path=os.path.join(self.testdir,
                                                                      "csv",
                                                                      "tata-header.csv"
                                                                      ),
                                                    dataset_id=fake_dataset_id
                                                    )

        self.assertEqual(dataset_id, fake_dataset_id)
        # Should raise error due to same tag
        with self.assertRaises(Exception):
            self.dataset_manager.add_database(name='test',
                                           tags=['titi'],
                                           data_type='csv',
                                           description='description',
                                           path=os.path.join(self.testdir,
                                                             "csv",
                                                             "tata-header.csv"
                                                             )
                                           )

        # Load data with normal different types
        self.dataset_manager.add_database(name='test',
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
            self.dataset_manager.add_database(name='test',
                                           tags=['tutu'],
                                           data_type='csv',
                                           description='description',
                                           path=os.path.join(self.testdir,
                                                             "csv",
                                                             "toto-error.csv"
                                                             )
                                           )

        # should raise n error because it is not a csv file (but data_format has been
        # passed as a 'csv' file)
        with self.assertRaises(AssertionError):
            dataset_id = self.dataset_manager.add_database(name='test',
                                                        tags=['titi-other'],
                                                        data_type='csv',
                                                        description='description',
                                                        path=os.path.join(self.testdir,
                                                                          "images"
                                                                          )
                                                        )


    def test_dataset_manager_17_add_database_wrong_datatype(self):
        """
        Tests if NotImplementedError is raised when specifying
        an unknown data type in add_database method
        """
        with self.assertRaises(NotImplementedError):
            self.dataset_manager.add_database(name='test',
                                           data_type='unknwon format',
                                           tags=['test'],
                                           description='this is a test',
                                           path='/a/path/to/some/data')


    @patch('uuid.uuid4')
    def test_dataset_manager_18_add_database_real_images_example_based(self, uuid_patch):

        """
        Test dataset_manager method for loading real images datasets
        """
        # patchers:
        uuid_patch.return_value = FakeUuid()

        # Load data with header example
        dataset_id = self.dataset_manager.add_database(name='test',
                                                    tags=['titi-tags'],
                                                    data_type='images',
                                                    description='description',
                                                    path=os.path.join(self.testdir,
                                                                      "images"
                                                                      )
                                                    )
        # check if dataset_id is correct when none is passed as argument
        self.assertEqual('dataset_' + str(FakeUuid.VALUE), dataset_id)

        # Should raise error due to same tag
        with self.assertRaises(Exception):
            self.dataset_manager.add_database(name='test',
                                           tags=['titi-tags'],
                                           data_type='images',
                                           description='description',
                                           path=os.path.join(self.testdir,
                                                             "images"
                                                             )
                                           )

        # should raise error because a csv dataset is loaded as image

        with self.assertRaises(AssertionError):
            self.dataset_manager.add_database(name='test',
                                           tags=['titi-unique'],
                                           data_type='images',
                                           description='description',
                                           path=os.path.join(self.testdir,
                                                             "csv",
                                                             "tata-header.csv"
                                                             ),
                                           )


    @patch('tinydb.table.Table.remove')
    @patch('fedbiomed.node.dataset_manager.DatasetManager.search_by_tags')
    def test_dataset_manager_19_remove_database(self,
                                             search_by_tags_patch,
                                             db_remove_patch):
        """
        Tests `remove_database` method by simulating the removal of a database
        through its tags
        """
        # arguments
        doc1 = MagicMock(doc_id=1)  # adding the attribute `doc_id` to doc
        # (usually, datasets are added with an integer from 1 to the number of datasets
        # contained in the database)

        dataset_tags = ['some', 'tags']
        search_result = [doc1]


        def db_remove_side_effect(doc_ids: List[int]):
            """
            Removes from `fake_database` global variable (that
            mimics a real database) entries listed in doc_ids
            Args:
            doc_ids (List[int]): list of doc ids
            """
            for doc_id in doc_ids:
                self.fake_database.pop(str(doc_id))

        # patchers
        search_by_tags_patch.return_value = search_result
        db_remove_patch.side_effect = db_remove_side_effect

        # action
        self.dataset_manager.remove_database(dataset_tags)

        # checks
        search_by_tags_patch.assert_called_once_with(dataset_tags)
        db_remove_patch.assert_called_once_with(doc_ids=[1])
        self.assertFalse(self.fake_database.get('1', False))


    @patch('fedbiomed.node.dataset_manager.DatasetManager.search_conflicting_tags')
    @patch('tinydb.table.Table.update')
    def test_dataset_manager_20_modify_database_info(self,
                                                  tinydb_update_patch,
                                                  conflicting_tags_patch):
        """
        Tests modify_database_info (normal case scenario),
        where one replaces an existing dataset by another one
        """
        def tinydb_update_side_effect(new_dataset: dict, existing_dataset):
            """
            side effect function that mimics the update of the database
            `fake_database`

            Args:
                new_dataset (dict): the new dataset to update
                existing_dataset unused, but is a QueryInstance/QueryImpl
            """
            fake_database['2'] = new_dataset

        def conflicting_tags_side_effect(tags):
            if all(t in fake_database.get('2')['tags'] for t in tags):
                return [{'dataset_id': dataset_id}]
            else:
                return []

        tinydb_update_patch.side_effect = tinydb_update_side_effect
        conflicting_tags_patch.side_effect = conflicting_tags_side_effect

        new_dataset_list = [
            {
                "name": "anther_test",
                "data_type": "csv",
                "tags": [ "other_tags"],
                "description": "another_description",
                "shape": [2000, 2],
                "path": "/path/to/my/other/data",
                "dataset_id": "dataset_9876",
                "dtypes": ["int64", "float64"]
            },
            {
                "tags": self.fake_database['2']['tags']
            },
            {
                "tags": self.fake_database['1']['tags'][:-1] + ['yet another tag']
            },
        ]

        for new_dataset in new_dataset_list:
            # defining a fake database
            fake_database = copy.deepcopy(self.fake_database)
            dataset_id = fake_database.get('2')['dataset_id']

            # action
            self.dataset_manager.modify_database_info(dataset_id, new_dataset)

            # checks
            # check that correct calls are made
            tinydb_update_patch.assert_called_once_with(new_dataset,
                                                        Query().dataset_id == dataset_id)
            # check database status after updating
            # first entry in database should be left unchanged ...
            self.assertEqual(fake_database.get('1'), self.fake_database.get('1'))
            # ... whereas second entry should be updated with variable `new_dataset`
            self.assertNotEqual(fake_database.get('2'), self.fake_database.get('2'))
            self.assertEqual(fake_database.get('2'), new_dataset)

            tinydb_update_patch.reset_mock()


    @patch('fedbiomed.node.dataset_manager.DatasetManager.search_conflicting_tags')
    def test_dataset_manager_20_modify_database_info_errpr(self,
                                                     conflicting_tags_patch):
        """
        Tests modify_database_info (error case scenario),
        where one replaces an existing dataset by another one
        """
        def conflicting_tags_side_effect(tags):
            return [{'dataset_id': 'one dataset is conflicting', 'name': 'fake dataset with conflicting tags'}]

        conflicting_tags_patch.side_effect = conflicting_tags_side_effect

        new_dataset_list = [
            {
                "name": "anther_test",
                "data_type": "csv",
                "tags": [ "other_tags"],
                "description": "another_description",
                "shape": [2000, 2],
                "path": "/path/to/my/other/data",
                "dataset_id": "dataset_9876",
                "dtypes": ["int64", "float64"]
            },
            {
                "tags": self.fake_database['2']['tags']
            },
            {
                "tags": self.fake_database['1']['tags'][:-1] + ['yet another tag']
            },
        ]

        for new_dataset in new_dataset_list:
            # defining a fake database
            fake_database = copy.deepcopy(self.fake_database)
            dataset_id = fake_database.get('2')['dataset_id']

            # action + check
            with self.assertRaises(FedbiomedDatasetManagerError):
                self.dataset_manager.modify_database_info(dataset_id, new_dataset)


    @patch('tinydb.table.Table.all')
    def test_dataset_manager_22_list_my_data(self,
                                             query_all_patch):
        """
        Checks `list_my_data` method in the normal case scenario
        """
        # arguments
        # `table_all_query` mimics the result of a
        # query `Table.all`
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
                            "shape": [1000, 2],
                            "path": "/path/to/my/data",
                            "dataset_id": "dataset_4567",
                            "dtypes": ["float64", "int64"]}
                           ]

        # patchers
        query_all_patch.return_value = table_all_query

        # action
        all_data = self.dataset_manager.list_my_data(True)

        # checks
        query_all_patch.assert_called_once()
        # check that none of the database contained on the node
        # doesnot contain `dtype`entry
        self.assertNotIn('dtypes', all_data[0].keys())
        self.assertNotIn('dtypes', all_data[1].keys())


    @patch('fedbiomed.node.dataset_manager.DatasetManager.load_default_database')
    def test_dataset_manager_23_load_as_dataloader_default(self,
                                                        load_default_database_patch):
        """
        Tests `load_as_dataloader` method where  the input
        dataset is the default dataset
        """
        # arguments

        # first entry in fake dataset is a MNIST default dataset
        fake_default_dataset = self.fake_database['1']

        # patchers
        load_default_database_patch.return_value = fake_default_dataset

        # action
        res = self.dataset_manager.load_as_dataloader(fake_default_dataset)

        # checks
        load_default_database_patch.assert_called_once_with(
            name=fake_default_dataset.get('name'),
            path=fake_default_dataset.get('path'),
            as_dataset=True
        )
        self.assertEqual(res['name'], fake_default_dataset['name'])
        self.assertEqual(res['dtypes'], fake_default_dataset['dtypes'])
        self.assertEqual(res['tags'], fake_default_dataset['tags'])
        self.assertEqual(res['path'], fake_default_dataset['path'])


    @patch('fedbiomed.node.dataset_manager.DatasetManager.load_images_dataset')
    def test_dataset_manager_24_load_as_dataloader_images(self, load_images_dataset_patch):
        """
        Tests `load_as_dataloader` method where  the input
        dataset is a images dataset
        """
        # arguments

        fake_dataset  = {"name": "test",
                         "data_type": "images",
                         "tags": ["some", "tags"],
                         "description": "test",
                         "shape": [1000, 2],
                         "path": "/path/to/my/data",
                         "dataset_id": "dataset_4567"}

        # patchers
        load_images_dataset_patch.return_value = self.fake_dataset

        # action
        res = self.dataset_manager.load_as_dataloader(fake_dataset)

        # checks
        load_images_dataset_patch.assert_called_once_with(folder_path=fake_dataset['path'],
                                                          as_dataset=True)
        self.assertEqual(len(res), self.fake_dataset_shape[0])
        self.assertIsInstance(res, Dataset)


    @patch('fedbiomed.node.dataset_manager.DatasetManager.read_csv')
    @patch('os.path.isfile')
    @patch('fedbiomed.node.dataset_manager.DatasetManager.search_by_tags')
    def test_dataset_manager_25_load_data_file(self,
                                            search_by_tags_patch,
                                            os_isfile_patch,
                                            read_csv_patch
                                            ):
        """
        Tests `load_data` method where a file is loaded (either a pandas
        dataframe, numpy array, or torch tensor)
        """
        # arguments
        tags = ['some', 'tags']
        search_by_tags_query = {'dataset_id': 'datset_id_1234',
                                'path': 'path/to/my/dataset',
                                'data_type': 'csv'}
        # arguments: generating random data for third test
        pandas_dataset_test_3 = pd.DataFrame(np.random.rand(100, 10))

        # patchers
        search_by_tags_patch.return_value = [search_by_tags_query]
        os_isfile_patch.return_value = True
        pandas_dataset = pd.DataFrame(self.dummy_data)
        read_csv_patch.return_value = pandas_dataset

        # action: first test with mode = 'pandas'
        pandas_df = self.dataset_manager.load_data(tags, mode = 'pandas')

        # first test checks
        search_by_tags_patch.assert_called_once_with(tags)
        # nota: `np.testing.assert_array_equal` returns None when
        # test passed
        self.assertIsNone(
            np.testing.assert_array_equal(pandas_df.values, pandas_dataset.values)
        )

        # resetting Mocks for second test
        search_by_tags_patch.reset_mock()
        search_by_tags_patch.return_value = [search_by_tags_query]
        read_csv_patch.return_value = pandas_dataset.drop(columns='chars')
        # action: second test with mode = 'numpy'
        np_array = self.dataset_manager.load_data(tags, mode = 'numpy')

        # second test checks
        search_by_tags_patch.assert_called_once_with(tags)
        self.assertIsNone(
            np.testing.assert_array_equal(np_array,
                                          pandas_dataset.drop(columns='chars').values)
        )

        # resetting Mocks for third test
        search_by_tags_patch.reset_mock()
        search_by_tags_patch.return_value = [search_by_tags_query]
        read_csv_patch.return_value = pd.DataFrame(pandas_dataset_test_3)
        # action: third test with mode = 'torch_tensor'
        torch_tensor = self.dataset_manager.load_data(tags, mode = 'torch_tensor')

        # third test checks
        search_by_tags_patch.assert_called_once_with(tags)
        self.assertIsNone(
            np.testing.assert_array_equal(torch_tensor,
                                          pandas_dataset_test_3.values)
        )


    @patch('fedbiomed.node.dataset_manager.DatasetManager.load_as_dataloader')
    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('fedbiomed.node.dataset_manager.DatasetManager.search_by_tags')
    def test_dataset_manager_26_load_data_folder(self,
                                              search_by_tags_patch,
                                              os_isfile_patch,
                                              os_isdir_patch,
                                              load_as_dataloader_patch):
        """
        Tests `load_data` method, in the cae where a folder is loaded
        """
        # arguments
        tags = ['some', 'tags']
        search_by_tags_query = {'dataset_id': 'datset_id_1234',
                                'path': 'path/to/my/folder',
                                'data_type': 'images'}
        # patches
        search_by_tags_patch.return_value = [search_by_tags_query]
        # mimicking behaviour where a folder has been found through search query
        os_isfile_patch.return_value = False
        os_isdir_patch.return_value = True

        load_as_dataloader_patch.return_value = self.fake_dataset

        # action: Test 1, loading with argument 'mode' = 'torch_dataset'
        torch_dataset = self.dataset_manager.load_data(tags, mode= 'torcH_dataset')

        # checks
        search_by_tags_patch.assert_called_once_with(tags)
        load_as_dataloader_patch.assert_called_once_with(search_by_tags_query)
        self.assertIsInstance(torch_dataset, Dataset)
        self.assertEqual(len(torch_dataset), self.fake_dataset_shape[0])

        # In this last part of the test, we are going to check if exceptions
        # (for the other modes) are raised
        with self.assertRaises(NotImplementedError):
            self.dataset_manager.load_data(tags, mode='torch_tensor')
        with self.assertRaises(NotImplementedError):
            self.dataset_manager.load_data(tags, mode='numpy')
        with self.assertRaises(NotImplementedError):
            self.dataset_manager.load_data(tags, mode='pandas')

    def test_dataset_manager_27_load_data_exception(self):
        """
        Tests if an exception is raised when passing an
        unknown mode to `load_data` method
        """
        # arguments
        tags = ['some', 'tags']

        # action and check
        with self.assertRaises(NotImplementedError):
            self.dataset_manager.load_data(tags, mode='unknown_mode')

    def test_dataset_manager_28_load_existing_mednist_dataset(self):
        """
        Tests case where one is loading mednist dataset without downloading it
        """

        def _create_fake_mednist_dataset(self):
            '''
            Create fake mednist dataset and save the images in tempdir

            Dataset folders structure:

            |- class_0 
            |   |- image_0.jpeg
            |- class_1 
            |   |- image_0.jpeg
            ...
            |_ class_n
                |- image_0.jpeg

            '''

            mednist_path = os.path.join(self.tempdir, 'MedNIST')
            os.makedirs(mednist_path)

            fake_img_data = np.random.randint(0,255,(64, 64))
            img = Image.fromarray(fake_img_data, 'L')
            n_classes = 6

            # one image per class
            for class_i in range(n_classes):
                class_path = os.path.join(mednist_path, f'class_{class_i}')
                os.makedirs(class_path)
                img_path = os.path.join(class_path, 'image_0.jpeg')
                img.save(img_path)

            return datasets.ImageFolder(mednist_path, transform=transforms.ToTensor())


        fake_dataset = _create_fake_mednist_dataset(self)

        # action
        # Test the load mednist method with input as_dataset False
        res_dataset_shape = self.dataset_manager.load_mednist_database(self.tempdir,
                                                                       as_dataset=False)

        # checks
        self.assertListEqual(res_dataset_shape, [6, 3, 64, 64])


        # Test the load mednist method with input as_dataset True
        res_dataset = self.dataset_manager.load_mednist_database(self.tempdir,
                                                                 as_dataset=True)

        for i in range(len(fake_dataset)):
            with self.subTest(i=i):
                self.assertTrue(torch.equal(res_dataset[i][0], fake_dataset[i][0]))
                # check that assigned classes are correct
                self.assertEqual(res_dataset[i][1], fake_dataset[i][1])


    @patch('fedbiomed.node.dataset_manager.urlretrieve')
    def test_dataset_manager_29_download_extrat_mednist(self,
                                                        urlretrieve_patch):
        """
        Tests the correct process of data download and extraction
        in order to make MedNIST dataset (retrieved from url limnk)
        """

        ARCHIVE_PATH = os.path.join(self.testdir, "images/MedNIST_test.tar.gz")

        def urlretieve_side_effect(url, path):
            """Mimics download of dataset by coping & pasting
            dataset archive in the good directory
            """
            # copying & loading archive into temporary file
            path = pathlib.Path(path).parent.absolute()
            shutil.copy2(ARCHIVE_PATH, path)
            os.rename(os.path.join(path, "MedNIST_test.tar.gz"),
                      os.path.join(path, "MedNIST.tar.gz"))
        # configuring patchers
        urlretrieve_patch.side_effect = urlretieve_side_effect

        with patch.object(os, 'remove') as os_remove_patch:
            os_remove_patch.return_value = None
            # action
            res_dataset = self.dataset_manager.load_mednist_database(self.tempdir,
                                                                     as_dataset=True)
        # Tests
        urlretrieve_patch.assert_called_once()
        self.assertListEqual(res_dataset.classes,
                             ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT'])
        label = 0
        for i in range(12):
            with self.subTest(i=i):
                # check each image has the correct extension 'jpeg'
                self.assertEqual(pathlib.Path(res_dataset.imgs[i][0]).suffix, '.jpeg')
                # NB: this test assumes that each images are sorted from label 0 to label 6
                # it may be wrong in future pytorch releases
                self.assertEqual(res_dataset.imgs[i][1], label)
                if i % 2 != 0:
                    label += 1

    def test_dataset_manager_30_load_mednist_database_exception(self):
        """
        Tests if exception `FedbiomedDatasetManagerError` is triggered
        when mednist dataset folder is empty
        """
        
        # case where Mednist folder is already existing
        mednist_path = os.path.join(self.tempdir, 'MedNIST')
        os.makedirs(mednist_path)

        with self.assertRaises(FedbiomedDatasetManagerError):
            # action: we are here passing an empty directory
            # and checking if method raises FedbiomedDatasetManagerError
            self.dataset_manager.load_mednist_database(self.tempdir)

    def test_dataset_manager_31_obfuscate_private_information(self):
        """Tests if error is raised if dataset is not parsable when calling `obfuscate_privte_information"""
        metadata_with_private_info  = [{
            'path': 'private/info',
            'nonprivate': 'info',
            'data_type': 'medical-folder',
            'dataset_parameters': {
                'tabular_file': 'private/info',
                'nonprivate': 'info'
                                   }
        }]
        private_metadata = DatasetManager.obfuscate_private_information(metadata_with_private_info)
        expected_private_metadata = [{
            'nonprivate': 'info',
            'data_type': 'medical-folder',
            'dataset_parameters': {
                'nonprivate': 'info'
            }
        }]
        self.assertEqual(private_metadata, expected_private_metadata)
        with self.assertRaises(FedbiomedDatasetManagerError):
            _ = DatasetManager.obfuscate_private_information([*metadata_with_private_info, 'non-dict-like'])

    @patch('os.path.isdir')
    def test_dataset_manager_32_data_loading_plan_save(self, patch_isdir):
        """Tests that DatasetManager correctly saves a DataLoadingPlan"""
        patch_isdir.return_value = True
        from test_data_loading_plan import LoadingBlockForTesting

        dlb1 = LoadingBlockForTesting()
        dlb2 = LoadingBlockForTesting()
        dlb2.data = {'some': 'other data'}

        dlp = DataLoadingPlan()
        dlp.desc = '1234'
        dlp[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING] = dlb1
        dlp[LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING] = dlb2

        dataset_manager = DatasetManager()
        dataset_manager.load_default_database = MagicMock(return_value=(1, 1))
        dataset_manager.add_database(
            name='dlp-test-db',
            data_type='default',
            tags=['test'],
            description='',
            dataset_id='test-id-dlp-1234',
            path='some/test/path',
            data_loading_plan=dlp
        )

        self.assertIn('Data_Loading_Plans', dataset_manager._db.tables())
        dataset = dataset_manager.get_by_id('test-id-dlp-1234')
        self.assertEqual(dataset['dlp_id'], dlp.dlp_id)

        dlp_metadata, loading_blocks_metadata = dataset_manager.get_dlp_by_id(dataset['dlp_id'])
        self.assertEqual(dlp_metadata['dlp_id'], dlp.dlp_id)
        new_dlp = DataLoadingPlan().deserialize(dlp_metadata, loading_blocks_metadata)
        self.assertIn(LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING, new_dlp)
        self.assertIn(LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING, new_dlp)
        self.assertDictEqual(new_dlp[LoadingBlockTypesForTesting.LOADING_BLOCK_FOR_TESTING].data, dlb1.data)
        self.assertDictEqual(new_dlp[LoadingBlockTypesForTesting.OTHER_LOADING_BLOCK_FOR_TESTING].data, dlb2.data)

        dataset_manager._db.close()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

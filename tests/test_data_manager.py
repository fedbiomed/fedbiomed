import unittest

import numpy as np
from unittest.mock import patch, MagicMock
import testsupport.mock_node_environ  # noqa (remove flake8 false warning)

from torch.utils.data import Dataset
from fedbiomed.common.data import DataManager
from fedbiomed.common.exceptions import FedbiomedDataManagerError


class TestModelManager(unittest.TestCase):

    class CustomDataset(Dataset):
        """ Create PyTorch Dataset for test purposes """
        def __init__(self):
            self.X_train = []
            self.Y_train = []

        def __len__(self):
            return len(self.Y_train)

        def __getitem__(self, idx):
            return self.X_train[idx], self.Y_train[idx]

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_data_manager_01_initialization(self):
        """ Testing different initializations of DataManger """

        # Test passing invalid argument
        with self.assertRaises(FedbiomedDataManagerError):
            DataManager(dataset='invalid-argument')

        # Test passing another invalid argument
        with self.assertRaises(FedbiomedDataManagerError):
            DataManager(dataset=12)

        # Test passing dataset as list
        with self.assertRaises(FedbiomedDataManagerError):
            DataManager(dataset=[12, 12, 12, 12])

        # Test passing dataset as Numpy Array
        dataset = np.array([[1, 2], [1, 2], [1, 2]])
        with self.assertRaises(FedbiomedDataManagerError):
            # Should raise an Error because when dataset in an array
            # target variable should be provided
            DataManager(dataset=dataset)

        # Test if dataset is Dataset and target is not none
        with self.assertRaises(FedbiomedDataManagerError):
            DataManager(dataset=TestModelManager.CustomDataset(), target=np.array([1, 2, 3]))

        # Test proper way of providing dataset for scikit-learn
        dataset = np.array([[1, 2], [1, 2], [1, 2]])
        target = np.array([1, 2, 3])
        try:
            DataManager(dataset=dataset, target=target)
        except:
            self.assertTrue(False, 'Can not build DataManager object')

        # Test init with PyTorch Dataset
        try:
            DataManager(dataset=TestModelManager.CustomDataset())
        except:
            self.assertTrue(False, 'Can not build DataManager object')

    @patch('fedbiomed.common.data._data_manager.TorchDataset')
    def test_data_manager_02___getattr__(self,
                                         mock_torch_dataset):

        """ Testing __getattr__ method of DataManager """

        torch_mock = MagicMock()
        torch_mock.split = MagicMock(return_value='test')
        mock_torch_dataset.return_value = torch_mock

        # Test implemented attribute split
        data_manager = DataManager(TestModelManager.CustomDataset())
        result = data_manager.split()
        self.assertEqual(result, 'test')

        # Should raise not implemented error
        with self.assertRaises(FedbiomedDataManagerError):
            result = data_manager.spliter()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

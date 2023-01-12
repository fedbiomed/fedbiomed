import unittest
import fedbiomed.common.data._torch_data_manager  # noqa
import numpy as np

from unittest.mock import patch
from torch.utils.data import Dataset, Subset
from fedbiomed.common.data import TorchDataManager
from fedbiomed.common.exceptions import FedbiomedTorchDataManagerError


class TestTorchDataManager(unittest.TestCase):
    class CustomDataset(Dataset):
        """ Create PyTorch Dataset for test purposes """

        def __init__(self):
            self.X_train = np.array([[1, 2, 3],
                                     [1, 2, 3],
                                     [1, 2, 3],
                                     [1, 2, 3],
                                     [1, 2, 3],
                                     [1, 2, 3]])
            self.Y_train = np.array([1, 2, 3, 4, 5, 6])

        def __len__(self):
            return len(self.Y_train)

        def __getitem__(self, idx):
            return self.X_train[idx], self.Y_train[idx]

    class CustomDatasetInvalid(Dataset):
        """ Create Invalid PyTorch Dataset for test purposes """

        def __init__(self):
            self.X_train = [[1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3]]
            self.Y_train = [1, 2, 3]

        def __getitem__(self, idx):
            return self.X_train[idx], self.Y_train[idx]

    class CustomDatasetAttrError(Dataset):
        """ Create PyTorch Dataset that raises Attr Error for test purposes """

        def __init__(self):
            self.data = None
            pass

        def __len__(self):
            raise AttributeError

        def __getitem__(self, idx):
            return self.data

    class CustomDatasetTypeError(Dataset):
        """ Create PyTorch Dataset that raises TypeError for test purposes """

        def __init__(self):
            self.data = None
            pass

        def __len__(self):
            raise TypeError

        def __getitem__(self, idx):
            return self.data

    def setUp(self):
        # Setup global TorchDataManager class
        self.dataset = TestTorchDataManager.CustomDataset()
        self.torch_data_manager = TorchDataManager(dataset=self.dataset,
                                                   batch_size=48,
                                                   shuffle=True)

    def tearDown(self):
        pass

    def test_torch_data_manager_01_init_failure(self):
        """ Testing build failure of Torch Data Manager """

        with self.assertRaises(FedbiomedTorchDataManagerError):
            TorchDataManager(dataset='wrong_type',
                             batch_size=48,
                             shuffle=True)

    def test_torch_data_manager_01_dataset(self):
        """ Testing dataset getter method """

        result = self.torch_data_manager.dataset
        self.assertEqual(result, self.dataset, 'dataset() returns un expected torch Dataset object')

    def test_torch_data_manager_02_split(self):
        """Testing split method of TorchDataManager class """

        # Test invalid ratio argument
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_data_manager.split(test_ratio=12)

        # Test invalid ratio argument
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_data_manager.split(test_ratio='12')

        # Test invalid ratio argument
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_data_manager.split(test_ratio=-12)

        # Test proper split
        try:
            self.torch_data_manager.split(0.3)
        except:
            self.assertTrue(False, 'Error while splitting TorchDataManager')

        # Test exceptions
        invalid = TestTorchDataManager.CustomDatasetInvalid()
        self.torch_data_manager._dataset = invalid
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_data_manager.split(0.3)

        invalid = TestTorchDataManager.CustomDatasetTypeError()
        self.torch_data_manager._dataset = invalid
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_data_manager.split(0.3)

        invalid = TestTorchDataManager.CustomDatasetAttrError()
        self.torch_data_manager._dataset = invalid
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_data_manager.split(0.3)

    def test_torch_data_manager_05_split_results(self):
        """ Test splitting result """

        # Test with split
        loader_train, loader_test = self.torch_data_manager.split(0.5)
        self.assertEqual(len(loader_train.dataset), len(self.dataset) / 2, 'Did not properly get loader '
                                                                           'of train partition')

        # If test partition is zero
        loader_train, loader_test = self.torch_data_manager.split(0)
        self.assertIsNone(loader_test, 'Loader is not None where it should be')

        # If test partition is 1
        loader_train, loader_test = self.torch_data_manager.split(1)
        self.assertIsNone(loader_train, 'Loader is not None where it should be')

    def test_torch_data_manager_05_subset_train(self):
        """ Testing the method load train partition """

        # Test with split
        self.torch_data_manager.split(0.5)
        subset = self.torch_data_manager.subset_train()
        self.assertIsInstance(subset, Subset, 'Can not get proper subset object')

    def test_torch_data_manager_05_subset_test(self):
        """ Testing the method load train partition """

        # Test with split
        self.torch_data_manager.split(0.5)
        subset = self.torch_data_manager.subset_test()
        self.assertIsInstance(subset, Subset, 'Can not get proper subset object')

    def test_torch_data_manager_05_load_all_samples(self):
        """ Testing the method load train partition """

        # Test with split
        self.torch_data_manager.split(0.5)
        loader = self.torch_data_manager.load_all_samples()
        self.assertEqual(len(loader.dataset), len(self.dataset), 'Did not properly get loader for all samples')

    @patch('fedbiomed.common.data._torch_data_manager.DataLoader')
    def test_torch_data_manager_06_create_torch_data_loader(self, data_loader):
        """ Test function create torch data loader """

        self.torch_data_manager.split(0.5)
        s = self.torch_data_manager.subset_test()

        data_loader.side_effect = TypeError()
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_data_manager._create_torch_data_loader(s)

        data_loader.side_effect = AttributeError()
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_data_manager._subset_loader(s)

        data_loader.side_effect = None
        data_loader.return_value = 'Data'
        result = self.torch_data_manager._subset_loader(s)
        self.assertEqual(result, 'Data')

    def test_torch_data_manager_07_to_sklearn(self):
        """Test converting TorchDataManage to SkLearnDataManager"""

        result = self.torch_data_manager.to_sklearn()
        self.assertIsInstance(result, fedbiomed.common.data._sklearn_data_manager.SkLearnDataManager)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

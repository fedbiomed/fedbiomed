import unittest
import testsupport.mock_node_environ  # noqa (remove flake8 false warning)

from torch.utils.data import Dataset, Subset
from fedbiomed.common.data import TorchDataManager
from fedbiomed.common.exceptions import FedbiomedTorchDataManagerError


class TestTorchDataManager(unittest.TestCase):

    class CustomDataset(Dataset):
        """ Create PyTorch Dataset for test purposes """

        def __init__(self):
            self.X_train = [[1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3],
                            [1, 2, 3]]
            self.Y_train = [1, 2, 3, 4, 5, 6]

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
            raise AttributeError

        def __getitem__(self, idx):
            return self.data

    def setUp(self):
        # Setup global TorchDataManager class
        self.dataset = TestTorchDataManager.CustomDataset()
        self.torch_dataset = TorchDataManager(dataset=self.dataset,
                                          batch_size=48,
                                          shuffle=True)

    def tearDown(self):
        pass

    def test_torch_dataset_01_dataset(self):
        """ Testing dataset getter method """

        result = self.torch_dataset.dataset()
        self.assertEqual(result, self.dataset, 'dataset() returns un expected torch Dataset object')

    def test_torch_dataset_02_split(self):
        """Testing split method of TorchDataManager class """

        # Test invalid ratio argument
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_dataset.split(ratio=12)

        # Test invalid ratio argument
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_dataset.split(ratio='12')

        # Test invalid ratio argument
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_dataset.split(ratio=-12)

        # Test proper split
        try:
            self.torch_dataset.split(0.3)
        except:
            self.assertTrue(False, 'Error while splitting TorchDataManager')

        # Test exceptions
        invalid = TestTorchDataManager.CustomDatasetInvalid()
        self.torch_dataset._dataset = invalid
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_dataset.split(0.3)

        invalid = TestTorchDataManager.CustomDatasetTypeError()
        self.torch_dataset._dataset = invalid
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_dataset.split(0.3)

        invalid = TestTorchDataManager.CustomDatasetAttrError()
        self.torch_dataset._dataset = invalid
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_dataset.split(0.3)

    def test_torch_dataset_03_load_train_partition(self):
        """ Testing the method load train partition """

        # Test raising error in case of requesting train partition
        # before splitting
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_dataset.load_train_partition()

        # Test with split
        self.torch_dataset.split(0.5)
        loader = self.torch_dataset.load_train_partition()
        self.assertEqual(len(loader.dataset), len(self.dataset)/2, 'Did not properly get loader of train partition')

        # If test partition is 1 and train is zero
        self.torch_dataset.split(1)
        loader = self.torch_dataset.load_train_partition()
        self.assertIsNone(loader, 'Loader is not None where it should be')

    def test_torch_dataset_04_load_test_partition(self):
        """ Testing the method load test partition """

        # Test raising error in case of requesting train partition
        # before splitting
        with self.assertRaises(FedbiomedTorchDataManagerError):
            self.torch_dataset.load_test_partition()

        # Test with split
        self.torch_dataset.split(0.5)
        loader = self.torch_dataset.load_test_partition()
        self.assertEqual(len(loader.dataset), len(self.dataset)/2, 'Did not properly get loader of train partition')

        # If test partition is zero
        self.torch_dataset.split(0)
        loader = self.torch_dataset.load_test_partition()
        self.assertIsNone(loader, 'Loader is not None where it should be')

    def test_torch_dataset_05_subset_train(self):
        """ Testing the method load train partition """

        # Test with split
        self.torch_dataset.split(0.5)
        subset = self.torch_dataset.subset_train()
        self.assertIsInstance(subset, Subset, 'Can not get proper subset object')

    def test_torch_dataset_05_subset_test(self):
        """ Testing the method load train partition """

        # Test with split
        self.torch_dataset.split(0.5)
        subset = self.torch_dataset.subset_test()
        self.assertIsInstance(subset, Subset, 'Can not get proper subset object')

    def test_torch_dataset_05_load_all_samples(self):
        """ Testing the method load train partition """

        # Test with split
        self.torch_dataset.split(0.5)
        loader = self.torch_dataset.load_all_samples()
        self.assertEqual(len(loader.dataset), len(self.dataset), 'Did not properly get loader for all samples')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

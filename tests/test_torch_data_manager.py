import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import Subset

import fedbiomed.common.datamanager._torch_data_manager  # noqa
from fedbiomed.common.datamanager import TorchDataManager
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.exceptions import FedbiomedError


class TestTorchDataManager(unittest.TestCase):
    class CustomDataset(Dataset):
        """Create PyTorch Dataset for test purposes"""

        def __init__(self):
            self.X_train = [
                torch.Tensor([1, 2, 3]),
                torch.Tensor([4, 5, 6]),
                torch.Tensor([7, 8, 9]),
                torch.Tensor([10, 11, 12]),
                torch.Tensor([13, 14, 15]),
                torch.Tensor([16, 17, 18]),
            ]
            self.Y_train = [
                torch.Tensor(1),
                torch.Tensor(2),
                torch.Tensor(3),
                torch.Tensor(4),
                torch.Tensor(5),
                torch.Tensor(6),
            ]

        def complete_initialization(self):
            pass

        def _apply_transforms(self, sample):
            pass

        def __len__(self):
            return len(self.Y_train)

        def __getitem__(self, idx):
            return self.X_train[idx], self.Y_train[idx]

    class CustomDatasetInvalid:
        """Create Invalid PyTorch Dataset for test purposes"""

        def __init__(self):
            self.X_train = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
            self.Y_train = [1, 2, 3]

        def complete_initialization(self):
            pass

        def _apply_transforms(self, sample):
            pass

        # no __len__ method

        def __getitem__(self, idx):
            return self.X_train[idx], self.Y_train[idx]

    class CustomDatasetAttrError(Dataset):
        """Create PyTorch Dataset that raises Attr Error for test purposes"""

        def __init__(self):
            self.data = None
            pass

        def complete_initialization(self):
            pass

        def _apply_transforms(self, sample):
            pass

        def __len__(self):
            raise AttributeError

        def __getitem__(self, idx):
            return self.data

    class CustomDatasetTypeError(Dataset):
        """Create PyTorch Dataset that raises TypeError for test purposes"""

        def __init__(self):
            self.data = None
            pass

        def complete_initialization(self):
            pass

        def _apply_transforms(self, sample):
            pass

        def __len__(self):
            raise TypeError

        def __getitem__(self, idx):
            return self.data

    def setUp(self):
        # Setup global TorchDataManager class
        self.dataset = TestTorchDataManager.CustomDataset()
        self.torch_data_manager = TorchDataManager(
            dataset=self.dataset, batch_size=48, shuffle=True
        )

    def tearDown(self):
        pass

    def test_torch_data_manager_01_init_failure(self):
        """Testing build failure of Torch Data Manager"""

        with self.assertRaises(FedbiomedError):
            TorchDataManager(dataset="wrong_type", batch_size=48, shuffle=True)

    def test_torch_data_manager_01_dataset(self):
        """Testing dataset getter method"""

        result = self.torch_data_manager.dataset
        self.assertEqual(
            result, self.dataset, "dataset() returns un expected torch Dataset object"
        )

    def test_torch_data_manager_02_split(self):
        """Testing split method of TorchDataManager class"""

        # Test invalid ratio argument
        with self.assertRaises(FedbiomedError):
            self.torch_data_manager.split(test_ratio=12, test_batch_size=0)

        # Test invalid ratio argument
        with self.assertRaises(FedbiomedError):
            self.torch_data_manager.split(test_ratio="12", test_batch_size=2)

        # Test invalid ratio argument
        with self.assertRaises(FedbiomedError):
            self.torch_data_manager.split(test_ratio=-12, test_batch_size=3)

        # Test proper split
        try:
            self.torch_data_manager.split(0.3, test_batch_size=None)
        except Exception:
            self.assertTrue(False, "Error while splitting TorchDataManager")

        # Test exceptions
        invalid = TestTorchDataManager.CustomDatasetInvalid()
        self.torch_data_manager._dataset = invalid
        with self.assertRaises(FedbiomedError):
            self.torch_data_manager.split(0.3, test_batch_size=4)

        invalid = TestTorchDataManager.CustomDatasetTypeError()
        self.torch_data_manager._dataset = invalid
        with self.assertRaises(FedbiomedError):
            self.torch_data_manager.split(0.3, test_batch_size=5)

        invalid = TestTorchDataManager.CustomDatasetAttrError()
        self.torch_data_manager._dataset = invalid
        with self.assertRaises(FedbiomedError):
            self.torch_data_manager.split(0.3, test_batch_size=6)

    def test_torch_data_manager_03_split_results(self):
        """Test splitting result"""

        # Test with split
        loader_train, loader_test = self.torch_data_manager.split(
            0.5, test_batch_size=None
        )
        self.assertEqual(
            len(loader_train.dataset),
            len(self.dataset) / 2,
            "Did not properly get loader of train partition",
        )

        # If test partition is zero

        loader_train, loader_test = self.torch_data_manager.split(
            0, test_batch_size=None
        )
        self.assertIsNone(loader_test, "Loader is not None where it should be")
        self.assertListEqual(self.torch_data_manager._testing_index, [])

        # If test partition is 1
        loader_train, loader_test = self.torch_data_manager.split(
            1, test_batch_size=None
        )
        self.assertIsNone(loader_train, "Loader is not None where it should be")
        self.assertListEqual(
            sorted(self.torch_data_manager._testing_index),
            list(range(len(self.dataset))),
        )

    def test_torch_data_manager_04_subset_train(self):
        """Testing the method load train partition"""

        # Test with split
        self.torch_data_manager.split(0.5, test_batch_size=None)
        subset = self.torch_data_manager.subset_train()
        self.assertIsInstance(subset, Subset, "Can not get proper subset object")

    def test_torch_data_manager_05_subset_test(self):
        """Testing the method load train partition"""

        # Test with split
        self.torch_data_manager.split(0.5, test_batch_size=None)
        subset = self.torch_data_manager.subset_test()
        self.assertIsInstance(subset, Subset, "Can not get proper subset object")

    def test_torch_data_manager_05_load_all_samples(self):
        """Testing the method load train partition"""

        # Test with split
        self.torch_data_manager.split(0.5, test_batch_size=None)
        loader = self.torch_data_manager.load_all_samples()
        self.assertEqual(
            len(loader.dataset),
            len(self.dataset),
            "Did not properly get loader for all samples",
        )

    @patch("fedbiomed.common.datamanager._torch_data_manager.PytorchDataLoader")
    def test_torch_data_manager_06_create_torch_data_loader(self, data_loader):
        """Test function create torch data loader"""

        self.torch_data_manager.split(0.5, test_batch_size=None)
        s = self.torch_data_manager.subset_test()

        data_loader.side_effect = TypeError()
        with self.assertRaises(FedbiomedError):
            self.torch_data_manager._create_torch_data_loader(s)

        data_loader.side_effect = AttributeError()
        with self.assertRaises(FedbiomedError):
            self.torch_data_manager._subset_loader(s)

        data_loader.side_effect = None
        data_loader.return_value = "Data"
        result = self.torch_data_manager._subset_loader(s)
        self.assertEqual(result, "Data")

    def test_torch_data_manager_08_save_load_state(self):
        test_ratio = 0.5

        loader_train, loader_test = self.torch_data_manager.split(
            test_ratio, test_batch_size=None
        )

        state = self.torch_data_manager.save_state()

        new_torch_data_manager = TorchDataManager(
            self.dataset, batch_size=48, shuffle=True
        )
        new_torch_data_manager.load_state(state)

        new_train_loader, new_test_loader = new_torch_data_manager.split(
            test_ratio=test_ratio, test_batch_size=None
        )

        self.assertListEqual(
            self.torch_data_manager._testing_index,
            new_torch_data_manager._testing_index,
        )

        for i in range(2):
            self.assertTrue(
                self.is_same_sample(
                    loader_train.dataset[i],
                    new_train_loader.dataset[i],
                )
            )

        for i in range(2):
            self.assertTrue(
                self.is_same_sample(
                    loader_test.dataset[i],
                    new_test_loader.dataset[i],
                )
            )

        # Even after shuffling, same samples can be drawn, so this test seems not valid
        #
        # # changing the `test_ratio` value
        # del new_torch_data_manager
        # new_torch_data_manager = TorchDataManager(
        #     self.dataset, batch_size=48, shuffle=True
        # )
        # new_torch_data_manager.load_state(state)
        # test_ratio = 0.25
        # shuffled_train_loader, shuffled_test_loader = new_torch_data_manager.split(
        #     test_ratio=test_ratio, test_batch_size=None
        # )
        #
        # for i in range(2):
        #     self.assertFalse(
        #         self.is_same_sample(
        #             loader_train.dataset[i],
        #             shuffled_train_loader.dataset[i],
        #         )
        #     )
        # for i in range(2):
        #     self.assertFalse(
        #         self.is_same_sample(
        #             loader_test.dataset[i],
        #             shuffled_test_loader.dataset[i],
        #         )
        #     )

        # new_torch_data_manager._testing_index = [1, 2, 39999]
        # new_torch_data_manager._training_index = []
        # raise IndexError(f"{new_torch_data_manager._testing_index}")

    def test_torch_data_manager_09_shuffle_testing_dataset(self):
        self.torch_data_manager = TorchDataManager(
            dataset=self.dataset,
            batch_size=48,
            shuffle=True,
        )

        test_ratio = 0.5
        loader_train, loader_test = self.torch_data_manager.split(
            test_ratio, test_batch_size=None, is_shuffled_testing_dataset=False
        )
        loader_train2, loader_test2 = self.torch_data_manager.split(
            test_ratio, test_batch_size=None, is_shuffled_testing_dataset=False
        )

        for i in range(2):
            self.assertTrue(
                self.is_same_sample(
                    loader_train.dataset[i],
                    loader_train2.dataset[i],
                )
            )
        for i in range(2):
            self.assertTrue(
                self.is_same_sample(
                    loader_test.dataset[i],
                    loader_test2.dataset[i],
                )
            )

        with patch(
            "fedbiomed.common.datamanager._torch_data_manager.random_split",
            return_value=(MagicMock(), MagicMock()),
        ) as mock_random_split:
            loader_train2, loader_test2 = self.torch_data_manager.split(
                test_ratio, test_batch_size=None, is_shuffled_testing_dataset=True
            )
            mock_random_split.assert_called_once()

    @staticmethod
    def is_same_sample(sample1, sample2) -> bool:
        return torch.equal(sample1[0], sample2[0]) and torch.equal(
            sample1[1], sample2[1]
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

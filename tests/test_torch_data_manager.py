import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import Subset

from fedbiomed.common.dataloader._pytorch_dataloader import (
    PytorchDataLoader,
    collate_optional_target,
)
from fedbiomed.common.datamanager import TorchDataManager
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.exceptions import FedbiomedError


# TODO - DATASET-REDESIGN: Update tests after removing DatasetDataItemModality and DataType
class TestTorchDataManager(unittest.TestCase):
    class CustomDataset(Dataset):
        """Create PyTorch Dataset for test purposes"""

        def __init__(self):
            self.X_train = [
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([4.0, 5.0, 6.0]),
                torch.tensor([7.0, 8.0, 9.0]),
                torch.tensor([10.0, 11.0, 12.0]),
                torch.tensor([13.0, 14.0, 15.0]),
                torch.tensor([16.0, 17.0, 18.0]),
            ]
            self.Y_train = [
                torch.tensor(1.0),
                torch.tensor(2.0),
                torch.tensor(3.0),
                torch.tensor(4.0),
                torch.tensor(5.0),
                torch.tensor(6.0),
            ]

        def load(self):
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

        def load(self):
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

        def load(self):
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

        def load(self):
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

    def test_torch_data_manager_10_none_target(self):
        """Unsupervised dataset (None target) is collated as (data, None)."""

        class UnsupervisedDataset(Dataset):
            def __init__(self):
                self.X = [torch.Tensor([i, i + 1, i + 2]) for i in range(6)]

            def load(self):
                pass

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], None

        dm = TorchDataManager(
            dataset=UnsupervisedDataset(), batch_size=3, shuffle=False
        )
        data, target = next(iter(dm.load_all_samples()))
        self.assertIsInstance(data, torch.Tensor)
        self.assertEqual(tuple(data.shape), (3, 3))
        self.assertIsNone(target)

    def test_torch_data_manager_11_inconsistent_target(self):
        """A dataset with a target on some samples but None on others is rejected."""

        class InconsistentDataset(Dataset):
            def __init__(self):
                self.X = [torch.Tensor([i, i + 1, i + 2]) for i in range(4)]

            def load(self):
                pass

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                # First sample has no target, later ones do.
                target = None if idx == 0 else torch.Tensor([idx])
                return self.X[idx], target

        dm = TorchDataManager(
            dataset=InconsistentDataset(), batch_size=4, shuffle=False
        )
        with self.assertRaisesRegex(FedbiomedError, "Inconsistent target"):
            next(iter(dm.load_all_samples()))

    def test_torch_data_manager_12_collate_optional_target_supervised(self):
        """A batch with targets is collated with the default behaviour."""
        batch = [(torch.tensor([1.0]), torch.tensor([0])) for _ in range(3)]
        data, target = collate_optional_target(batch)
        self.assertEqual(tuple(data.shape), (3, 1))
        self.assertEqual(tuple(target.shape), (3, 1))

    def test_torch_data_manager_13_dataloader_default_collate(self):
        """PytorchDataLoader uses the optional-target collate unless overridden."""
        dataset = [(torch.tensor([float(i)]), None) for i in range(4)]
        loader = PytorchDataLoader(dataset, batch_size=2)
        self.assertIs(loader.collate_fn, collate_optional_target)
        _, target = next(iter(loader))
        self.assertIsNone(target)

    def test_torch_data_manager_14_dataloader_explicit_collate(self):
        """An explicit collate_fn is not overridden by the default."""
        dataset = [(torch.tensor([float(i)]), None) for i in range(4)]
        loader = PytorchDataLoader(dataset, batch_size=2, collate_fn=lambda b: "custom")
        self.assertIsNot(loader.collate_fn, collate_optional_target)
        self.assertEqual(next(iter(loader)), "custom")

    @patch(
        "fedbiomed.common.datamanager._torch_data_manager.TorchDataManager._loader_class"
    )
    def test_torch_data_manager_06_create_torch_data_loader(self, data_loader):
        """Test function create torch data loader"""

        self.torch_data_manager.split(0.5, test_batch_size=None)
        s = self.torch_data_manager.subset_test()

        data_loader.side_effect = TypeError()
        with self.assertRaises(FedbiomedError):
            self.torch_data_manager._create_data_loader(s)

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

    def test_torch_data_manager_09_reproducible_dataset(self):
        self.torch_data_manager = TorchDataManager(
            dataset=self.dataset,
            batch_size=48,
            shuffle=True,
        )

        test_ratio = 0.5

        # When fixing same seed, split should be the same
        # between training/testing datasets, plus order of samples
        # should be the same in each dataset
        torch.manual_seed(12345)
        loader_train, loader_test = self.torch_data_manager.split(
            test_ratio, test_batch_size=None, is_shuffled_testing_dataset=True
        )
        torch.manual_seed(12345)
        loader_train2, loader_test2 = self.torch_data_manager.split(
            test_ratio, test_batch_size=None, is_shuffled_testing_dataset=True
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

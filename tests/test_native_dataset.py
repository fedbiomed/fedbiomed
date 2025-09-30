import unittest

import numpy as np
import torch

from fedbiomed.common.dataset._native_dataset import NativeDataset
from fedbiomed.common.dataset_types import DataReturnFormat

# Assume the necessary imports and classes are already defined
# from your_module import NativeDataset, DataReturnFormat


class TestNativeDataset(unittest.TestCase):
    def test_dataset_is_array(self):
        """Test the behavior when dataset is just a numpy array."""
        dataset = np.array([1, 2, 3, 4, 5])
        target = "some_target"

        # Initialize NativeDataset
        native_dataset = NativeDataset(dataset, target)

        # Check that dataset is an array
        self.assertTrue(isinstance(native_dataset.dataset, np.ndarray))
        self.assertEqual(len(native_dataset.dataset), 5)

    def test_len_and_get_item(self):
        """Test that len and __getitem__ are implemented correctly in a dataset."""
        dataset = [1, 2, 3, 4, 5]  # Simple list to simulate a dataset
        target = "some_target"

        # Initialize NativeDataset
        native_dataset = NativeDataset(dataset, target)

        # Check that len works
        self.assertEqual(len(native_dataset.dataset), 5)

        # Check that __getitem__ works
        self.assertEqual(native_dataset.dataset[0], 1)
        self.assertEqual(native_dataset.dataset[4], 5)

    def test_torch_dataset(self):
        """Test the behavior when the dataset is a TorchDataset."""

        class MockTorchDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = MockTorchDataset([torch.tensor(1), torch.tensor(2), torch.tensor(3)])
        target = "some_target"

        # Initialize NativeDataset
        native_dataset = NativeDataset(dataset, target)

        # Check that the dataset is recognized as a TorchDataset
        self.assertTrue(native_dataset.is_torch_dataset)
        self.assertEqual(len(native_dataset.dataset), 3)

    def test_supervised_dataset(self):
        """Test the behavior when the dataset is supervised."""
        dataset = [(1, 2), (3, 4), (5, 6)]  # Supervised (returns a tuple)
        target = "some_target"

        # Initialize NativeDataset
        native_dataset = NativeDataset(dataset, target)

        # Check if it's supervised
        self.assertTrue(native_dataset.is_supervised)

    def test_unsupervised_dataset(self):
        """Test the behavior when the dataset is unsupervised."""
        dataset = [1, 3, 5]  # Unsupervised (returns a single element)
        target = "some_target"

        # Initialize NativeDataset
        native_dataset = NativeDataset(dataset, target)

        # Check if it's unsupervised
        self.assertFalse(native_dataset.is_supervised)

    def test_torch_dataset_with_sklearn_format(self):
        """Test when it's a TorchDataset, but the framework (to_format) is SkLearn."""

        class MockTorchDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = MockTorchDataset([torch.tensor(1), torch.tensor(2), torch.tensor(3)])
        target = "some_target"

        # Initialize NativeDataset
        native_dataset = NativeDataset(dataset, target)

        # Set framework to SkLearn
        native_dataset.complete_initialization(
            controller_kwargs={}, to_format=DataReturnFormat.SKLEARN
        )

        # Check that the dataset is converted to numpy array
        self.assertTrue(isinstance(native_dataset.dataset[0], np.ndarray))

    def test_sklearn_dataset_with_torch_format(self):
        """Test when it's a Sklearn dataset, but the framework (to_format) is Torch."""
        dataset = np.array([1, 2, 3, 4, 5])
        target = "some_target"

        # Initialize NativeDataset
        native_dataset = NativeDataset(dataset, target)

        # Set framework to Torch
        native_dataset.complete_initialization(
            controller_kwargs={}, to_format=DataReturnFormat.TORCH
        )

        # Check that the dataset is converted to torch tensor
        self.assertTrue(isinstance(native_dataset.dataset[0], torch.Tensor))

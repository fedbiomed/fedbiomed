import numpy as np
import pytest
import torch

from fedbiomed.common.dataset._custom_dataset import CustomDataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError


# Valid implementation for testing
class ValidCustomDataset(CustomDataset):
    def read(self):
        self.data = torch.randn(10, 5)  # 10 samples, 5 features
        self.targets = torch.randint(0, 2, (10,))  # Binary targets

    def __len__(self):
        return len(self.data)

    def get_item(self, idx):
        return self.data[idx], self.targets[idx]


class WrongFormatDataset(CustomDataset):
    def read(self):
        self.data = np.random.randn(10, 5)  # NumPy array instead of torch.Tensor
        self.targets = np.random.randint(0, 2, 10)

    def __len__(self):
        return 10

    def get_item(self, idx):
        return self.data[idx], self.targets[idx]


def test_valid_dataset_creation():
    dataset = ValidCustomDataset()
    dataset.complete_initialization(path="dummy_path", to_format=DataReturnFormat.TORCH)
    assert len(dataset) == 10
    data, target = dataset[0]
    assert isinstance(data, torch.Tensor)
    assert isinstance(target, torch.Tensor)


def test_missing_read_method():
    with pytest.raises(FedbiomedError, match="must implement a 'read' method"):

        class NoReadDataset(CustomDataset):
            def __len__(self):
                return 0

            def get_item(self, idx):
                return None, None


def test_missing_get_item_method():
    with pytest.raises(FedbiomedError, match="must implement a 'get_item' method"):

        class NoGetItemDataset(CustomDataset):
            def read(self):
                pass

            def __len__(self):
                return 0


def test_missing_len_method():
    with pytest.raises(FedbiomedError, match="must implement a '__len__' method"):

        class NoLenDataset(CustomDataset):
            def read(self):
                pass

            def get_item(self, idx):
                return None, None


def test_non_tuple_return():
    with pytest.raises(FedbiomedError):

        class NonTupleReturnDataset(CustomDataset):
            def read(self):
                self.data = torch.randn(10, 5)
                self.targets = torch.randint(0, 2, (10,))

            def __len__(self):
                return len(self.data)

            def get_item(self, idx):
                return self.data[idx]  # Not returning a tuple


def test_wrong_format():
    dataset = WrongFormatDataset()
    with pytest.raises(FedbiomedError):
        dataset.complete_initialization(
            path="dummy_path", to_format=DataReturnFormat.TORCH
        )

    with pytest.raises(FedbiomedError):
        _ = dataset[0]


def test_override_getitem():
    with pytest.raises(
        FedbiomedError, match="Overriding __getitem__ .* is not allowed"
    ):

        class OverrideGetItemDataset(CustomDataset):
            def read(self):
                pass

            def __len__(self):
                return 0

            def __getitem__(self, idx):
                return None, None


def test_override_init():
    with pytest.raises(FedbiomedError, match="Overriding __init__ .* is not allowed"):

        class OverrideInitDataset(CustomDataset):
            def __init__(self):
                pass

            def read(self):
                pass

            def get_item(self, idx):
                return None, None

            def __len__(self):
                return 0


def test_initialization_parameters():
    dataset = ValidCustomDataset()
    path = "test_path"
    dataset.complete_initialization(path=path, to_format=DataReturnFormat.TORCH)
    assert dataset.path == path
    assert dataset._to_format == DataReturnFormat.TORCH


def test_data_access():
    dataset = ValidCustomDataset()
    dataset.complete_initialization(path="dummy_path", to_format=DataReturnFormat.TORCH)

    # Test multiple indices
    for i in range(len(dataset)):
        data, target = dataset[i]
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert data.shape == (5,)  # Check expected shape
        assert target.shape == ()  # Single target value

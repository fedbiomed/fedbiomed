import numpy as np
import pytest
import torch

from fedbiomed.common.dataset import CustomDataset
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
    dataset.complete_initialization(
        controller_kwargs={"root": "dummy_path"}, to_format=DataReturnFormat.TORCH
    )
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


def test_wrong_format():
    dataset = WrongFormatDataset()
    with pytest.raises(FedbiomedError):
        dataset.complete_initialization(
            controller_kwargs={"root": "dummy_path"}, to_format=DataReturnFormat.TORCH
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
    dataset.complete_initialization(
        controller_kwargs={"root": path}, to_format=DataReturnFormat.TORCH
    )
    assert dataset.path == path
    assert dataset._to_format == DataReturnFormat.TORCH


def test_data_access():
    dataset = ValidCustomDataset()
    dataset.complete_initialization(
        controller_kwargs={"root": "dummy_path"}, to_format=DataReturnFormat.TORCH
    )

    # Test multiple indices
    for i in range(len(dataset)):
        data, target = dataset[i]
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert data.shape == (5,)  # Check expected shape
        assert target.shape == ()  # Single target value


def test_read_exception_is_wrapped(tmp_path):
    class DS(CustomDataset):
        def read(self):
            raise ValueError("boom")

        def get_item(self, idx):
            return ([], [])

        def __len__(self):
            return 1

    ds = DS()
    with pytest.raises(FedbiomedError):
        ds.complete_initialization(
            controller_kwargs={"root": str(tmp_path)},
            to_format=DataReturnFormat.TORCH,
        )


def test_get_item_exception_is_wrapped(tmp_path):
    class DS(CustomDataset):
        def read(self): ...
        def get_item(self, idx):
            raise RuntimeError("cannot get")

        def __len__(self):
            return 1

    ds = DS()
    with pytest.raises(FedbiomedError):
        ds.complete_initialization(
            controller_kwargs={"root": str(tmp_path)},
            to_format=DataReturnFormat.TORCH,
        )


def test_get_item_must_return_tuple_of_len_2(tmp_path):
    class DSWrongTuple(CustomDataset):
        def read(self): ...
        def get_item(self, idx):
            return [1, 2, 3]  # not a (data, target) tuple

        def __len__(self):
            return 1

    ds = DSWrongTuple()
    with pytest.raises(FedbiomedError):
        ds.complete_initialization(
            controller_kwargs={"root": str(tmp_path)},
            to_format=DataReturnFormat.TORCH,
        )

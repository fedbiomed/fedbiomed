import numpy as np
import pytest
import torch

from fedbiomed.common.dataset import CustomDataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError

# Define dataset classes for testing purposes


class ValidTorchDataset(CustomDataset):
    def read(self):
        self.data = torch.randn(10, 5)  # 10 samples, 5 features
        self.targets = torch.randint(0, 2, (10,))  # Binary targets

    def __len__(self):
        return len(self.data)

    def get_item(self, idx):
        return self.data[idx], self.targets[idx]


class ValidTorchComposedDataset(ValidTorchDataset):
    def get_item(self, idx):
        data = {
            "modality_1": self.data[idx],
            "modality_2": self.data[idx] * 2,
        }
        target = {
            "label_1": self.targets[idx],
            "label_2": (self.targets[idx] + 1) % 2,
        }
        return data, target


class ValidNumpyDataset(CustomDataset):
    def read(self):
        self.data = np.random.randn(10, 5).astype(np.float32)
        self.targets = np.random.randint(0, 2, (10, 1))

    def __len__(self):
        return 10

    def get_item(self, idx):
        return self.data[idx], self.targets[idx]


class ValidNumpyComposedDataset(ValidNumpyDataset):
    def get_item(self, idx):
        data = {
            "modality_1": self.data[idx],
            "modality_2": self.data[idx] * 2,
        }
        target = {
            "label_1": self.targets[idx],
            "label_2": (self.targets[idx] + 1) % 2,
        }
        return data, target


class WrongFormatDataset(CustomDataset):
    def read(self):
        self.data = np.random.randn(10, 5).astype(np.float32)
        self.targets = np.random.randint(0, 2, 10)  # Int instead of ndarray

    def __len__(self):
        return 10

    def get_item(self, idx):
        return self.data[idx], self.targets[idx]


class WrongFormatComposedDataset(WrongFormatDataset):
    def get_item(self, idx):
        data = {
            "modality_1": self.data[idx],
            "modality_2": self.data[idx] * 2,
        }
        target = {
            "label_1": self.targets[idx],  # int instead of array
            "label_2": (self.targets[idx] + 1) % 2,
        }
        return data, target


class WrongFormatChangingDataset(ValidNumpyDataset):
    def get_item(self, idx):
        if idx % 2 != 0:
            return {"data": self.data[idx]}, {"target": self.targets[idx]}
        else:
            return self.data[idx], self.targets[idx]


class WrongFormatChangingComposedDataset(ValidNumpyDataset):
    def get_item(self, idx):
        if idx % 2 != 1:
            return {"data": self.data[idx]}, {"target": self.targets[idx]}
        else:
            return self.data[idx], self.targets[idx]


class EmptyDataset(CustomDataset):
    def read(self):
        pass

    def __len__(self):
        return 0

    def get_item(self, idx):
        return torch.tensor([]), torch.tensor([])


# Test dataset creation and initialization


def test_valid_dataset_creation():
    dataset = ValidTorchDataset()
    dataset.load(
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
    _dataset_classes = [
        WrongFormatDataset,
        WrongFormatComposedDataset,
    ]
    for ds_class in _dataset_classes:
        dataset = ds_class()
        with pytest.raises(FedbiomedError):
            dataset.load(
                controller_kwargs={"root": "dummy_path"},
                to_format=DataReturnFormat.SKLEARN,
            )

        with pytest.raises(FedbiomedError):
            _ = dataset[1]


def test_wrong_format_changing_dataset():
    dataset = WrongFormatChangingDataset()
    dataset.load(
        controller_kwargs={"root": "dummy_path"},
        to_format=DataReturnFormat.SKLEARN,
    )

    with pytest.raises(FedbiomedError):
        _ = dataset[1]


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
    dataset = ValidTorchDataset()
    path = "test_path"
    dataset.load(controller_kwargs={"root": path}, to_format=DataReturnFormat.TORCH)
    assert dataset.path == path
    assert dataset._to_format == DataReturnFormat.TORCH


def test_data_access():
    dataset = ValidTorchDataset()
    dataset.load(
        controller_kwargs={"root": "dummy_path"}, to_format=DataReturnFormat.TORCH
    )

    # Test multiple indices
    for i in range(len(dataset)):
        data, target = dataset[i]
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert data.shape == (5,)  # Check expected shape
        assert target.shape == ()  # Single target value


def test_load_missing_root_key():
    """load raises when 'root' key is absent from controller_kwargs."""
    ds = ValidTorchDataset()
    with pytest.raises(FedbiomedError, match="'root' must be provided"):
        ds.load(controller_kwargs={}, to_format=DataReturnFormat.SKLEARN)


def test_load_root_is_none():
    """load raises when 'root' is explicitly None."""
    ds = ValidTorchDataset()
    with pytest.raises(FedbiomedError, match="'root' must be provided"):
        ds.load(controller_kwargs={"root": None}, to_format=DataReturnFormat.SKLEARN)


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
        ds.load(
            controller_kwargs={"root": str(tmp_path)},
            to_format=DataReturnFormat.TORCH,
        )


def test_load_empty_dataset():
    """load raises when dataset length is 0 after read()."""

    ds = EmptyDataset()
    with pytest.raises(FedbiomedError, match="dataset is empty"):
        ds.load(
            controller_kwargs={"root": "dummy_path"}, to_format=DataReturnFormat.TORCH
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
        ds.load(
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
        ds.load(
            controller_kwargs={"root": str(tmp_path)},
            to_format=DataReturnFormat.TORCH,
        )


# Test dataset usage


def test_getitem_not_composed():
    """__getitem__ returns a valid non-composed result."""

    _dataset_classes = [ValidTorchDataset, ValidNumpyDataset]
    _data_formats = [DataReturnFormat.TORCH, DataReturnFormat.SKLEARN]
    _data_types = [torch.Tensor, np.ndarray]

    for ds_class, d_type, _d_format in zip(
        _dataset_classes,
        _data_types,
        _data_formats,
        strict=True,
    ):
        ds = ds_class()
        ds.load(controller_kwargs={"root": "dummy_path"}, to_format=_d_format)
        result = ds.__getitem__(0)
        assert isinstance(result, tuple) and len(result) == 2
        data, target = result
        assert isinstance(data, d_type)
        assert isinstance(target, d_type)


def test_getitem_composed():
    """__getitem__ returns a valid composed dict result."""

    _dataset_classes = [ValidTorchComposedDataset, ValidNumpyComposedDataset]
    _data_formats = [DataReturnFormat.TORCH, DataReturnFormat.SKLEARN]
    _data_types = [torch.Tensor, np.ndarray]

    for ds_class, d_type, _d_format in zip(
        _dataset_classes,
        _data_types,
        _data_formats,
        strict=True,
    ):
        ds = ds_class()
        ds.load(controller_kwargs={"root": "dummy_path"}, to_format=_d_format)
        result = ds.__getitem__(0)
        assert isinstance(result, tuple) and len(result) == 2
        data, target = result
        assert isinstance(data, dict)
        assert isinstance(target, dict)
        assert "modality_1" in data and "modality_2" in data
        assert "label_1" in target and "label_2" in target
        assert isinstance(data["modality_1"], d_type)
        assert isinstance(data["modality_2"], d_type)
        assert isinstance(target["label_1"], d_type)
        assert isinstance(target["label_2"], d_type)


def test_apply_default_types_exception_is_wrapped():
    """_apply_default_types wraps errors raised while converting data."""

    ds = ValidNumpyDataset()
    ds.load(
        controller_kwargs={"root": "dummy_path"},
        to_format=DataReturnFormat.SKLEARN,
    )

    bad_data = {
        "modality_1": np.array([1.0, 2.0], dtype=np.float32),
        "modality_2": "not-an-array",
    }

    with pytest.raises(
        FedbiomedError, match="Failed to apply default training plan types"
    ):
        ds._apply_default_types(bad_data, "data")

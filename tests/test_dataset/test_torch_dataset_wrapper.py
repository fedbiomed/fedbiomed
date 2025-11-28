import numpy as np
import pytest
import torch

from fedbiomed.common.datamanager._torch_data_manager import _DatasetWrapper
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.exceptions import FedbiomedError


class NoneDataDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return None, torch.tensor([1.0])

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        return sample


class NonTensorDataDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return np.array([1.0, 2.0]), torch.tensor([1.0])

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        return sample


class DictNonTensorValuesDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"a": np.array([1.0]), "b": np.array([2.0])}, torch.tensor([1.0])

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        return sample


class NoneTargetDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor([1.0]), None

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        return sample


class InvalidTargetTypeDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor([1.0]), 999

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        return sample


class ValidDictTensorDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"x": torch.tensor([1.0, 2.0])}, {"x": torch.tensor([1.0])}

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        return sample


def test_more_info_non_dict():
    w = _DatasetWrapper(NoneDataDataset())
    assert "type" in w._more_info(3.14)


def test_more_info_dict_with_non_tensor_values():
    w = _DatasetWrapper(NoneDataDataset())
    info = w._more_info({"a": np.array([1.0]), "b": "x"})
    assert "dict with modalities" in info


def test_getitem_none_data():
    w = _DatasetWrapper(NoneDataDataset())
    with pytest.raises(FedbiomedError):
        w[0]


def test_getitem_non_tensor_data():
    w = _DatasetWrapper(NonTensorDataDataset())
    with pytest.raises(FedbiomedError):
        w[0]


def test_getitem_dict_non_tensor_values():
    w = _DatasetWrapper(DictNonTensorValuesDataset())
    with pytest.raises(FedbiomedError):
        w[0]


def test_getitem_none_target():
    w = _DatasetWrapper(NoneTargetDataset())
    with pytest.raises(FedbiomedError):
        w[0]


def test_getitem_invalid_target_type():
    w = _DatasetWrapper(InvalidTargetTypeDataset())
    with pytest.raises(FedbiomedError):
        w[0]


def test_getitem_valid_dict_tensor():
    w = _DatasetWrapper(ValidDictTensorDataset())
    data, target = w[0]
    assert isinstance(data, dict)
    assert isinstance(target, dict)
    assert isinstance(data["x"], torch.Tensor)
    assert isinstance(target["x"], torch.Tensor)

import pytest

from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError


# Minimal concrete subclass for testing
class DummyController:
    def __init__(self, **kwargs):
        pass

    def get_sample(self, idx):
        return {"data": 1, "target": 2}

    def __len__(self):
        return 42


class DummyDataset(Dataset):
    _native_to_framework = {DataReturnFormat.SKLEARN: lambda x: x}
    _controller_cls = DummyController
    _transform = staticmethod(lambda x: x)
    _target_transform = staticmethod(lambda x: x)

    def complete_initialization(self):
        super().complete_initialization()

    def __getitem__(self, idx):
        return (1, 2)


def test_01_to_format_setter_getter():
    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN
    assert ds.to_format == DataReturnFormat.SKLEARN


def test_02_to_format_setter_invalid():
    ds = DummyDataset()
    with pytest.raises(FedbiomedValueError):
        ds.to_format = "not_a_format"


def test_03_get_format_conversion_callable_not_set():
    ds = DummyDataset()
    with pytest.raises(AttributeError):
        ds._get_format_conversion_callable()


def test_04_get_format_conversion_callable_valid():
    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN
    assert callable(ds._get_format_conversion_callable())


def test_05_validate_transform_none():
    ds = DummyDataset()
    fn = ds._validate_transform(None)
    assert fn(123) == 123


def test_06_validate_transform_callable():
    ds = DummyDataset()
    fn = ds._validate_transform(lambda x: x + 1)
    assert fn(1) == 2


def test_07_validate_transform_invalid():
    ds = DummyDataset()
    with pytest.raises(FedbiomedValueError):
        ds._validate_transform("not_callable")


def test_08_init_controller_invalid_type():
    ds = DummyDataset()
    with pytest.raises(FedbiomedError):
        ds._init_controller("not_a_dict")


def test_09_init_controller_valid():
    ds = DummyDataset()
    ds._init_controller({})
    assert isinstance(ds._controller, DummyController)


def test_10_len():
    ds = DummyDataset()
    ds._init_controller({})
    assert len(ds) == 42


def test_11_validate_format_conversion_pipeline_type_error():
    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN
    ds._native_to_framework = {DataReturnFormat.SKLEARN: lambda x: "wrong_type"}
    ds._to_format = DataReturnFormat.SKLEARN
    with pytest.raises(FedbiomedError):
        ds._validate_format_conversion("data")


def test_12_validate_transformation_pipeline_type_error():
    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN
    with pytest.raises(FedbiomedError):
        ds._validate_transformation("data", lambda x: "wrong_type")

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

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

    def complete_initialization(self, controller_kwargs=None, to_format=None):
        super().complete_initialization(controller_kwargs, to_format)

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


def test_13_default_types_torch():
    ds = DummyDataset()

    # float tensor -> default float dtype
    x_float = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y_float = ds._default_types_torch(x_float)
    assert isinstance(y_float, torch.Tensor)
    assert y_float.dtype == torch.get_default_dtype()

    # int32 tensor -> long
    x_int = torch.tensor([1, 2], dtype=torch.int32)
    y_int = ds._default_types_torch(x_int)
    assert y_int.dtype == torch.long

    # bool tensor remains unchanged
    x_bool = torch.tensor([True, False])
    y_bool = ds._default_types_torch(x_bool)
    assert y_bool.dtype == x_bool.dtype

    with pytest.raises(FedbiomedError):
        ds._default_types_torch([1, 2, 3])


def test_14_default_types_sklearn():
    ds = DummyDataset()

    x_float = np.array([1.0, 2.0], dtype=np.float32)
    y_float = ds._default_types_sklearn(x_float)
    assert y_float.dtype == np.float64

    x_int = np.array([1, 2], dtype=np.int32)
    y_int = ds._default_types_sklearn(x_int)
    assert y_int.dtype == np.int64

    # non-numeric dtype left untouched
    x_str = np.array(["a", "b"], dtype=object)
    y_str = ds._default_types_sklearn(x_str)
    assert y_str is x_str

    with pytest.raises(FedbiomedError):
        ds._default_types_sklearn([1, 2, 3])


def test_15_get_default_types():
    ds = DummyDataset()

    # SKLEARN case: function returned should behave like _default_types_sklearn
    ds.to_format = DataReturnFormat.SKLEARN
    fn = ds._get_default_types_callable()

    x = np.array([1, 2], dtype=np.int32)
    out_via_fn = fn(x)
    out_direct = ds._default_types_sklearn(x)
    np.testing.assert_array_equal(out_via_fn, out_direct)

    # TORCH case: function returned should behave like _default_types_torch
    ds.to_format = DataReturnFormat.TORCH
    fn2 = ds._get_default_types_callable()

    t = torch.tensor([1, 2], dtype=torch.int32)
    out_via_fn2 = fn2(t)
    out_direct2 = ds._default_types_torch(t)
    assert torch.equal(out_via_fn2, out_direct2)


def test_16_validate_format_conversion():
    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN

    arr = np.array([1, 2])

    # conversion returns correct type
    ds._native_to_framework = {DataReturnFormat.SKLEARN: lambda x: x}
    out = ds._validate_format_conversion(arr)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)

    def boom(_):
        raise ValueError("boom")

    ds._native_to_framework = {DataReturnFormat.SKLEARN: boom}

    with pytest.raises(FedbiomedError) as exc:
        ds._validate_format_conversion(np.array([1, 2]))
    assert "Unable to perform type conversion" in str(exc.value)


def test_17_validate_transformation_success():
    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN

    x = np.array([1, 2])
    out = ds._validate_transformation(x, lambda a: a)
    np.testing.assert_array_equal(out, x)

    def boom(_):
        raise ValueError("boom")

    with pytest.raises(FedbiomedError) as exc:
        ds._validate_transformation(np.array([1, 2]), boom)
    assert "Unable to apply transform" in str(exc.value)


def test_18_iter():
    ds = DummyDataset()
    ds._init_controller({})
    items = list(ds)
    assert len(items) == 42
    assert all(item == (1, 2) for item in items)


def test_19_complete_initialization_calls_super():
    """Covers abstract body at line 51."""
    ds = DummyDataset()
    ds.complete_initialization(
        controller_kwargs={}, to_format=DataReturnFormat.SKLEARN
    )  # DummyDataset calls super() → hits `pass`


def test_20_abstract_getitem_body():
    """Covers abstract body at line 55 by calling super().__getitem__."""

    class SuperCallingDataset(Dataset):
        _native_to_framework = {DataReturnFormat.SKLEARN: lambda x: x}
        _controller_cls = DummyController
        _transform = staticmethod(lambda x: x)
        _target_transform = staticmethod(lambda x: x)

        def complete_initialization(self):
            pass

        def __getitem__(self, idx):
            return super().__getitem__(idx)  # hits the abstract `pass`

    ds = SuperCallingDataset()
    result = ds.__getitem__(0)
    assert result is None


def test_21_get_default_types_callable_not_set():
    """Covers line 133: AttributeError when _to_format is not set."""
    ds = DummyDataset()
    with pytest.raises(AttributeError):
        ds._get_default_types_callable()


def test_22_validate_format_and_transformations_success():
    """Covers lines 243-253: full format + transform pipeline."""
    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN
    ds._native_to_framework = {DataReturnFormat.SKLEARN: lambda x: x}

    arr = np.array([1.0, 2.0])
    out = ds._validate_format_and_transformations(arr, lambda x: x, label="col1")
    np.testing.assert_array_equal(out, arr)


def test_23_validate_format_and_transformations_no_label():
    """Covers the `label=None` branch in _validate_format_and_transformations."""
    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN
    ds._native_to_framework = {DataReturnFormat.SKLEARN: lambda x: x}

    arr = np.array([3.0])
    out = ds._validate_format_and_transformations(arr, lambda x: x)
    np.testing.assert_array_equal(out, arr)


def test_24_init_controller_instantiation_failure():
    """Covers lines 275-276: FedbiomedError when controller __init__ raises."""

    class BrokenController:
        def __init__(self, **kwargs):
            raise RuntimeError("broken")

    class BrokenDataset(Dataset):
        _native_to_framework = {}
        _controller_cls = BrokenController
        _transform = staticmethod(lambda x: x)
        _target_transform = staticmethod(lambda x: x)

        def complete_initialization(self):
            pass

        def __getitem__(self, idx):
            pass

    ds = BrokenDataset()
    with pytest.raises(FedbiomedError, match="Failed to create Controller"):
        ds._init_controller({})


def test_25_apply_transforms_no_target():
    """Covers apply_transforms when sample has no target (lines 289-308)."""
    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN
    ds._native_to_framework = {DataReturnFormat.SKLEARN: lambda x: x}

    sample = {"data": np.array([1.0, 2.0]), "target": None}
    result = ds.apply_transforms(sample)
    assert isinstance(result["data"], np.ndarray)
    assert result["target"] is None


def test_26_apply_transforms_with_target():
    """Covers apply_transforms when sample has a target (lines 309-328)."""
    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN
    ds._native_to_framework = {DataReturnFormat.SKLEARN: lambda x: x}

    sample = {"data": np.array([1.0, 2.0]), "target": np.array([0.0])}
    result = ds.apply_transforms(sample)
    assert isinstance(result["data"], np.ndarray)
    assert isinstance(result["target"], np.ndarray)


def test_27_apply_transforms_transform_failure():
    """Covers the except branch in apply_transforms data transform (line 296)."""

    def boom(_):
        raise ValueError("boom")

    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN
    ds._native_to_framework = {DataReturnFormat.SKLEARN: lambda x: x}
    ds._transform = boom  # instance attribute shadows class one

    with pytest.raises(FedbiomedError, match="Failed to apply `transform` to `data`"):
        ds.apply_transforms({"data": np.array([1.0]), "target": None})


def test_28_apply_transforms_target_transform_failure():
    """Covers the except branch for target_transform failure (line 317)."""

    def boom(_):
        raise ValueError("boom")

    ds = DummyDataset()
    ds.to_format = DataReturnFormat.SKLEARN
    ds._native_to_framework = {DataReturnFormat.SKLEARN: lambda x: x}
    ds._target_transform = boom  # instance attribute shadows class one

    with pytest.raises(FedbiomedError, match="Failed to apply `target_transform`"):
        ds.apply_transforms({"data": np.array([1.0]), "target": np.array([0.0])})


def test_29_compute_stats():
    """Covers lines 352-353: compute_stats delegates to AnalyticsOrchestrator."""
    ds = DummyDataset()

    mock_result = {"mean": 1.0}
    with patch(
        "fedbiomed.common.dataset._dataset.AnalyticsOrchestrator"
    ) as MockOrchestrator:
        mock_instance = MagicMock()
        mock_instance.compute_stats.return_value = mock_result
        MockOrchestrator.return_value = mock_instance

        result = ds.compute_stats(stats=["mean"])

    MockOrchestrator.assert_called_once()
    mock_instance.compute_stats.assert_called_once_with(
        ds, dataset_schema=None, stats=["mean"], stats_args=None
    )
    assert result == mock_result


def test_30_len_no_controller():
    """Covers line 362: FedbiomedError when __len__ called before _init_controller."""
    ds = DummyDataset()
    with pytest.raises(FedbiomedError, match="has not completed"):
        len(ds)

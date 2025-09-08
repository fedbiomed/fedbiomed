# tests/test_tabular_dataset.py
import numpy as np
import polars as pl
import pytest

from fedbiomed.common.dataset._tabular_dataset import TabularDataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

# ---------- complete_initialization wiring ----------


def test_complete_initialization_wires_controller_and_validates(mocker):
    ds = TabularDataset(
        input_columns=["data"], target_columns=["target"], transform=None
    )

    df = pl.DataFrame({"data": [1], "target": [2]})

    # Stub a controller that returns a sample
    class StubController:
        def get_sample(self, idx):
            assert idx == 0
            return df

        def normalize_columns(self, cols):
            return cols

    captured_kwargs = {}

    def fake_init_controller(*, controller_kwargs):
        captured_kwargs.update(controller_kwargs)
        ds._controller = StubController()

    # Patch instance methods
    mocker.patch.object(ds, "_init_controller", side_effect=fake_init_controller)

    ds.complete_initialization(
        controller_kwargs={"root": "/path"},
        to_format=DataReturnFormat.SKLEARN,
    )

    # Controller created and stored
    assert hasattr(ds, "_controller") and ds._controller is not None

    # The kwargs were enriched with columns
    assert captured_kwargs["root"] == "/path"


# ---------- _get_format_conversion_callable ----------


def test_get_format_conversion_callable_returns_sklearn_callable():
    ds = TabularDataset(input_columns=["a"], target_columns=["y"], transform=None)
    ds.to_format = DataReturnFormat.SKLEARN
    fn = ds._get_format_conversion_callable()
    # It should be the exact function stored in the class mapping
    assert fn is TabularDataset._native_to_framework[DataReturnFormat.SKLEARN]


def test_get_format_conversion_callable_returns_torch_callable():
    ds = TabularDataset(input_columns=["a"], target_columns=["y"], transform=None)
    ds.to_format = DataReturnFormat.TORCH
    fn = ds._get_format_conversion_callable()
    assert fn is TabularDataset._native_to_framework[DataReturnFormat.TORCH]


def test_get_format_conversion_callable_raises_for_unknown_format():
    ds = TabularDataset(input_columns=["a"], target_columns=["y"], transform=None)

    class UnknownFmt:
        pass

    # Force an unknown format key
    ds._to_format = UnknownFmt()
    with pytest.raises(KeyError):
        _ = ds._get_format_conversion_callable()


# ---------- __getitem__ behavior ----------


def test_getitem_happy_path_with_transform_and_conversion(mocker):
    # transform multiplies numeric inputs
    ds = TabularDataset(
        input_columns=[1],
        target_columns=[2],
        transform=lambda x: x * 10,
        target_transform=lambda x: x * 10,
    )

    # Pretend controller returns raw sample
    class StubController:
        def get_sample(self, idx):
            assert idx == 7
            return pl.DataFrame({"col1": [1], "col2": [2], "col3": [3]})[0]

        def normalize_columns(self, cols):
            return cols

    ds._controller = StubController()
    ds._to_format = DataReturnFormat.SKLEARN  # proper enum (typed OK)

    # Conversion function is identity for this test
    data, target = ds[7]
    assert data == [[10]]
    assert target == [[20]]


def test_getitem_raises_if_not_initialized():
    ds = TabularDataset(input_columns=[1], target_columns=[2], transform=None)
    with pytest.raises(FedbiomedError) as exc:
        _ = ds[0]
    assert "has not completed initialization" in str(exc.value)


def test_getitem_transform_error_on_data(mocker):
    # transform raises
    def boom(_):
        raise ValueError("boom")

    ds = TabularDataset(input_columns=[1], target_columns=[2], transform=boom)
    ds.to_format = DataReturnFormat.SKLEARN

    class StubController:
        def get_sample(self, idx):
            return pl.DataFrame({"c1": [1], "c2": [2]})

        def normalize_columns(self, cols):
            return cols

    ds._controller = StubController()
    mocker.patch.object(ds, "_get_format_conversion_callable", return_value=lambda x: x)

    with pytest.raises(FedbiomedError) as exc:
        _ = ds[5]
    assert "Failed to apply `transform` to `data`" in str(exc.value)


def test_getitem_transform_error_on_target(mocker):
    # Transform fails only for target (= 2)
    def sometimes_fails(x):
        print(x)
        if x == [[1]]:
            raise ValueError("nope")
        return x * 10

    ds = TabularDataset(
        input_columns=[1], target_columns=[2], transform=sometimes_fails
    )
    ds._to_format = DataReturnFormat.SKLEARN

    class StubController:
        def get_sample(self, idx):
            return pl.DataFrame({"c1": [1], "c2": [2]})

        def normalize_columns(self, cols):
            return cols

    ds._controller = StubController()

    with pytest.raises(FedbiomedError) as _exc:
        _ = ds[5]


# ---------- _validate_transform ----------


def test_validate_transform_accepts_none_identity():
    ds = TabularDataset(input_columns=[0], target_columns=[1], transform=None)
    assert ds._transform("x") == "x"


def test_validate_transform_accepts_callable():
    ds = TabularDataset(
        input_columns=[0], target_columns=[1], transform=lambda z: z + 1
    )
    assert ds._transform(41) == 42


def test_validate_transform_rejects_other_types():
    with pytest.raises(FedbiomedValueError):
        _ = TabularDataset(
            input_columns=[0], target_columns=[1], transform="not callable"
        )


# ---------- _validate_pipeline ----------


def test_validate_pipeline_raises_on_conversion_failure(mocker):
    ds = TabularDataset(input_columns=[0], target_columns=[1], transform=None)
    ds.to_format = DataReturnFormat.SKLEARN

    # Force conversion to fail
    mocker.patch.object(
        ds,
        "_get_format_conversion_callable",
        return_value=lambda _: (_ for _ in ()).throw(ValueError("boom")),
    )

    with pytest.raises(FedbiomedError) as exc:
        ds._validate_format_and_transformations(data="payload", transform=lambda x: x)
    assert "Unable to perform type conversion" in str(exc.value)


def test_validate_pipeline_raises_on_type_mismatch_after_conversion(mocker):
    ds = TabularDataset(input_columns=[0], target_columns=[1], transform=None)
    ds.to_format = DataReturnFormat.SKLEARN  # expected type: numpy.ndarray

    # Convert returns the wrong type instance
    mocker.patch.object(
        ds, "_get_format_conversion_callable", return_value=lambda _: {"not": "ndarray"}
    )

    with pytest.raises(FedbiomedError) as exc:
        ds._validate_format_and_transformations(data="payload", transform=lambda x: x)
    assert "Expected type conversion" in str(exc.value)


def test_validate_pipeline_raises_when_transform_crashes(mocker):
    ds = TabularDataset(input_columns=[0], target_columns=[1], transform=None)
    ds.to_format = DataReturnFormat.SKLEARN  # expected type: numpy.ndarray

    # Conversion returns correct type
    mocker.patch.object(
        ds,
        "_get_format_conversion_callable",
        return_value=lambda _: np.array([1, 2, 3]),
    )

    with pytest.raises(FedbiomedError) as exc:
        ds._validate_format_and_transformations(
            data="payload",
            transform=lambda _: (_ for _ in ()).throw(ValueError("boom")),
        )
    assert "Unable to apply transform" in str(exc.value)


def test_validate_pipeline_raises_when_transform_returns_wrong_type(mocker):
    ds = TabularDataset(input_columns=[0], target_columns=[1], transform=None)
    ds.to_format = DataReturnFormat.SKLEARN  # expected type: numpy.ndarray

    # Conversion ok -> numpy.ndarray
    mocker.patch.object(
        ds,
        "_get_format_conversion_callable",
        return_value=lambda _: np.array([1, 2, 3]),
    )

    class Wrong: ...

    # Transform returns Wrong
    with pytest.raises(FedbiomedError) as exc:
        ds._validate_format_and_transformations(
            data="payload", transform=lambda _: Wrong()
        )
    assert "Expected `transform` to return" in str(exc.value)


# ---------- _apply_transforms ----------


def test_apply_transforms_happy_path_uses_conversion_and_transform():
    # Dummy carrier that supports both conversions used by the mapping
    class Carrier:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def to_numpy(self):
            return self._arr

        def to_torch(self):  # not used in this test, but present to mirror mapping
            return ("torch", tuple(self._arr.tolist()))

    # Transform applied AFTER conversion
    def transform(x):
        # For SKLEARN this will receive a numpy array
        return x * 2

    ds = TabularDataset(input_columns=["a"], target_columns=["y"], transform=transform)
    ds.to_format = DataReturnFormat.SKLEARN

    sample = {"data": Carrier([1, 2]), "target": Carrier([3, 4])}
    data = ds.apply_transforms(sample)
    np.testing.assert_array_equal(data["data"], np.array([2, 4]))
    np.testing.assert_array_equal(data["target"], np.array([3, 4]))


def test_apply_transforms_wraps_errors_in_FedbiomedError(mocker):
    ds = TabularDataset(
        input_columns=["a"], target_columns=["y"], transform=lambda x: x
    )

    # Cause the conversion step to fail
    def boom(_):
        raise ValueError("conversion failed")

    ds.to_format = DataReturnFormat.SKLEARN
    mocker.patch.object(ds, "_get_format_conversion_callable", return_value=boom)

    # Content doesn't matter; conversion callable will raise
    sample = {"data": object(), "target": object()}

    with pytest.raises(FedbiomedError) as _exc:
        _ = ds.apply_transforms(sample)

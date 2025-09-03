# tests/test_tabular_dataset.py
import numpy as np
import pytest

from fedbiomed.common.dataset._tabular_dataset import TabularDataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

# ---------- complete_initialization wiring ----------


def test_complete_initialization_wires_controller_and_validates(mocker):
    ds = TabularDataset(input_columns=["a"], target_columns=["y"], transform=None)

    # Stub a controller that returns a sample
    class StubController:
        def _get_nontransformed_item(self, idx):
            assert idx == 0
            return {"data": "D0", "target": "T0"}

    captured_kwargs = {}

    def fake_init_controller(*, controller_kwargs):
        captured_kwargs.update(controller_kwargs)
        ds._controller = StubController()

    # Patch instance methods
    mocker.patch.object(ds, "_init_controller", side_effect=fake_init_controller)
    validate_spy = mocker.patch.object(ds, "_validate_pipeline", return_value=None)

    ds.complete_initialization(
        controller_kwargs={"root": "/path"},
        to_format=DataReturnFormat.SKLEARN,
    )

    # Controller created and stored
    assert hasattr(ds, "_controller") and ds._controller is not None

    # The kwargs were enriched with columns
    assert captured_kwargs["root"] == "/path"
    assert captured_kwargs["input_columns"] == ["a"]
    assert captured_kwargs["target_columns"] == ["y"]

    # _validate_pipeline called for both data and target with ds._transform
    validate_spy.assert_any_call("D0", transform=ds._transform)
    validate_spy.assert_any_call("T0", transform=ds._transform)
    assert validate_spy.call_count == 2


# ---------- __getitem__ behavior ----------


def test_getitem_happy_path_with_transform_and_conversion(mocker):
    # transform multiplies numeric inputs
    ds = TabularDataset(
        input_columns=[1], target_columns=[2], transform=lambda x: x * 10
    )

    # Pretend controller returns raw sample
    class StubController:
        def _get_nontransformed_item(self, idx):
            assert idx == 7
            return {"data": 1, "target": 2}

    ds._controller = StubController()
    ds.to_format = DataReturnFormat.SKLEARN  # proper enum (typed OK)

    # Conversion function is identity for this test
    mocker.patch.object(ds, "_get_format_conversion_callable", return_value=lambda x: x)

    data, target = ds[7]
    assert data == 10
    assert target == 20


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
        def _get_nontransformed_item(self, idx):
            return {"data": 1, "target": 2}

    ds._controller = StubController()
    mocker.patch.object(ds, "_get_format_conversion_callable", return_value=lambda x: x)

    with pytest.raises(FedbiomedError) as exc:
        _ = ds[5]
    assert "Failed to apply `transform` to `data`" in str(exc.value)


def test_getitem_transform_error_on_target(mocker):
    # Transform fails only for target (= 2)
    def sometimes_fails(x):
        if x == 2:
            raise ValueError("nope")
        return x * 10

    ds = TabularDataset(
        input_columns=[1], target_columns=[2], transform=sometimes_fails
    )
    ds.to_format = DataReturnFormat.SKLEARN

    class StubController:
        def _get_nontransformed_item(self, idx):
            return {"data": 1, "target": 2}

    ds._controller = StubController()
    mocker.patch.object(ds, "_get_format_conversion_callable", return_value=lambda x: x)

    with pytest.raises(FedbiomedError) as exc:
        _ = ds[5]
    assert "Failed to apply `_transform` to `target`" in str(exc.value)


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
        ds._validate_pipeline(data="payload", transform=lambda x: x)
    assert "Unable to perform type conversion" in str(exc.value)


def test_validate_pipeline_raises_on_type_mismatch_after_conversion(mocker):
    ds = TabularDataset(input_columns=[0], target_columns=[1], transform=None)
    ds.to_format = DataReturnFormat.SKLEARN  # expected type: numpy.ndarray

    # Convert returns the wrong type instance
    mocker.patch.object(
        ds, "_get_format_conversion_callable", return_value=lambda _: {"not": "ndarray"}
    )

    with pytest.raises(FedbiomedError) as exc:
        ds._validate_pipeline(data="payload", transform=lambda x: x)
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
        ds._validate_pipeline(
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
        ds._validate_pipeline(data="payload", transform=lambda _: Wrong())
    assert "Expected `transform` to return" in str(exc.value)

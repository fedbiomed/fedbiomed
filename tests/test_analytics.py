# SPDX-License-Identifier: Apache-2.0

import numpy as np
import polars as pl
import pytest
import torch

from fedbiomed.common.analytics import validate_dataset_arguments_for_fa
from fedbiomed.common.constants import DatasetTypes
from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.exceptions import FedbiomedError

# =============================================================================
# Mock infrastructure (Reader + Controller)
# =============================================================================


class MockReader:
    """Minimal reader mock exposing Polars schema only."""

    def __init__(self, schema: dict[str, pl.DataType]):
        self.data = pl.DataFrame(
            {name: pl.Series(name, [], dtype=dtype) for name, dtype in schema.items()}
        )


class MockTabularController:
    """Minimal controller mock used by TabularAnalytics."""

    def __init__(self, schema: dict[str, pl.DataType]):
        self._reader = MockReader(schema)

    def normalize_columns(self, columns):
        if isinstance(columns, (str, int)):
            return [columns]
        return list(columns)


# =============================================================================
# Mock dataset
# =============================================================================


class MockTabularDataset(TabularDataset):
    """Mock dataset for TabularAnalytics tests.

    Provides:
    - __len__
    - __getitem__
    - mocked controller with Polars schema
    """

    def __init__(
        self,
        data_list,
        target_list,
        input_columns,
        target_columns,
        schema,
    ):
        if len(data_list) != len(target_list):
            raise ValueError("data_list and target_list must have same length")

        self.data_list = data_list
        self.target_list = target_list
        self._input_columns = input_columns
        self._target_columns = target_columns

        # Inject mocked controller
        self._controller = MockTabularController(schema)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.target_list[idx]


# =============================================================================
# NumPy tests
# =============================================================================


@pytest.fixture
def numpy_numeric_dataset():
    schema = {
        "age": pl.Float64,
        "weight": pl.Float64,
        "salary": pl.Float64,
    }

    data = [
        np.array([25.0, 1.0]),
        np.array([30.0, 3.0]),
        np.array([35.0, 5.0]),
    ]
    target = [
        np.array([50000.0]),
        np.array([60000.0]),
        np.array([70000.0]),
    ]

    return MockTabularDataset(
        data,
        target,
        input_columns=["age", "weight"],
        target_columns=["salary"],
        schema=schema,
    )


def test_mean_numpy_numeric(numpy_numeric_dataset):
    result = numpy_numeric_dataset.mean()

    assert set(result.keys()) == {"age", "weight", "salary"}
    assert np.isclose(result["age"], 30.0)
    assert np.isclose(result["weight"], 3.0)
    assert np.isclose(result["salary"], 60000.0)


@pytest.fixture
def numpy_mixed_dataset():
    schema = {
        "age": pl.Float64,
        "sex": pl.Utf8,
        "salary": pl.Float64,
    }

    data = [
        np.array([25.0, "M"], dtype=object),
        np.array([30.0, "F"], dtype=object),
        np.array([35.0, "M"], dtype=object),
    ]
    target = [
        np.array([50000.0]),
        np.array([60000.0]),
        np.array([70000.0]),
    ]

    return MockTabularDataset(
        data,
        target,
        input_columns=["age", "sex"],
        target_columns=["salary"],
        schema=schema,
    )


def test_validate_dataset_arguments_for_fa(numpy_mixed_dataset):
    # Valid arguments
    dataset_args = {"col_names": ["age", "sex"]}

    validate_dataset_arguments_for_fa(dataset_args, DatasetTypes.TABULAR)

    pytest.raises(
        FedbiomedError,
        validate_dataset_arguments_for_fa,
        {"invalid_arg": 123},
        DatasetTypes.TABULAR,
    )


def test_mean_numpy_ignores_non_numeric(numpy_mixed_dataset):
    result = numpy_mixed_dataset.mean()

    assert set(result.keys()) == {"age", "salary"}
    assert np.isclose(result["age"], 30.0)
    assert np.isclose(result["salary"], 60000.0)


@pytest.fixture
def numpy_all_non_numeric_dataset():
    schema = {
        "sex": pl.Utf8,
        "group": pl.Utf8,
        "label": pl.Utf8,
    }

    data = [
        np.array(["M", "A"], dtype=object),
        np.array(["F", "B"], dtype=object),
    ]
    target = [
        np.array(["X"], dtype=object),
        np.array(["Y"], dtype=object),
    ]

    return MockTabularDataset(
        data,
        target,
        input_columns=["sex", "group"],
        target_columns=["label"],
        schema=schema,
    )


def test_mean_numpy_all_non_numeric_raises(numpy_all_non_numeric_dataset):
    with pytest.raises(FedbiomedError):
        numpy_all_non_numeric_dataset.mean()


# =============================================================================
# Torch tests
# =============================================================================


@pytest.fixture
def torch_numeric_dataset():
    schema = {
        "age": pl.Float64,
        "weight": pl.Float64,
        "salary": pl.Float64,
    }

    data = [
        torch.tensor([25.0, 1.0]),
        torch.tensor([30.0, 3.0]),
        torch.tensor([35.0, 5.0]),
    ]
    target = [
        torch.tensor([50000.0]),
        torch.tensor([60000.0]),
        torch.tensor([70000.0]),
    ]

    return MockTabularDataset(
        data,
        target,
        input_columns=["age", "weight"],
        target_columns=["salary"],
        schema=schema,
    )


def test_mean_torch_numeric(torch_numeric_dataset):
    result = torch_numeric_dataset.mean()

    assert set(result.keys()) == {"age", "weight", "salary"}
    assert torch.isclose(result["age"], torch.tensor(30.0))
    assert torch.isclose(result["weight"], torch.tensor(3.0))
    assert torch.isclose(result["salary"], torch.tensor(60000.0))


@pytest.fixture
def torch_mixed_dataset():
    """
    Torch cannot store strings, so we simulate a mixed schema by:
    - schema declaring a non-numeric column
    - data only containing numeric tensor for numeric column
    Analytics must rely on schema, not tensor content.
    """
    schema = {
        "age": pl.Float64,
        "sex": pl.Utf8,  # non-numeric, must be ignored
        "salary": pl.Float64,
    }

    data = [
        torch.tensor([25.0]),
        torch.tensor([30.0]),
        torch.tensor([35.0]),
    ]
    target = [
        torch.tensor([50000.0]),
        torch.tensor([60000.0]),
        torch.tensor([70000.0]),
    ]

    return MockTabularDataset(
        data,
        target,
        input_columns=["age", "sex"],
        target_columns=["salary"],
        schema=schema,
    )


def test_mean_torch_ignores_non_numeric(torch_mixed_dataset):
    result = torch_mixed_dataset.mean()

    assert set(result.keys()) == {"age", "salary"}
    assert torch.isclose(result["age"], torch.tensor(30.0))
    assert torch.isclose(result["salary"], torch.tensor(60000.0))


@pytest.fixture
def torch_all_non_numeric_dataset():
    schema = {
        "sex": pl.Utf8,
        "group": pl.Utf8,
        "label": pl.Utf8,
    }

    data = [
        np.array(["M", "A"], dtype=object),
        np.array(["F", "B"], dtype=object),
    ]
    target = [
        np.array(["X"], dtype=object),
        np.array(["Y"], dtype=object),
    ]

    return MockTabularDataset(
        data,
        target,
        input_columns=["sex", "group"],
        target_columns=["label"],
        schema=schema,
    )


def test_mean_torch_all_non_numeric_raises(torch_all_non_numeric_dataset):
    with pytest.raises(FedbiomedError):
        torch_all_non_numeric_dataset.mean()


# =============================================================================
# Empty dataset test
# =============================================================================


@pytest.fixture
def empty_dataset():
    schema = {
        "age": pl.Float64,
        "weight": pl.Float64,
        "salary": pl.Float64,
    }

    return MockTabularDataset(
        data_list=[],
        target_list=[],
        input_columns=["age", "weight"],
        target_columns=["salary"],
        schema=schema,
    )


def test_mean_empty_dataset_raises(empty_dataset):
    with pytest.raises(FedbiomedError):
        empty_dataset.mean()

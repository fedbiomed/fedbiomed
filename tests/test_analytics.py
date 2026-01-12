# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from fedbiomed.common.analytics import (
    TabularAnalytics,
    validate_dataset_arguments_for_fa,
)
from fedbiomed.common.constants import DatasetTypes
from fedbiomed.common.exceptions import FedbiomedError


class MockTabularDataset(TabularAnalytics):
    """Mock dataset class that implements TabularAnalytics for testing"""

    def __init__(self, data_list, input_columns):
        """
        Args:
            data_list: List of tuples (data, target) where data is np.ndarray
            input_columns: List of column names
        """
        self._data_list = data_list
        self._input_columns = input_columns

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]


@pytest.fixture
def simple_dataset():
    """Simple numeric dataset fixture"""
    data = [
        (np.array([1.0, 2.0, 3.0]), None),
        (np.array([4.0, 5.0, 6.0]), None),
        (np.array([7.0, 8.0, 9.0]), None),
    ]
    columns = ["col1", "col2", "col3"]
    return MockTabularDataset(data, columns), columns


@pytest.fixture
def mixed_dataset():
    """Dataset with mixed numeric types fixture"""
    data = [
        (np.array([1, 2.5, 3]), None),
        (np.array([2, 3.5, 4]), None),
        (np.array([3, 4.5, 5]), None),
    ]
    columns = ["int_col", "float_col", "mixed_col"]
    return MockTabularDataset(data, columns), columns


@pytest.fixture
def nan_dataset():
    """Dataset with NaN values fixture"""
    data = [
        (np.array([1.0, 2.0, np.nan]), None),
        (np.array([4.0, np.nan, 6.0]), None),
        (np.array([7.0, 8.0, 9.0]), None),
    ]
    columns = ["col1", "col2", "col3"]
    return MockTabularDataset(data, columns), columns


@pytest.fixture
def inf_dataset():
    """Dataset with infinity values fixture"""
    data = [
        (np.array([1.0, 2.0, np.inf]), None),
        (np.array([4.0, -np.inf, 6.0]), None),
        (np.array([7.0, 8.0, 9.0]), None),
    ]
    columns = ["col1", "col2", "col3"]
    return MockTabularDataset(data, columns), columns


@pytest.fixture
def single_dataset():
    """Single row dataset fixture"""
    data = [(np.array([5.0, 10.0, 15.0]), None)]
    columns = ["a", "b", "c"]
    return MockTabularDataset(data, columns), columns


@pytest.fixture
def negative_dataset():
    """Dataset with negative values fixture"""
    data = [
        (np.array([-1.0, -2.0, -3.0]), None),
        (np.array([1.0, 2.0, 3.0]), None),
        (np.array([-5.0, 0.0, 5.0]), None),
    ]
    columns = ["x", "y", "z"]
    return MockTabularDataset(data, columns), columns


@pytest.fixture
def zero_dataset():
    """Dataset with zeros fixture"""
    data = [
        (np.array([0.0, 0.0, 0.0]), None),
        (np.array([0.0, 0.0, 0.0]), None),
    ]
    columns = ["zero1", "zero2", "zero3"]
    return MockTabularDataset(data, columns), columns


# ==================== MEAN TESTS ====================


def test_validate_dataset_arguments_for_fa():
    # Valid arguments
    dataset_args = {"col_names": ["age", "sex"]}

    validate_dataset_arguments_for_fa(dataset_args, DatasetTypes.TABULAR)

    pytest.raises(
        FedbiomedError,
        validate_dataset_arguments_for_fa,
        {"invalid_arg": 123},
        DatasetTypes.TABULAR,
    )


def test_tabular_analytics_mean_simple(simple_dataset):
    """Test mean calculation on simple numeric dataset"""
    dataset, columns = simple_dataset
    result = dataset.mean()

    assert isinstance(result, dict)
    assert len(result) == 3
    assert result["col1"] == pytest.approx(4.0)
    assert result["col2"] == pytest.approx(5.0)
    assert result["col3"] == pytest.approx(6.0)


def test_tabular_analytics_mean_mixed_types(mixed_dataset):
    """Test mean calculation on dataset with mixed numeric types"""
    dataset, columns = mixed_dataset
    result = dataset.mean()

    assert isinstance(result, dict)
    assert result["int_col"] == pytest.approx(2.0)
    assert result["float_col"] == pytest.approx(3.5)
    assert result["mixed_col"] == pytest.approx(4.0)


def test_tabular_analytics_mean_with_nan(nan_dataset):
    """Test mean calculation with NaN values (should skip NaN)"""
    dataset, columns = nan_dataset
    result = dataset.mean()

    assert isinstance(result, dict)
    assert result["col1"] == pytest.approx(4.0)  # (1 + 4 + 7) / 3
    assert result["col2"] == pytest.approx(5.0)  # (2 + 8) / 2
    assert result["col3"] == pytest.approx(7.5)  # (6 + 9) / 2


def test_tabular_analytics_mean_with_inf(inf_dataset):
    """Test mean calculation with infinity values (should skip inf)"""
    dataset, columns = inf_dataset
    result = dataset.mean()

    assert isinstance(result, dict)
    assert result["col1"] == pytest.approx(4.0)  # (1 + 4 + 7) / 3
    assert result["col2"] == pytest.approx(5.0)  # (2 + 8) / 2
    assert result["col3"] == pytest.approx(7.5)  # (6 + 9) / 2


def test_tabular_analytics_mean_single_row(single_dataset):
    """Test mean calculation on single row dataset"""
    dataset, columns = single_dataset
    result = dataset.mean()

    assert isinstance(result, dict)
    assert result["a"] == pytest.approx(5.0)
    assert result["b"] == pytest.approx(10.0)
    assert result["c"] == pytest.approx(15.0)


def test_tabular_analytics_mean_negative_values(negative_dataset):
    """Test mean calculation with negative values"""
    dataset, columns = negative_dataset
    result = dataset.mean()

    assert isinstance(result, dict)
    assert result["x"] == pytest.approx(-5.0 / 3)  # (-1 + 1 + -5) / 3
    assert result["y"] == pytest.approx(0.0)  # (-2 + 2 + 0) / 3
    assert result["z"] == pytest.approx(5.0 / 3)  # (-3 + 3 + 5) / 3


def test_tabular_analytics_mean_zeros(zero_dataset):
    """Test mean calculation with all zeros"""
    dataset, columns = zero_dataset
    result = dataset.mean()

    assert isinstance(result, dict)
    assert result["zero1"] == pytest.approx(0.0)
    assert result["zero2"] == pytest.approx(0.0)
    assert result["zero3"] == pytest.approx(0.0)


# ==================== MAX TESTS ====================


def test_tabular_analytics_max_simple(simple_dataset):
    """Test max calculation on simple numeric dataset"""
    dataset, columns = simple_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert len(result) == 3
    assert result["col1"]["max"] == pytest.approx(7.0)
    assert result["col2"]["max"] == pytest.approx(8.0)
    assert result["col3"]["max"] == pytest.approx(9.0)


def test_tabular_analytics_max_mixed_types(mixed_dataset):
    """Test max calculation on dataset with mixed numeric types"""
    dataset, columns = mixed_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["int_col"]["max"] == pytest.approx(3.0)
    assert result["float_col"]["max"] == pytest.approx(4.5)
    assert result["mixed_col"]["max"] == pytest.approx(5.0)


def test_tabular_analytics_max_with_nan(nan_dataset):
    """Test max calculation with NaN values (should skip NaN)"""
    dataset, columns = nan_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["col1"]["max"] == pytest.approx(7.0)
    assert result["col2"]["max"] == pytest.approx(8.0)
    assert result["col3"]["max"] == pytest.approx(9.0)


def test_tabular_analytics_max_with_inf(inf_dataset):
    """Test max calculation with infinity values (should skip inf)"""
    dataset, columns = inf_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["col1"]["max"] == pytest.approx(7.0)
    assert result["col2"]["max"] == pytest.approx(8.0)
    assert result["col3"]["max"] == pytest.approx(9.0)


def test_tabular_analytics_max_single_row(single_dataset):
    """Test max calculation on single row dataset"""
    dataset, columns = single_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["a"]["max"] == pytest.approx(5.0)
    assert result["b"]["max"] == pytest.approx(10.0)
    assert result["c"]["max"] == pytest.approx(15.0)


def test_tabular_analytics_max_negative_values(negative_dataset):
    """Test max calculation with negative values"""
    dataset, columns = negative_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["x"]["max"] == pytest.approx(1.0)
    assert result["y"]["max"] == pytest.approx(2.0)
    assert result["z"]["max"] == pytest.approx(5.0)


def test_tabular_analytics_max_zeros(zero_dataset):
    """Test max calculation with all zeros"""
    dataset, columns = zero_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["zero1"]["max"] == pytest.approx(0.0)
    assert result["zero2"]["max"] == pytest.approx(0.0)
    assert result["zero3"]["max"] == pytest.approx(0.0)


# ==================== MIN TESTS ====================


def test_tabular_analytics_min_simple(simple_dataset):
    """Test min calculation on simple numeric dataset"""
    dataset, columns = simple_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert len(result) == 3
    assert result["col1"]["min"] == pytest.approx(1.0)
    assert result["col2"]["min"] == pytest.approx(2.0)
    assert result["col3"]["min"] == pytest.approx(3.0)


def test_tabular_analytics_min_mixed_types(mixed_dataset):
    """Test min calculation on dataset with mixed numeric types"""
    dataset, columns = mixed_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["int_col"]["min"] == pytest.approx(1.0)
    assert result["float_col"]["min"] == pytest.approx(2.5)
    assert result["mixed_col"]["min"] == pytest.approx(3.0)


def test_tabular_analytics_min_with_nan(nan_dataset):
    """Test min calculation with NaN values (should skip NaN)"""
    dataset, columns = nan_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["col1"]["min"] == pytest.approx(1.0)
    assert result["col2"]["min"] == pytest.approx(2.0)
    assert result["col3"]["min"] == pytest.approx(6.0)


def test_tabular_analytics_min_with_inf(inf_dataset):
    """Test min calculation with infinity values (should skip inf)"""
    dataset, columns = inf_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["col1"]["min"] == pytest.approx(1.0)
    assert result["col2"]["min"] == pytest.approx(2.0)
    assert result["col3"]["min"] == pytest.approx(6.0)


def test_tabular_analytics_min_single_row(single_dataset):
    """Test min calculation on single row dataset"""
    dataset, columns = single_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["a"]["min"] == pytest.approx(5.0)
    assert result["b"]["min"] == pytest.approx(10.0)
    assert result["c"]["min"] == pytest.approx(15.0)


def test_tabular_analytics_min_negative_values(negative_dataset):
    """Test min calculation with negative values"""
    dataset, columns = negative_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["x"]["min"] == pytest.approx(-5.0)
    assert result["y"]["min"] == pytest.approx(-2.0)
    assert result["z"]["min"] == pytest.approx(-3.0)


def test_tabular_analytics_min_zeros(zero_dataset):
    """Test min calculation with all zeros"""
    dataset, columns = zero_dataset
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["zero1"]["min"] == pytest.approx(0.0)
    assert result["zero2"]["min"] == pytest.approx(0.0)
    assert result["zero3"]["min"] == pytest.approx(0.0)


# ==================== STD TESTS ====================


def test_tabular_analytics_std_simple(simple_dataset):
    """Test standard deviation calculation on simple numeric dataset"""
    dataset, columns = simple_dataset
    result = dataset.basic_stats()

    assert isinstance(result, dict)
    assert len(result) == 3
    # Std is part of stats
    assert result["col1"]["std"] == pytest.approx(np.sqrt(6.0))
    assert result["col2"]["std"] == pytest.approx(np.sqrt(6.0))
    assert result["col3"]["std"] == pytest.approx(np.sqrt(6.0))


def test_tabular_analytics_std_mixed_types(mixed_dataset):
    """Test standard deviation calculation on dataset with mixed numeric types"""
    dataset, columns = mixed_dataset
    result = dataset.basic_stats()

    assert isinstance(result, dict)
    assert result["int_col"]["std"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert result["float_col"]["std"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert result["mixed_col"]["std"] == pytest.approx(np.sqrt(2.0 / 3.0))


def test_tabular_analytics_std_with_nan(nan_dataset):
    """Test standard deviation calculation with NaN values (should skip NaN)"""
    dataset, columns = nan_dataset
    result = dataset.basic_stats()

    assert isinstance(result, dict)
    assert result["col1"]["std"] == pytest.approx(np.sqrt(6.0))
    assert result["col2"]["std"] == pytest.approx(np.sqrt(9.0))
    assert result["col3"]["std"] == pytest.approx(np.sqrt(2.25))


def test_tabular_analytics_std_with_inf(inf_dataset):
    """Test standard deviation calculation with infinity values (should skip inf)"""
    dataset, columns = inf_dataset
    result = dataset.basic_stats()

    assert isinstance(result, dict)
    assert result["col1"]["std"] == pytest.approx(np.sqrt(6.0))
    assert result["col2"]["std"] == pytest.approx(np.sqrt(9.0))
    assert result["col3"]["std"] == pytest.approx(np.sqrt(2.25))


def test_tabular_analytics_std_single_row(single_dataset):
    """Test standard deviation calculation on single row dataset"""
    dataset, columns = single_dataset
    result = dataset.basic_stats()

    assert isinstance(result, dict)
    # Single value std should be 0
    assert result["a"]["std"] == pytest.approx(0.0)
    assert result["b"]["std"] == pytest.approx(0.0)
    assert result["c"]["std"] == pytest.approx(0.0)


def test_tabular_analytics_std_negative_values(negative_dataset):
    """Test standard deviation calculation with negative values"""
    dataset, columns = negative_dataset
    result = dataset.basic_stats()

    assert isinstance(result, dict)
    assert result["x"]["std"] == pytest.approx(np.sqrt(56.0 / 9.0), rel=1e-5)
    assert result["y"]["std"] == pytest.approx(np.sqrt(8.0 / 3.0), rel=1e-5)
    assert result["z"]["std"] == pytest.approx(np.sqrt(104.0 / 9.0), rel=1e-5)


def test_tabular_analytics_std_zeros(zero_dataset):
    """Test standard deviation calculation with all zeros"""
    dataset, columns = zero_dataset
    result = dataset.basic_stats()

    assert isinstance(result, dict)
    assert result["zero1"]["std"] == pytest.approx(0.0)
    assert result["zero2"]["std"] == pytest.approx(0.0)
    assert result["zero3"]["std"] == pytest.approx(0.0)


# ==================== EDGE CASES AND INTEGRATION ====================


def test_tabular_analytics_column_names_preserved(simple_dataset):
    """Test that all methods preserve column names correctly"""
    dataset, columns = simple_dataset
    methods = ["mean", "min_max", "basic_stats"]

    for method_name in methods:
        method = getattr(dataset, method_name)
        result = method()

        assert set(result.keys()) == set(columns)


def test_tabular_analytics_all_methods_return_dict(simple_dataset):
    """Test that all methods return dictionaries"""
    dataset, columns = simple_dataset
    methods = ["mean", "min_max", "basic_stats"]

    for method_name in methods:
        method = getattr(dataset, method_name)
        result = method()

        assert isinstance(result, dict)

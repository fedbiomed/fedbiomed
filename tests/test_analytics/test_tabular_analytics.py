# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from fedbiomed.common.analytics import (
    TabularAnalytics,
)
from fedbiomed.common.exceptions import FedbiomedError


class MockTabularDataset(TabularAnalytics):
    """Mock dataset class that implements TabularAnalytics for testing"""

    def __init__(self, data_list, input_columns):
        """
        Args:
            data_list: List of tuples (data, target) where data is np.ndarray or torch.Tensor
            input_columns: List of column names
        """
        self._data_list = data_list
        self._input_columns = input_columns

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx], None


@pytest.fixture
def dataset():
    """Simple numeric dataset with numpy arrays"""
    data = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    ]
    columns = ["col1", "col2", "col3"]
    return MockTabularDataset(data, columns)


# ==================== BASIC STATS TESTS ====================


def test_tabular_analytics_mean_simple(dataset):
    """Test mean calculation"""
    result = dataset.basic_stats()

    assert isinstance(result, dict)
    assert len(result) == 3
    # Result format: {col1: {mean: 4.0, ...}, col2: {mean: 5.0, ...}}
    assert result["col1"]["mean"] == pytest.approx(4.0)
    assert result["col2"]["mean"] == pytest.approx(5.0)
    assert result["col3"]["mean"] == pytest.approx(6.0)


def test_tabular_analytics_min_max_simple(dataset):
    """Test min_max calculation"""
    result = dataset.basic_stats({"min", "max"})

    assert isinstance(result, dict)
    assert result["col1"]["min"] == 1.0
    assert result["col1"]["max"] == 7.0
    assert result["col2"]["min"] == 2.0
    assert result["col2"]["max"] == 8.0
    assert result["col3"]["min"] == 3.0
    assert result["col3"]["max"] == 9.0


def test_tabular_analytics_std_variance(dataset):
    """Test standard deviation and variance calculation"""
    result = dataset.basic_stats({"std", "variance", "mean"})

    # Col 1: 1, 4, 7 -> Mean 4, Var 6, Std sqrt(6)
    assert result["col1"]["std"] == pytest.approx(np.sqrt(6.0))
    assert result["col1"]["variance"] == pytest.approx(6.0)

    # Col 2: 2, 5, 8 -> Mean 5, Var 6, Std sqrt(6)
    assert result["col2"]["std"] == pytest.approx(np.sqrt(6.0))
    assert result["col2"]["variance"] == pytest.approx(6.0)


def test_tabular_analytics_sum_count(dataset):
    """Test sum and count calculation"""
    result = dataset.basic_stats({"sum", "count"})

    assert result["col1"]["sum"] == 12.0
    assert result["col1"]["count"] == 3
    assert "mean" not in result["col1"]


def test_tabular_analytics_nan_values():
    """Test statistics with NaN and Infinite values"""
    data = [
        np.array([1.0, np.nan, 3.0]),
        np.array([4.0, 5.0, np.inf]),
        np.array([np.nan, 8.0, 9.0]),
        np.array([7.0, -np.inf, np.nan]),
    ]
    columns = ["col1", "col2", "col3"]
    dataset = MockTabularDataset(data, columns)

    result = dataset.basic_stats({"count", "min", "max", "mean", "sum"})

    # Col 1: 1, 4, 7 (NaN ignored). Count 3. Sum 12. Mean 4. Min 1. Max 7.
    assert result["col1"]["count"] == 3
    assert result["col1"]["sum"] == 12.0
    assert result["col1"]["mean"] == 4.0
    assert result["col1"]["min"] == 1.0
    assert result["col1"]["max"] == 7.0

    # Col 2: nan, 5, 8, -inf (ignored).
    # valid: 5.0, 8.0.
    assert result["col2"]["count"] == 2
    assert result["col2"]["sum"] == 13.0
    assert result["col2"]["mean"] == 6.5

    # Col 3: 3, inf, 9, nan.
    # valid: 3, 9.
    assert result["col3"]["count"] == 2
    assert result["col3"]["sum"] == 12.0
    assert result["col3"]["mean"] == 6.0


# ==================== HISTOGRAM TESTS ====================


def test_tabular_analytics_histogram_with_dict_bin_edges(dataset):
    """Test histogram with dict bin_edges"""
    columns = dataset._input_columns
    bin_edges = {
        "col1": np.array([0.5, 3.5, 6.5, 9.5]),
        "col2": np.array([1.5, 4.5, 7.5, 9.5]),
        "col3": np.array([2.5, 5.5, 8.5, 9.5]),
    }
    result = dataset.histogram(bin_edges)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(columns)
    for col in columns:
        assert isinstance(result[col], dict)
        assert "histogram" in result[col]
        assert "counts" in result[col]["histogram"]
        assert len(result[col]["histogram"]["counts"]) == 3  # 3 bins for each


def test_tabular_analytics_histogram_with_array_bin_edges(dataset):
    """Test histogram with array bin_edges applied to all columns on both datasets"""
    columns = dataset._input_columns
    bin_edges = np.array([0.5, 3.5, 6.5, 9.5])
    result = dataset.histogram(bin_edges)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(columns)
    for col in columns:
        assert isinstance(result[col], dict)
        assert "histogram" in result[col]
        assert "counts" in result[col]["histogram"]
        assert len(result[col]["histogram"]["counts"]) == 3  # 3 bins


def test_tabular_analytics_histogram_errors(dataset):
    """Test error conditions for histogram"""
    # 1. Invalid bin edges type/shape
    with pytest.raises(FedbiomedError, match="bin_edges must be a 1D array"):
        dataset.histogram(np.array(1.0))

    with pytest.raises(FedbiomedError, match="bin_edges must be a 1D array"):
        dataset.histogram(np.array([1.0]))

    # 2. Non-increasing edges
    with pytest.raises(FedbiomedError, match="bin_edges must be strictly increasing"):
        dataset.histogram(np.array([1.0, 1.0, 2.0]))

    with pytest.raises(FedbiomedError, match="bin_edges must be strictly increasing"):
        dataset.histogram(np.array([2.0, 1.0]))

    # 3. Missing column in bin_edges dict
    with pytest.raises(
        FedbiomedError, match="Column 'col1' .* not found in bin_edges dict"
    ):
        dataset.histogram({"col2": [1, 2, 3], "col3": [1, 2, 3]})


# ==================== QUANTILE TESTS ====================

'''
def test_tabular_analytics_quantile_single(dataset):
    """Test quantile with single quantile value"""
    columns = dataset._input_columns
    bin_edges = {
        "col1": np.array([0.5, 3.5, 6.5, 9.5]),
        "col2": np.array([1.5, 4.5, 7.5, 9.5]),
        "col3": np.array([2.5, 5.5, 8.5, 9.5]),
    }
    result = dataset.quantile(bin_edges, q=0.5)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(columns)
    for col in columns:
        assert isinstance(result[col], dict)
        assert 0.5 in result[col]


def test_tabular_analytics_quantile_multiple(dataset):
    """Test quantile with multiple quantile values on both numpy and torch datasets"""
    columns = dataset._input_columns
    bin_edges = {
        "col1": np.array([0.5, 3.5, 6.5, 9.5]),
        "col2": np.array([1.5, 4.5, 7.5, 9.5]),
        "col3": np.array([2.5, 5.5, 8.5, 9.5]),
    }
    result = dataset.quantile(bin_edges, q=[0.25, 0.5, 0.75])

    assert isinstance(result, dict)
    assert set(result.keys()) == set(columns)
    for col in columns:
        assert isinstance(result[col], dict)
        assert (
            0.25 in result[col]
            and bin_edges[col].min() <= result[col][0.25] <= bin_edges[col].max()
        )
        assert (
            0.5 in result[col]
            and bin_edges[col].min() <= result[col][0.5] <= bin_edges[col].max()
        )
        assert (
            0.75 in result[col]
            and bin_edges[col].min() <= result[col][0.75] <= bin_edges[col].max()
        )
'''


# ==================== ADDITIONAL TESTS ====================


def test_tabular_analytics_column_names_preserved(dataset):
    """Test that mean preserves column names correctly"""
    columns = dataset._input_columns
    result = dataset.basic_stats({"mean"})
    assert set(result.keys()) == set(columns)


def test_tabular_analytics_empty_dataset():
    """Test handling of empty dataset"""
    dataset = MockTabularDataset([], ["col1"])
    result = dataset.basic_stats()
    assert result == {}


def test_tabular_analytics_all_stats_no_args(dataset):
    """Test that default basic_stats returns expected set of statistics"""
    result = dataset.basic_stats()
    expected_stats = {"min", "max", "count", "mean", "std"}

    for col in ["col1", "col2", "col3"]:
        assert set(result[col].keys()) == expected_stats


def test_tabular_analytics_mixed_types_ignored():
    """Test that non-numeric types are ignored if present"""
    # Use object array to allow mixing types
    d1 = np.array([1.0, 2.0], dtype=object)
    d2 = np.array([3.0, "invalid"], dtype=object)

    dataset = MockTabularDataset([d1, d2], ["col1", "col2"])

    result = dataset.basic_stats({"count", "sum"})

    # Col 1: 1.0, 3.0 -> valid
    assert result["col1"]["count"] == 2
    assert result["col1"]["sum"] == 4.0

    # Col 2: 2.0, "invalid" -> "invalid" should be masked out
    assert result["col2"]["count"] == 1
    assert result["col2"]["sum"] == 2.0

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from fedbiomed.common.analytics import (
    TabularAnalytics,
)


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


# ==================== MEAN TESTS ====================


def test_tabular_analytics_mean_simple(dataset):
    """Test mean calculation"""
    result = dataset.mean()

    assert isinstance(result, dict)
    assert len(result) == 3
    assert result["col1"] == pytest.approx(4.0)
    assert result["col2"] == pytest.approx(5.0)
    assert result["col3"] == pytest.approx(6.0)


# ==================== MINMAX TESTS ====================


def test_tabular_analytics_min_max_simple(dataset):
    """Test min_max calculation"""
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert result["col1"] == {"min": 1.0, "max": 7.0}
    assert result["col2"] == {"min": 2.0, "max": 8.0}
    assert result["col3"] == {"min": 3.0, "max": 9.0}


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
        assert isinstance(result[col], np.ndarray)
        assert len(result[col]) == 3  # 3 bins for each


def test_tabular_analytics_histogram_with_array_bin_edges(dataset):
    """Test histogram with array bin_edges applied to all columns on both datasets"""
    columns = dataset._input_columns
    bin_edges = np.array([0.5, 3.5, 6.5, 9.5])
    result = dataset.histogram(bin_edges)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(columns)
    for col in columns:
        assert isinstance(result[col], np.ndarray)
        assert len(result[col]) == 3  # 3 bins


# ==================== QUANTILE TESTS ====================


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


# ==================== INTEGRATION TESTS ====================


def test_tabular_analytics_column_names_preserved_mean(dataset):
    """Test that mean preserves column names correctly on both numpy and torch datasets"""
    columns = dataset._input_columns
    result = dataset.mean()

    assert set(result.keys()) == set(columns)

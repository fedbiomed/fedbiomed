from unittest.mock import MagicMock

import numpy as np
import pytest

from fedbiomed.common.analytics.accumulators._row import RowAccumulator
from fedbiomed.common.constants import FedbiomedError


@pytest.fixture
def mock_registry(monkeypatch):
    """Mocks the AnalyticsRegistry to return a predictable specialized Accumulator class."""
    mock_acc_class = MagicMock()
    # Ensure instances of this class are also mocks with proper behavior
    mock_instance = MagicMock()
    mock_instance.update = MagicMock()
    # Mock finalize to return a dummy vector result
    mock_instance.finalize.return_value = np.array([10.0, 20.0])
    mock_acc_class.return_value = mock_instance

    # Mock registry lookup
    mock_get = MagicMock(return_value=mock_acc_class)
    monkeypatch.setattr(
        "fedbiomed.common.analytics.accumulators._registry.AnalyticsRegistry.get_accumulator_class",
        mock_get,
    )

    return mock_acc_class, mock_instance


@pytest.fixture
def basic_row_config():
    """Provides a valid standard configuration for RowAccumulator."""
    return {
        "columns": ["age", "salary"],
        "conf": {
            "age": {"mean": {}},  # Vectorizable (no args)
            "salary": {"mean": {}},  # Vectorizable (no args)
        },
    }


def test_row_accumulator_init_success(mock_registry, basic_row_config):
    """
    Test that RowAccumulator initializes correctly, groups vectorizable stats,
    and instantiates the underlying accumulator classes.
    """
    _, mock_acc_instance = mock_registry

    row_acc = RowAccumulator(basic_row_config)

    # Verify column mapping creation
    assert row_acc.col_map == {"age": 0, "salary": 1}

    # Verify vectorization grouping
    # Both 'age' (0) and 'salary' (1) use 'mean'
    assert "mean" in row_acc.vectorized_indices
    assert sorted(row_acc.vectorized_indices["mean"]) == [0, 1]

    # Verify underlying accumulator was instantiated once for the 'mean' group
    assert "mean" in row_acc.vectorized_accumulators
    assert row_acc.vectorized_accumulators["mean"] == mock_acc_instance

    # Verify independent accumulators are empty for this pure vectorizable config
    assert not row_acc.independent_accumulators


def test_row_accumulator_init_missing_columns(basic_row_config):
    """
    Test that RowAccumulator raises FedbiomedError if strict configuration validation fails.
    Scenario: 'height' is in 'columns' but missing from 'conf'.
    """
    basic_row_config["columns"].append("height")
    # 'height' not added to 'conf'

    with pytest.raises(FedbiomedError, match="must have corresponding entries"):
        RowAccumulator(basic_row_config)


def test_row_accumulator_update_logic(mock_registry, basic_row_config):
    """
    Test that data updates are correctly routed to the underlying vectorized aggregators.
    """
    _, mock_acc_instance = mock_registry
    row_acc = RowAccumulator(basic_row_config)

    # Update with a 2-element array (matching "age", "salary")
    data = np.array([30.0, 50000.0])
    row_acc.update(data)

    # Verify the mocked sub-accumulator received the data slice
    mock_acc_instance.update.assert_called_once()
    call_args = mock_acc_instance.update.call_args[0][0]

    # It should receive the values at indices [0, 1] corresponding to age and salary
    np.testing.assert_array_equal(call_args, data)


def test_row_accumulator_update_shape_mismatch(basic_row_config):
    """Test that updating with a non-1D array raises an error."""
    row_acc = RowAccumulator(basic_row_config)

    with pytest.raises(FedbiomedError, match="Expected 1D array"):
        row_acc.update(np.array([[1, 2]]))


def test_row_accumulator_finalize_mapping(mock_registry, basic_row_config):
    """
    Test that finalize correctly uppacks the vectorized results back to their
    original column names.
    """
    _, mock_acc_instance = mock_registry
    # Setup mock return: [val_for_idx_0, val_for_idx_1] -> [age_val, salary_val]
    mock_acc_instance.finalize.return_value = np.array([42.0, 999.0])

    row_acc = RowAccumulator(basic_row_config)
    results = row_acc.finalize()

    # 'age' is index 0 -> should map to 42.0
    assert results["age"]["mean"] == 42.0
    # 'salary' is index 1 -> should map to 999.0
    assert results["salary"]["mean"] == 999.0


def test_row_accumulator_finalize_complex_structure(mock_registry, basic_row_config):
    """
    Test unpacking when the underlying accumulator returns a dictionary (e.g., Variance).
    """
    _, mock_acc_instance = mock_registry
    # Simulate e.g. Variance returning multiple metrics per vector index
    mock_acc_instance.finalize.return_value = {
        "var": np.array([1.1, 2.2]),
        "mean": np.array([10, 20]),
    }

    row_acc = RowAccumulator(basic_row_config)
    results = row_acc.finalize()

    # Check unpacking of dictionary keys
    assert results["age"]["var"] == 1.1
    assert results["age"]["mean"] == 10
    assert results["salary"]["var"] == 2.2
    assert results["salary"]["mean"] == 20


def test_row_accumulator_partial_columns(mock_registry):
    """
    Test configuration where only a subset of columns have statistics.
    """
    config = {
        "columns": ["id", "age"],
        "conf": {
            "id": {},  # No stats for ID
            "age": {"mean": {}},  # specific stat for age
        },
    }
    row_acc = RowAccumulator(config)

    # Update with 2 values
    row_acc.update(np.array([123, 30.0]))

    # Check that the sub-accumulator only saw the data for 'age' (index 1)
    _, mock_acc_instance = mock_registry
    call_args = mock_acc_instance.update.call_args[0][0]
    np.testing.assert_array_equal(call_args, np.array([30.0]))

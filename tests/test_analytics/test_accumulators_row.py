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


def test_row_accumulator_init_missing_columns_key():
    """Test that missing 'columns' key raises FedbiomedError."""
    config = {"conf": {}}
    with pytest.raises(FedbiomedError, match="RowAccumulator requires 'columns'"):
        RowAccumulator(config)


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


def test_row_accumulator_independent_accumulators(mock_registry):
    """
    Test that accumulators with arguments fall back to independent instantiation.
    """
    mock_class, mock_instance = mock_registry

    # "percentile" with args -> should be treated as independent
    config = {"columns": ["salary"], "conf": {"salary": {"percentile": {"q": 90}}}}

    row_acc = RowAccumulator(config)

    # 1. Verify it's registered as independent
    col_idx = 0
    assert col_idx in row_acc.independent_accumulators
    stat_name, acc_obj = row_acc.independent_accumulators[col_idx][0]

    assert stat_name == "percentile"
    # Ensure it's the instance returned by our mock class
    # Since mock_class() returns mock_instance
    assert acc_obj == mock_instance

    # 2. Verify update calls the independent accumulator with a SCALAR
    row_acc.update(np.array([50000.0]))

    mock_instance.update.assert_called_once()
    # Independent generic accumulators usually expect scalar or single value updates depending on impl,
    # but RowAccumulator passes `value[col_idx]`.
    val_arg = mock_instance.update.call_args[0][0]
    assert val_arg == 50000.0

    # 3. Verify finalize collects result
    mock_instance.finalize.return_value = 12345.0
    results = row_acc.finalize()

    assert results["salary"]["percentile"] == 12345.0


def test_row_accumulator_mixed_vectorized_and_independent(mock_registry):
    """
    Test a scenario with both vectorized (no args) and independent (with args) stats.
    """
    mock_class, mock_instance_vec = mock_registry

    # We need to distinguish between the two accumulator instances (vec vs independent).
    # The current mock_registry fixture returns the same class/instance for all calls.
    # Let's adjust the mock behavior locally for this test or just inspect call counts.
    # A better approach: The class constructor is called twice.
    # Let's make the class return a NEW mock instance each time.

    mock_class.side_effect = [MagicMock(), MagicMock()]

    config = {
        "columns": ["age", "salary"],
        "conf": {
            "age": {"mean": {}},  # Vectorized
            "salary": {"percentile": {"q": 50}},  # Independent
        },
    }

    row_acc = RowAccumulator(config)

    # We expect:
    # 1. One vectorized accumulator for "mean" acting on column 0 ("age")
    # 2. One independent accumulator for "percentile" acting on column 1 ("salary")

    vec_acc_instance = row_acc.vectorized_accumulators["mean"]
    ind_acc_instance = row_acc.independent_accumulators[1][0][1]

    assert vec_acc_instance is not ind_acc_instance

    # Update
    row_acc.update(np.array([30.0, 60000.0]))

    # Check vectorized update: slice for 'age' (index 0)
    vec_acc_instance.update.assert_called_once()
    vec_call_arg = vec_acc_instance.update.call_args[0][0]
    np.testing.assert_array_equal(vec_call_arg, np.array([30.0]))

    # Check independent update: scalar for 'salary' (index 1)
    ind_acc_instance.update.assert_called_once_with(60000.0)

    # Check finalize
    vec_acc_instance.finalize.return_value = np.array(
        [30.5]
    )  # Result for the 1 column in this group
    ind_acc_instance.finalize.return_value = 60000.0

    results = row_acc.finalize()

    assert results["age"]["mean"] == 30.5
    assert results["salary"]["percentile"] == 60000.0

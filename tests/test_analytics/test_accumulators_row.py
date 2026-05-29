from unittest.mock import MagicMock

import numpy as np
import pytest

from fedbiomed.common.analytics.accumulators._row import RowAccumulator
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def mock_registry(monkeypatch):
    mock_acc_class = MagicMock()
    mock_instance = MagicMock()
    mock_instance.finalize.return_value = {
        "sum": [10.0, 20.0],
        "count": [2.0, 2.0],
    }
    mock_acc_class.return_value = mock_instance

    monkeypatch.setattr(
        "fedbiomed.common.analytics.accumulators._registry.AnalyticsRegistry.get_accumulators",
        MagicMock(return_value=[mock_acc_class]),
    )

    return mock_acc_class, mock_instance


@pytest.fixture
def basic_row_config():
    return {
        "schema_columns": ["age", "salary"],
        "conf": {
            "age": {"mean": {}},
            "salary": {"mean": {}},
        },
    }


# --- init validation ---


def test_row_accumulator_init_missing_schema_columns():
    with pytest.raises(FedbiomedError, match="requires 'schema_columns'"):
        RowAccumulator({"conf": {"age": {"mean": {}}}})


def test_row_accumulator_init_empty_schema_columns():
    with pytest.raises(FedbiomedError, match="requires 'schema_columns'"):
        RowAccumulator({"schema_columns": [], "conf": {"age": {"mean": {}}}})


def test_row_accumulator_init_empty_conf():
    with pytest.raises(FedbiomedError, match="non-empty 'conf'"):
        RowAccumulator({"schema_columns": ["age"], "conf": {}})


def test_row_accumulator_init_unknown_column():
    with pytest.raises(FedbiomedError, match="All columns in 'conf' must be present"):
        RowAccumulator({"schema_columns": ["age"], "conf": {"unknown": {"mean": {}}}})


# --- update ---


def test_row_accumulator_update_logic(mock_registry, basic_row_config):
    _, mock_instance = mock_registry
    row_acc = RowAccumulator(basic_row_config)

    data = np.array([30.0, 50000.0])
    row_acc.update(data)

    mock_instance.update.assert_called_once()
    np.testing.assert_array_equal(mock_instance.update.call_args[0][0], data)


def test_row_accumulator_update_wrong_ndim(mock_registry, basic_row_config):
    row_acc = RowAccumulator(basic_row_config)
    with pytest.raises(FedbiomedError, match="Expected 1D array"):
        row_acc.update(np.array([[1, 2]]))


def test_row_accumulator_update_too_few_elements(mock_registry, basic_row_config):
    """Array shorter than schema_columns raises a length-mismatch error."""
    row_acc = RowAccumulator(basic_row_config)
    with pytest.raises(FedbiomedError, match="Expected 2 elements, got 1"):
        row_acc.update(np.array([1.0]))


def test_row_accumulator_update_too_many_elements(mock_registry, basic_row_config):
    """Array longer than schema_columns raises a length-mismatch error."""
    row_acc = RowAccumulator(basic_row_config)
    with pytest.raises(FedbiomedError, match="Expected 2 elements, got 3"):
        row_acc.update(np.array([1.0, 2.0, 3.0]))


# --- finalize ---


def test_row_accumulator_finalize_mapping(mock_registry, basic_row_config):
    _, mock_instance = mock_registry
    mock_instance.finalize.return_value = {
        "sum": [42.0, 999.0],
        "count": [1.0, 1.0],
    }

    results = RowAccumulator(basic_row_config).finalize()

    assert results["age"]["sum"] == 42.0
    assert results["salary"]["sum"] == 999.0


def test_row_accumulator_finalize_multi_key_output(mock_registry, basic_row_config):
    """All keys from a multi-output accumulator (mean → sum+count) are extracted per column."""
    _, mock_instance = mock_registry
    mock_instance.finalize.return_value = {
        "sum": [100.0, 500.0],
        "count": [2.0, 5.0],
    }

    results = RowAccumulator(basic_row_config).finalize()

    assert results["age"]["sum"] == 100.0
    assert results["age"]["count"] == 2.0
    assert results["salary"]["sum"] == 500.0
    assert results["salary"]["count"] == 5.0


# --- structural / mixed ---


def test_row_accumulator_uncovered_column_excluded(mock_registry):
    """A schema column absent from conf does not participate in any accumulator update."""
    _, mock_instance = mock_registry
    config = {
        "schema_columns": ["id", "age"],
        "conf": {"age": {"mean": {}}},
    }
    row_acc = RowAccumulator(config)
    row_acc.update(np.array([123.0, 30.0]))

    # The vectorized slice for 'age' (index 1) is [30.0]; 'id' (index 0) is never passed.
    np.testing.assert_array_equal(
        mock_instance.update.call_args[0][0], np.array([30.0])
    )


def test_row_accumulator_independent_accumulators(mock_registry):
    """Stats with args are instantiated as independent (per-column) accumulators."""
    mock_class, mock_instance = mock_registry
    config = {
        "schema_columns": ["salary"],
        "conf": {"salary": {"percentile": {"q": 90}}},
    }
    row_acc = RowAccumulator(config)

    stat_name, acc_obj = row_acc.independent_accumulators[0][0]
    assert stat_name == "percentile"
    assert acc_obj == mock_instance

    row_acc.update(np.array([50000.0]))
    assert mock_instance.update.call_args[0][0] == 50000.0

    mock_instance.finalize.return_value = 12345.0
    assert row_acc.finalize()["salary"]["percentile"] == 12345.0


def test_row_accumulator_mixed_vectorized_and_independent(mock_registry):
    """Vectorized and independent accumulators coexist and receive correct data slices."""
    mock_class, _ = mock_registry
    mock_class.side_effect = [MagicMock(), MagicMock()]

    config = {
        "schema_columns": ["age", "salary"],
        "conf": {
            "age": {"mean": {}},
            "salary": {"percentile": {"q": 50}},
        },
    }
    row_acc = RowAccumulator(config)
    (vec_instance,) = row_acc.vectorized_accumulators["mean"]
    ind_instance = row_acc.independent_accumulators[1][0][1]

    row_acc.update(np.array([30.0, 60000.0]))

    np.testing.assert_array_equal(vec_instance.update.call_args[0][0], np.array([30.0]))
    ind_instance.update.assert_called_once_with(60000.0)

    vec_instance.finalize.return_value = {
        "sum": [30.5],
        "count": [1.0],
    }
    ind_instance.finalize.return_value = 60000.0
    results = row_acc.finalize()

    assert results["age"]["sum"] == 30.5
    assert results["salary"]["percentile"] == 60000.0

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Tests for scalar and 1D accumulator classes."""

import numpy as np
import pytest

from fedbiomed.common.analytics.accumulators._scalar_1d import (
    CountAccumulator,
    MaxAccumulator,
    MeanAccumulator,
    MinAccumulator,
    ScalarBuffer,
    VarianceAccumulator,
)
from fedbiomed.common.constants import FedbiomedError

# =============================================================================
# ScalarBuffer Tests
# =============================================================================


def test_scalar_buffer_init_valid_length():
    """Test initialization with valid length."""
    buffer = ScalarBuffer(10)
    assert len(buffer) == 10
    assert buffer.buffer.dtype == np.float32
    assert np.all(np.isnan(buffer.buffer))


def test_scalar_buffer_init_with_stat_functions():
    """Test initialization with stat functions."""
    stats = {"mean": np.mean, "std": np.std}
    buffer = ScalarBuffer(5, stat_functions=stats)
    assert buffer.stat_functions == stats


@pytest.mark.parametrize("invalid_length", [0, -1, -10, 0.5, "10"])
def test_scalar_buffer_init_invalid_length(invalid_length):
    """Test initialization with invalid length."""
    with pytest.raises(FedbiomedError, match="must be positive integer"):
        ScalarBuffer(invalid_length)


def test_scalar_buffer_len():
    """Test __len__ method."""
    buffer = ScalarBuffer(7)
    assert len(buffer) == 7


def test_scalar_buffer_update_sequential():
    """Test sequential updates."""
    buffer = ScalarBuffer(3)
    buffer.update(1.0)
    buffer.update(2.0)
    buffer.update(3.0)

    assert buffer._next_index == 3
    np.testing.assert_array_equal(
        buffer.buffer, np.array([1.0, 2.0, 3.0], dtype=np.float32)
    )


def test_scalar_buffer_update_buffer_full():
    """Test update when buffer is full."""
    buffer = ScalarBuffer(2)
    buffer.update(1.0)
    buffer.update(2.0)

    with pytest.raises(FedbiomedError, match="Buffer full"):
        buffer.update(3.0)


def test_scalar_buffer_set_stat_functions_valid():
    """Test setting valid stat functions."""
    buffer = ScalarBuffer(5)
    stats = {"mean": np.mean, "median": np.median}
    buffer.set_stat_functions(stats)

    assert "mean" in buffer.stat_functions
    assert "median" in buffer.stat_functions


def test_scalar_buffer_set_stat_functions_invalid_type():
    """Test setting stat functions with invalid type."""
    buffer = ScalarBuffer(5)

    with pytest.raises(FedbiomedError, match="must be dict"):
        buffer.set_stat_functions("not a dict")


def test_scalar_buffer_set_stat_functions_non_callable():
    """Test setting stat functions with non-callable value."""
    buffer = ScalarBuffer(5)

    with pytest.raises(FedbiomedError, match="not callable"):
        buffer.set_stat_functions({"mean": 42})


def test_scalar_buffer_finalize_empty_buffer():
    """Test finalize with empty buffer."""
    buffer = ScalarBuffer(5, stat_functions={"mean": np.mean})
    result = buffer.finalize()

    assert "mean" in result
    assert np.isnan(result["mean"])


def test_scalar_buffer_finalize_with_data():
    """Test finalize with data."""
    buffer = ScalarBuffer(5, stat_functions={"mean": np.mean, "std": np.std})
    for val in [1.0, 2.0, 3.0, 4.0, 5.0]:
        buffer.update(val)

    result = buffer.finalize()

    assert "mean" in result
    assert "std" in result
    assert result["mean"] == pytest.approx(3.0)
    assert result["std"] == pytest.approx(np.std([1, 2, 3, 4, 5]))


def test_scalar_buffer_finalize_partial_buffer():
    """Test finalize with partially filled buffer."""
    buffer = ScalarBuffer(10, stat_functions={"mean": np.mean})
    buffer.update(10.0)
    buffer.update(20.0)

    result = buffer.finalize()
    assert result["mean"] == pytest.approx(15.0)


def test_scalar_buffer_finalize_filters_non_finite():
    """Test finalize filters out non-finite values."""
    buffer = ScalarBuffer(5, stat_functions={"mean": np.mean})
    buffer.buffer[:] = [1.0, np.nan, 3.0, np.inf, 5.0]

    result = buffer.finalize()
    # Should only include 1.0, 3.0, 5.0
    assert result["mean"] == pytest.approx(3.0)


def test_scalar_buffer_finalize_error_handling():
    """Test finalize error handling when stat function fails."""

    def failing_func(data):
        raise ValueError("Test error")

    buffer = ScalarBuffer(3, stat_functions={"bad": failing_func})
    buffer.update(1.0)

    with pytest.raises(FedbiomedError, match="Error computing stat 'bad'"):
        buffer.finalize()


# =============================================================================
# BaseStatAccumulator Tests
# =============================================================================


def test_base_stat_accumulator_validate_shape_first_update():
    """Test shape validation on first update."""
    acc = CountAccumulator()  # Use concrete class
    val = np.array([1.0, 2.0, 3.0])

    acc._validate_shape(val)
    assert acc._shape == (3,)


def test_base_stat_accumulator_validate_shape_consistent():
    """Test shape validation with consistent shapes."""
    acc = CountAccumulator()
    val1 = np.array([1.0, 2.0])
    val2 = np.array([3.0, 4.0])

    acc._validate_shape(val1)
    acc._validate_shape(val2)  # Should not raise


def test_base_stat_accumulator_validate_shape_mismatch():
    """Test shape validation with mismatched shapes."""
    acc = CountAccumulator()
    val1 = np.array([1.0, 2.0])
    val2 = np.array([1.0, 2.0, 3.0])

    acc._validate_shape(val1)
    with pytest.raises(FedbiomedError, match="Shape mismatch"):
        acc._validate_shape(val2)


# =============================================================================
# CountAccumulator Tests
# =============================================================================


def test_count_accumulator_init():
    """Test initialization."""
    acc = CountAccumulator()
    assert acc.counts is None


def test_count_accumulator_update_single_value():
    """Test update with single value."""
    acc = CountAccumulator()
    acc.update(np.array([5.0]))

    np.testing.assert_array_equal(acc.counts, np.array([1], dtype=np.int32))


def test_count_accumulator_update_multiple_values():
    """Test update with multiple values."""
    acc = CountAccumulator()
    acc.update(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_array_equal(acc.counts, np.array([1, 1, 1], dtype=np.int32))


def test_count_accumulator_update_with_nan():
    """Test update with NaN values."""
    acc = CountAccumulator()
    acc.update(np.array([1.0, np.nan, 3.0]))

    # NaN should not be counted
    np.testing.assert_array_equal(acc.counts, np.array([1, 0, 1], dtype=np.int32))


def test_count_accumulator_update_with_inf():
    """Test update with inf values."""
    acc = CountAccumulator()
    acc.update(np.array([1.0, np.inf, -np.inf]))

    # inf should not be counted
    np.testing.assert_array_equal(acc.counts, np.array([1, 0, 0], dtype=np.int32))


def test_count_accumulator_multiple_updates():
    """Test multiple updates accumulate correctly."""
    acc = CountAccumulator()
    acc.update(np.array([1.0, np.nan, 3.0]))
    acc.update(np.array([4.0, 5.0, np.nan]))

    np.testing.assert_array_equal(acc.counts, np.array([2, 1, 1], dtype=np.int32))


def test_count_accumulator_finalize():
    """Test finalize returns correct format."""
    acc = CountAccumulator()
    acc.update(np.array([1.0, 2.0]))

    result = acc.finalize()
    assert isinstance(result, dict)
    assert "count" in result
    np.testing.assert_array_equal(result["count"], np.array([1, 1], dtype=np.int32))


def test_count_accumulator_finalize_no_data():
    """Test finalize with no updates."""
    acc = CountAccumulator()
    result = acc.finalize()

    assert result["count"] == 0


# =============================================================================
# MinAccumulator Tests
# =============================================================================


def test_min_accumulator_init():
    """Test initialization."""
    acc = MinAccumulator()
    assert acc.min_val is None


def test_min_accumulator_update_first_values():
    """Test first update initializes correctly."""
    acc = MinAccumulator()
    acc.update(np.array([3.0, 1.0, 5.0]))

    np.testing.assert_array_almost_equal(
        acc.min_val, np.array([3.0, 1.0, 5.0], dtype=np.float32)
    )


def test_min_accumulator_update_with_nan_first():
    """Test first update with NaN replaces with inf."""
    acc = MinAccumulator()
    acc.update(np.array([1.0, np.nan, 3.0]))

    expected = np.array([1.0, np.inf, 3.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(acc.min_val, expected)


def test_min_accumulator_update_with_inf_first():
    """Test first update with inf replaces with inf."""
    acc = MinAccumulator()
    acc.update(np.array([1.0, np.inf, -np.inf]))

    expected = np.array([1.0, np.inf, np.inf], dtype=np.float32)
    np.testing.assert_array_almost_equal(acc.min_val, expected)


def test_min_accumulator_multiple_updates_finds_min():
    """Test multiple updates find minimum."""
    acc = MinAccumulator()
    acc.update(np.array([5.0, 3.0, 7.0]))
    acc.update(np.array([2.0, 4.0, 1.0]))

    np.testing.assert_array_almost_equal(
        acc.min_val, np.array([2.0, 3.0, 1.0], dtype=np.float32)
    )


def test_min_accumulator_update_nan_ignored_in_subsequent():
    """Test NaN is ignored in subsequent updates using fmin."""
    acc = MinAccumulator()
    acc.update(np.array([5.0, 3.0, 7.0]))
    acc.update(np.array([np.nan, 2.0, np.nan]))

    # fmin(5, nan) = 5, fmin(3, 2) = 2, fmin(7, nan) = 7
    np.testing.assert_array_almost_equal(
        acc.min_val, np.array([5.0, 2.0, 7.0], dtype=np.float32)
    )


def test_min_accumulator_finalize():
    """Test finalize returns correct format."""
    acc = MinAccumulator()
    acc.update(np.array([2.0, 3.0]))

    result = acc.finalize()
    assert isinstance(result, dict)
    assert "min" in result
    np.testing.assert_array_almost_equal(
        result["min"], np.array([2.0, 3.0], dtype=np.float32)
    )


# =============================================================================
# MaxAccumulator Tests
# =============================================================================


def test_max_accumulator_init():
    """Test initialization."""
    acc = MaxAccumulator()
    assert acc.max_val is None


def test_max_accumulator_update_first_values():
    """Test first update initializes correctly."""
    acc = MaxAccumulator()
    acc.update(np.array([3.0, 1.0, 5.0]))

    np.testing.assert_array_almost_equal(
        acc.max_val, np.array([3.0, 1.0, 5.0], dtype=np.float32)
    )


def test_max_accumulator_update_with_nan_first():
    """Test first update with NaN replaces with -inf."""
    acc = MaxAccumulator()
    acc.update(np.array([1.0, np.nan, 3.0]))

    expected = np.array([1.0, -np.inf, 3.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(acc.max_val, expected)


def test_max_accumulator_update_with_inf_first():
    """Test first update with inf replaces with -inf."""
    acc = MaxAccumulator()
    acc.update(np.array([1.0, np.inf, -np.inf]))

    expected = np.array([1.0, -np.inf, -np.inf], dtype=np.float32)
    np.testing.assert_array_almost_equal(acc.max_val, expected)


def test_max_accumulator_multiple_updates_finds_max():
    """Test multiple updates find maximum."""
    acc = MaxAccumulator()
    acc.update(np.array([5.0, 3.0, 7.0]))
    acc.update(np.array([2.0, 4.0, 1.0]))

    np.testing.assert_array_almost_equal(
        acc.max_val, np.array([5.0, 4.0, 7.0], dtype=np.float32)
    )


def test_max_accumulator_update_nan_ignored_in_subsequent():
    """Test NaN is ignored in subsequent updates using fmax."""
    acc = MaxAccumulator()
    acc.update(np.array([5.0, 3.0, 7.0]))
    acc.update(np.array([np.nan, 10.0, np.nan]))

    # fmax(5, nan) = 5, fmax(3, 10) = 10, fmax(7, nan) = 7
    np.testing.assert_array_almost_equal(
        acc.max_val, np.array([5.0, 10.0, 7.0], dtype=np.float32)
    )


def test_max_accumulator_finalize():
    """Test finalize returns correct format."""
    acc = MaxAccumulator()
    acc.update(np.array([2.0, 3.0]))

    result = acc.finalize()
    assert isinstance(result, dict)
    assert "max" in result
    np.testing.assert_array_almost_equal(
        result["max"], np.array([2.0, 3.0], dtype=np.float32)
    )


# =============================================================================
# MeanAccumulator Tests
# =============================================================================


def test_mean_accumulator_init():
    """Test initialization."""
    acc = MeanAccumulator()
    assert acc.sum_val is None
    assert acc.counts is None


def test_mean_accumulator_update_basic():
    """Test basic update."""
    acc = MeanAccumulator()
    acc.update(np.array([2.0, 4.0, 6.0]))

    np.testing.assert_array_almost_equal(
        acc.sum_val, np.array([2.0, 4.0, 6.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(acc.counts, np.array([1, 1, 1], dtype=np.int32))


def test_mean_accumulator_update_with_nan():
    """Test update with NaN values."""
    acc = MeanAccumulator()
    acc.update(np.array([2.0, np.nan, 6.0]))

    # NaN replaced with 0 in sum
    np.testing.assert_array_almost_equal(
        acc.sum_val, np.array([2.0, 0.0, 6.0], dtype=np.float32)
    )
    # NaN not counted
    np.testing.assert_array_equal(acc.counts, np.array([1, 0, 1], dtype=np.int32))


def test_mean_accumulator_update_with_inf():
    """Test update with inf values."""
    acc = MeanAccumulator()
    acc.update(np.array([2.0, np.inf, -np.inf]))

    # inf replaced with 0 in sum
    np.testing.assert_array_almost_equal(
        acc.sum_val, np.array([2.0, 0.0, 0.0], dtype=np.float32)
    )
    # inf not counted
    np.testing.assert_array_equal(acc.counts, np.array([1, 0, 0], dtype=np.int32))


def test_mean_accumulator_multiple_updates():
    """Test multiple updates accumulate correctly."""
    acc = MeanAccumulator()
    acc.update(np.array([1.0, 2.0, 3.0]))
    acc.update(np.array([4.0, 5.0, 6.0]))

    np.testing.assert_array_almost_equal(
        acc.sum_val, np.array([5.0, 7.0, 9.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(acc.counts, np.array([2, 2, 2], dtype=np.int32))


def test_mean_accumulator_finalize_basic():
    """Test finalize computes mean correctly."""
    acc = MeanAccumulator()
    acc.update(np.array([2.0, 4.0, 6.0]))
    acc.update(np.array([4.0, 2.0, 0.0]))

    result = acc.finalize()

    assert "mean" in result
    assert "count" in result
    np.testing.assert_array_almost_equal(result["mean"], np.array([3.0, 3.0, 3.0]))
    np.testing.assert_array_equal(result["count"], np.array([2, 2, 2], dtype=np.int32))


def test_mean_accumulator_finalize_with_zero_count():
    """Test finalize with zero count returns NaN."""
    acc = MeanAccumulator()
    acc.update(np.array([1.0, np.nan, 3.0]))

    result = acc.finalize()

    # Second element has count=0, should be NaN
    assert np.isfinite(result["mean"][0])
    assert np.isnan(result["mean"][1])
    assert np.isfinite(result["mean"][2])


def test_mean_accumulator_finalize_no_data():
    """Test finalize with no updates."""
    acc = MeanAccumulator()
    result = acc.finalize()

    assert np.isnan(result["mean"])
    assert result["count"] == 0


# =============================================================================
# VarianceAccumulator Tests
# =============================================================================


def test_variance_accumulator_init():
    """Test initialization."""
    acc = VarianceAccumulator()
    assert acc.mean_val is None
    assert acc.m2_val is None
    assert acc.counts is None


def test_variance_accumulator_update_first():
    """Test first update initializes correctly."""
    acc = VarianceAccumulator()
    acc.update(np.array([2.0, 4.0, 6.0]))

    np.testing.assert_array_almost_equal(
        acc.mean_val, np.array([2.0, 4.0, 6.0], dtype=np.float32)
    )
    np.testing.assert_array_almost_equal(
        acc.m2_val, np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(acc.counts, np.array([1, 1, 1], dtype=np.int32))


def test_variance_accumulator_update_with_nan_first():
    """Test first update with NaN."""
    acc = VarianceAccumulator()
    acc.update(np.array([2.0, np.nan, 6.0]))

    np.testing.assert_array_almost_equal(
        acc.mean_val, np.array([2.0, 0.0, 6.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(acc.counts, np.array([1, 0, 1], dtype=np.int32))


def test_variance_accumulator_multiple_updates_welfords():
    """Test Welford's algorithm with multiple updates."""
    acc = VarianceAccumulator()
    acc.update(np.array([1.0]))
    acc.update(np.array([2.0]))
    acc.update(np.array([3.0]))

    # Mean should be 2.0
    assert acc.mean_val[0] == pytest.approx(2.0)
    # Variance should be 1.0 (sample variance)
    result = acc.finalize()
    assert result["variance"][0] == pytest.approx(1.0)


def test_variance_accumulator_finalize_basic():
    """Test finalize computes variance correctly."""
    acc = VarianceAccumulator()
    values = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

    for val in values:
        acc.update(np.array([val]))

    result = acc.finalize()

    # Expected variance (sample)
    expected_var = np.var(values, ddof=1)
    assert result["variance"][0] == pytest.approx(expected_var, rel=1e-5)
    assert result["mean"][0] == pytest.approx(np.mean(values))
    assert result["count"][0] == len(values)


def test_variance_accumulator_finalize_single_value():
    """Test finalize with single value returns NaN variance."""
    acc = VarianceAccumulator()
    acc.update(np.array([5.0]))

    result = acc.finalize()

    # With n=1, variance is undefined (returns NaN)
    assert np.isnan(result["variance"][0])
    assert result["mean"][0] == pytest.approx(5.0)
    assert result["count"][0] == 1


def test_variance_accumulator_finalize_vector():
    """Test finalize with vector data."""
    acc = VarianceAccumulator()
    acc.update(np.array([1.0, 10.0, 100.0]))
    acc.update(np.array([2.0, 20.0, 200.0]))
    acc.update(np.array([3.0, 30.0, 300.0]))

    result = acc.finalize()

    # Check each element independently
    assert result["variance"][0] == pytest.approx(1.0)  # var([1,2,3])
    assert result["variance"][1] == pytest.approx(100.0)  # var([10,20,30])
    assert result["variance"][2] == pytest.approx(10000.0)  # var([100,200,300])


def test_variance_accumulator_finalize_no_data():
    """Test finalize with no updates."""
    acc = VarianceAccumulator()
    result = acc.finalize()

    assert np.isnan(result["variance"])
    assert np.isnan(result["mean"])
    assert result["count"] == 0


def test_variance_accumulator_finalize_returns_dict():
    """Test finalize returns correct dictionary format."""
    acc = VarianceAccumulator()
    acc.update(np.array([1.0, 2.0]))
    acc.update(np.array([3.0, 4.0]))

    result = acc.finalize()

    assert isinstance(result, dict)
    assert set(result.keys()) == {"variance", "mean", "count"}


# =============================================================================
# Integration Tests
# =============================================================================


def test_integration_all_accumulators_same_data():
    """Test all accumulators on same dataset for consistency."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Initialize all accumulators
    count_acc = CountAccumulator()
    min_acc = MinAccumulator()
    max_acc = MaxAccumulator()
    mean_acc = MeanAccumulator()
    var_acc = VarianceAccumulator()

    # Feed same data
    for val in data:
        count_acc.update(np.array([val]))
        min_acc.update(np.array([val]))
        max_acc.update(np.array([val]))
        mean_acc.update(np.array([val]))
        var_acc.update(np.array([val]))

    # Get results
    count_result = count_acc.finalize()
    min_result = min_acc.finalize()
    max_result = max_acc.finalize()
    mean_result = mean_acc.finalize()
    var_result = var_acc.finalize()

    # Verify consistency
    assert count_result["count"][0] == 5
    assert min_result["min"][0] == pytest.approx(1.0)
    assert max_result["max"][0] == pytest.approx(5.0)
    assert mean_result["mean"][0] == pytest.approx(3.0)
    assert mean_result["count"][0] == 5
    assert var_result["mean"][0] == pytest.approx(3.0)
    assert var_result["variance"][0] == pytest.approx(np.var(data, ddof=1))
    assert var_result["count"][0] == 5


def test_integration_vectorized_multi_column():
    """Test vectorized operations across multiple columns."""
    # Simulate 3 columns with 5 samples each
    col1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    col2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    col3 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    mean_acc = MeanAccumulator()

    # Process row by row (vectorized across columns)
    for i in range(len(col1)):
        row = np.array([col1[i], col2[i], col3[i]])
        mean_acc.update(row)

    result = mean_acc.finalize()

    # Verify each column mean
    assert result["mean"][0] == pytest.approx(np.mean(col1))
    assert result["mean"][1] == pytest.approx(np.mean(col2))
    assert result["mean"][2] == pytest.approx(np.mean(col3))

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Tests for scalar and 1D accumulator classes."""

import numpy as np
import pytest

from fedbiomed.common.analytics.accumulators._operations import (
    CountAccumulator,
    HistogramAccumulator,
    SumAccumulator,
    SumSqAccumulator,
)
from fedbiomed.common.constants import FedbiomedError

# =============================================================================
# BaseStatAccumulator
# =============================================================================


def test_base_stat_accumulator_validate_shape_first_update():
    acc = CountAccumulator()
    acc._validate_shape(np.array([1.0, 2.0, 3.0]))
    assert acc._shape == (3,)


def test_base_stat_accumulator_validate_shape_consistent():
    acc = CountAccumulator()
    acc._validate_shape(np.array([1.0, 2.0]))
    acc._validate_shape(np.array([3.0, 4.0]))  # should not raise


def test_base_stat_accumulator_validate_shape_mismatch():
    acc = CountAccumulator()
    acc._validate_shape(np.array([1.0, 2.0]))
    with pytest.raises(FedbiomedError, match="Shape mismatch"):
        acc._validate_shape(np.array([1.0, 2.0, 3.0]))


# =============================================================================
# CountAccumulator
# =============================================================================


def test_count_accumulator_init():
    assert CountAccumulator()._value is None


def test_count_accumulator_update_single_value():
    acc = CountAccumulator()
    acc.update(np.array([5.0]))
    np.testing.assert_array_equal(acc._value, np.array([1], dtype=np.int32))


def test_count_accumulator_update_multiple_values():
    acc = CountAccumulator()
    acc.update(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(acc._value, np.array([1, 1, 1], dtype=np.int32))


def test_count_accumulator_update_with_nan():
    acc = CountAccumulator()
    acc.update(np.array([1.0, np.nan, 3.0]))
    np.testing.assert_array_equal(acc._value, np.array([1, 0, 1], dtype=np.int32))


def test_count_accumulator_update_with_inf():
    acc = CountAccumulator()
    acc.update(np.array([1.0, np.inf, -np.inf]))
    np.testing.assert_array_equal(acc._value, np.array([1, 0, 0], dtype=np.int32))


def test_count_accumulator_multiple_updates():
    acc = CountAccumulator()
    acc.update(np.array([1.0, np.nan, 3.0]))
    acc.update(np.array([4.0, 5.0, np.nan]))
    np.testing.assert_array_equal(acc._value, np.array([2, 1, 1], dtype=np.int32))


def test_count_accumulator_finalize():
    acc = CountAccumulator()
    acc.update(np.array([1.0, 2.0]))
    result = acc.finalize()
    assert isinstance(result, dict)
    np.testing.assert_array_equal(result["count"], np.array([1, 1], dtype=np.int32))


def test_count_accumulator_finalize_no_data():
    assert CountAccumulator().finalize()["count"] == 0


# =============================================================================
# SumAccumulator
# =============================================================================


def test_sum_accumulator_init():
    acc = SumAccumulator()
    assert acc._value is None


def test_sum_accumulator_update_basic():
    acc = SumAccumulator()
    acc.update(np.array([2.0, 4.0, 6.0]))
    np.testing.assert_array_almost_equal(acc._value, [2.0, 4.0, 6.0])


def test_sum_accumulator_update_with_nan():
    # NaN treated as 0 in sum
    acc = SumAccumulator()
    acc.update(np.array([2.0, np.nan, 6.0]))
    np.testing.assert_array_almost_equal(acc._value, [2.0, 0.0, 6.0])


def test_sum_accumulator_update_with_inf():
    # inf treated as 0 in sum
    acc = SumAccumulator()
    acc.update(np.array([2.0, np.inf, -np.inf]))
    np.testing.assert_array_almost_equal(acc._value, [2.0, 0.0, 0.0])


def test_sum_accumulator_multiple_updates():
    acc = SumAccumulator()
    acc.update(np.array([1.0, 2.0, 3.0]))
    acc.update(np.array([4.0, 5.0, 6.0]))
    np.testing.assert_array_almost_equal(acc._value, [5.0, 7.0, 9.0])


def test_sum_accumulator_finalize_basic():
    acc = SumAccumulator()
    acc.update(np.array([2.0, 4.0, 6.0]))
    acc.update(np.array([4.0, 2.0, 0.0]))
    result = acc.finalize()
    assert set(result.keys()) == {"sum"}
    np.testing.assert_array_almost_equal(result["sum"], [6.0, 6.0, 6.0])


def test_sum_accumulator_finalize_no_data():
    result = SumAccumulator().finalize()
    assert np.isnan(result["sum"])


# =============================================================================
# SumSqAccumulator
# =============================================================================


def test_sum_sq_accumulator_init():
    acc = SumSqAccumulator()
    assert acc._value is None


def test_sum_sq_accumulator_update_basic():
    acc = SumSqAccumulator()
    acc.update(np.array([2.0, 3.0]))
    np.testing.assert_array_almost_equal(acc._value, [4.0, 9.0])


def test_sum_sq_accumulator_update_with_nan():
    # NaN treated as 0 in sum_sq
    acc = SumSqAccumulator()
    acc.update(np.array([2.0, np.nan]))
    np.testing.assert_array_almost_equal(acc._value, [4.0, 0.0])


def test_sum_sq_accumulator_update_with_inf():
    # inf treated as 0 in sum_sq
    acc = SumSqAccumulator()
    acc.update(np.array([3.0, np.inf]))
    np.testing.assert_array_almost_equal(acc._value, [9.0, 0.0])


def test_sum_sq_accumulator_multiple_updates():
    acc = SumSqAccumulator()
    acc.update(np.array([1.0]))
    acc.update(np.array([2.0]))
    acc.update(np.array([3.0]))
    assert acc._value[0] == pytest.approx(14.0)  # 1² + 2² + 3²


def test_sum_sq_accumulator_finalize_basic():
    acc = SumSqAccumulator()
    values = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    for val in values:
        acc.update(np.array([val]))
    result = acc.finalize()
    assert set(result.keys()) == {"sum_sq"}
    assert result["sum_sq"][0] == pytest.approx(float(np.sum(values**2)))


def test_sum_sq_accumulator_finalize_vector():
    acc = SumSqAccumulator()
    acc.update(np.array([1.0, 10.0, 100.0]))
    acc.update(np.array([2.0, 20.0, 200.0]))
    acc.update(np.array([3.0, 30.0, 300.0]))
    result = acc.finalize()
    np.testing.assert_array_almost_equal(result["sum_sq"], [14.0, 1400.0, 140000.0])


def test_sum_sq_accumulator_finalize_no_data():
    result = SumSqAccumulator().finalize()
    assert np.isnan(result["sum_sq"])


# =============================================================================
# Integration
# =============================================================================


def test_integration_all_accumulators_same_data():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    count_acc = CountAccumulator()
    sum_acc = SumAccumulator()
    sum_sq_acc = SumSqAccumulator()

    for val in data:
        v = np.array([val])
        count_acc.update(v)
        sum_acc.update(v)
        sum_sq_acc.update(v)

    assert count_acc.finalize()["count"][0] == 5
    assert sum_acc.finalize()["sum"][0] == pytest.approx(15.0)
    assert sum_sq_acc.finalize()["sum_sq"][0] == pytest.approx(55.0)  # 1+4+9+16+25


def test_integration_vectorized_multi_column():
    col1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    col2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    col3 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    acc = SumAccumulator()
    for i in range(len(col1)):
        acc.update(np.array([col1[i], col2[i], col3[i]]))

    result = acc.finalize()
    assert result["sum"][0] == pytest.approx(np.sum(col1))
    assert result["sum"][1] == pytest.approx(np.sum(col2))
    assert result["sum"][2] == pytest.approx(np.sum(col3))


# =============================================================================
# HistogramAccumulator
# =============================================================================


def test_histogram_accumulator_init_valid():
    edges = [0.0, 1.0, 2.0]
    acc = HistogramAccumulator(edges)
    np.testing.assert_array_equal(acc._bin_edges, np.array(edges, dtype=np.float32))
    np.testing.assert_array_equal(acc._counts, np.zeros(2, dtype=np.int32))


def test_histogram_accumulator_init_too_few_bins():
    with pytest.raises(FedbiomedError, match="must define at least 2 bins"):
        HistogramAccumulator([0.0, 1.0])


def test_histogram_accumulator_init_not_strictly_increasing():
    with pytest.raises(FedbiomedError, match="must be strictly increasing"):
        HistogramAccumulator([0.0, 2.0, 1.0])

    with pytest.raises(FedbiomedError, match="must be strictly increasing"):
        HistogramAccumulator([0.0, 1.0, 1.0])


def test_histogram_accumulator_init_multidimensional_edges():
    with pytest.raises(FedbiomedError, match="at least 2 bins"):
        HistogramAccumulator([[0.0, 1.0], [2.0, 3.0]])


def test_histogram_accumulator_update_basic():
    acc = HistogramAccumulator([0.0, 10.0, 20.0, 30.0])
    acc.update(5.0)
    acc.update(15.0)
    acc.update(25.0)
    np.testing.assert_array_equal(acc._counts, [1, 1, 1])


def test_histogram_accumulator_update_edges():
    acc = HistogramAccumulator([0.0, 10.0, 20.0])
    acc.update(0.0)  # bin 0
    acc.update(9.99)  # bin 0
    acc.update(10.0)  # bin 1
    acc.update(20.0)  # bin 1 (rightmost edge)
    np.testing.assert_array_equal(acc._counts, [2, 2])


def test_histogram_accumulator_update_out_of_bounds():
    acc = HistogramAccumulator([0.0, 10.0, 20.0])
    acc.update(-1.0)
    acc.update(20.1)
    np.testing.assert_array_equal(acc._counts, [0, 0])


def test_histogram_accumulator_update_nan_inf():
    acc = HistogramAccumulator([0.0, 5.0, 10.0])
    acc.update(np.nan)
    acc.update(np.inf)
    acc.update(-np.inf)
    np.testing.assert_array_equal(acc._counts, [0, 0])


def test_histogram_accumulator_finalize():
    edges = [0.0, 1.0, 2.0]
    acc = HistogramAccumulator(edges)
    acc.update(0.5)
    acc.update(1.5)
    result = acc.finalize()
    assert result["histogram"]["bin_edges"] == edges
    assert result["histogram"]["counts"] == [1, 1]

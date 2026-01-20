# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from fedbiomed.common.analytics._aggregators import (
    aggregate_count,
    aggregate_histogram,
    aggregate_max,
    aggregate_mean,
    aggregate_min,
    aggregate_quantile,
    aggregate_std,
    aggregate_sum,
    aggregate_variance,
)
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


def test_aggregator_validation_args():
    """Test aggregator validation decorator."""
    # Argument is not a list
    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_min(123)
    assert ErrorNumbers.FB633.value in str(excinfo.value)
    assert "must be a list" in str(excinfo.value)

    # Missing argument
    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_min()
    assert ErrorNumbers.FB633.value in str(excinfo.value)
    assert "Missing required argument" in str(excinfo.value)

    # Extra arguments are ignored (filtered out by decorator)
    # aggregate_min([1.0, 2.0], extra_arg=True) should not raise TypeError
    assert aggregate_min([1.0, 2.0], extra_arg=True) == 1.0


def test_aggregate_min():
    """Test aggregate_min function."""
    # Normal case
    assert aggregate_min([1.0, 2.0, 3.0]) == 1.0
    assert aggregate_min([-5.0, 0.0, 5.0]) == -5.0

    # Edge cases
    with pytest.raises(FedbiomedError):
        aggregate_min([])
    assert aggregate_min([10.0]) == 10.0
    assert aggregate_min([np.inf, 1.0]) == 1.0


def test_aggregate_max():
    """Test aggregate_max function."""
    # Normal case
    assert aggregate_max([1.0, 2.0, 3.0]) == 3.0
    assert aggregate_max([-5.0, 0.0, 5.0]) == 5.0

    # Edge cases
    with pytest.raises(FedbiomedError):
        aggregate_max([])
    assert aggregate_max([10.0]) == 10.0
    assert aggregate_max([-np.inf, 1.0]) == 1.0


def test_aggregate_count():
    """Test aggregate_count function."""
    # Normal case
    assert aggregate_count([10, 20, 30]) == 60
    assert aggregate_count([0, 5]) == 5

    # Edge cases
    with pytest.raises(FedbiomedError):
        aggregate_count([])

    # Error cases
    # Negative count
    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_count([10, -5])
    assert ErrorNumbers.FB633.value in str(excinfo.value)

    # non-integer
    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_count([10, "5"])
    assert ErrorNumbers.FB633.value in str(excinfo.value)

    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_count([10, 1.5])
    assert ErrorNumbers.FB633.value in str(excinfo.value)


def test_aggregate_sum():
    """Test aggregate_sum function."""
    # Normal case
    sums = [2, 4, 6]
    assert aggregate_sum(sums) == 12

    # Floats
    sums = [1.5, 2.5]
    assert aggregate_sum(sums) == 4.0

    # Empty
    with pytest.raises(FedbiomedError):
        aggregate_sum([])


def test_aggregate_mean():
    """Test aggregate_mean function."""
    # Normal case
    means = [10.0, 20.0]
    counts = [2, 2]
    # (20 + 40) / 4 = 15.0
    assert aggregate_mean(means, counts) == 15.0

    # Weighted case
    means = [10.0, 20.0]
    counts = [1, 3]  # Total count 4. Sum = 10*1 + 20*3 = 70. Mean = 70/4 = 17.5
    assert aggregate_mean(means, counts) == 17.5

    # Edge cases
    with pytest.raises(FedbiomedError):
        aggregate_mean([], [])
    # Zero total count
    assert np.isnan(aggregate_mean([10.0], [0]))


def test_aggregate_variance():
    """Test aggregate_variance function."""

    means = [2.0, 6.0]
    variances = [2.0, 2.0]
    counts = [2, 2]

    res = aggregate_variance(means, variances, counts)
    assert np.isclose(res, 20.0 / 3.0)

    # Single element total (variance undefined for N=1 if using sample variance)
    # If count=1, func returns nan
    assert np.isnan(aggregate_variance([1.0], [0.0], [1]))

    # Error cases
    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_variance([1.0], [1.0], [])
    assert ErrorNumbers.FB633.value in str(excinfo.value)

    # Mismatch mean/variance length
    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_variance([1.0, 2.0], [1.0], [1, 2])
    assert ErrorNumbers.FB633.value in str(excinfo.value)


def test_aggregate_std():
    """Test aggregate_std function."""
    # Same logic as variance
    means = [2.0, 6.0]
    stds = [np.sqrt(2.0), np.sqrt(2.0)]
    counts = [2, 2]

    expected_std = np.sqrt(20.0 / 3.0)
    res = aggregate_std(means, stds, counts)
    assert np.isclose(res, expected_std)


def test_aggregate_histogram():
    """Test aggregate_histogram function."""

    hist1 = {"bin_edges": [0, 1, 2], "counts": [1, 2]}
    hist2 = {"bin_edges": [0, 1, 2], "counts": [3, 4]}

    # Normal case
    res = aggregate_histogram([hist1, hist2])
    assert res["bin_edges"] == hist1["bin_edges"]
    assert res["counts"] == [4, 6]

    # Mismatched bin_edges
    hist3 = {"bin_edges": [0, 1, 3], "counts": [1, 2]}
    res_mismatch = aggregate_histogram([hist1, hist3])
    assert res_mismatch is None

    # Empty list -> FedbiomedError
    with pytest.raises(FedbiomedError):
        aggregate_histogram([])


def test_aggregate_quantile():
    """Test aggregate_quantile function."""

    # 1. Single quantile
    q1 = {"q": [0.5], "values": [10.0]}
    q2 = {"q": [0.5], "values": [20.0]}
    res = aggregate_quantile([q1, q2])
    assert res["q"] == [0.5]
    assert np.isclose(res["values"][0], 15.0)

    # 2. Multiple quantiles
    q1 = {"q": [0.25, 0.5], "values": [10.0, 5.0]}
    q2 = {"q": [0.25, 0.5], "values": [20.0, 15.0]}

    # Normal case
    res = aggregate_quantile([q1, q2])
    assert res["q"] == [0.25, 0.5]
    assert np.allclose(res["values"], [15.0, 10.0])

    # 3. Mismatched q -> None
    q1 = {"q": [0.25, 0.5], "values": [10.0, 5.0]}
    q2 = {"q": [0.5], "values": [20.0]}
    assert aggregate_quantile([q1, q2]) is None

    # Mismatched q values (different order or values)
    q3 = {"q": [0.5, 0.25], "values": [5.0, 10.0]}
    assert aggregate_quantile([q1, q3]) is None

    # 4. Empty inputs
    with pytest.raises(FedbiomedError):
        aggregate_quantile([])

    # 5. Empty content (valid structure but empty lists)
    # Ideally should handle or return empty lists
    q_empty = {"q": [], "values": []}
    res = aggregate_quantile([q_empty, q_empty])
    assert res["q"] == []
    assert res["values"] == []

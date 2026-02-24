# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import inspect

import numpy as np
import pytest

from fedbiomed.common.analytics._aggregators import (
    AGGREGATORS_MAP,
    aggregate_count,
    aggregate_histogram,
    aggregate_max,
    aggregate_mean,
    aggregate_min,
    aggregate_quantile,
    aggregate_std,
    aggregate_sum,
    aggregate_variance,
    aggregator,
)
from fedbiomed.common.constants import ErrorNumbers, Stats
from fedbiomed.common.exceptions import FedbiomedError


def test_all_aggregator_params_are_valid_stats():
    """Every parameter name in every registered aggregator must be a Stats value."""
    valid = {s.value for s in Stats}
    for stat_name, func in AGGREGATORS_MAP.items():
        params = list(inspect.signature(func).parameters)
        invalid = [p for p in params if p not in valid]
        assert not invalid, (
            f"Aggregator for '{stat_name}' has invalid parameter(s) {invalid}. "
            f"Valid Stats values: {sorted(valid)}"
        )


def test_aggregator_invalid_param_raises_at_decoration():
    """Decorating a function with a non-Stats parameter name raises FedbiomedError."""
    with pytest.raises(FedbiomedError, match="not valid Stats enum values"):

        @aggregator("test_stat")
        def _bad(not_a_stat_name: list):
            pass


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
    means = [2.0, 4.0, 6.0]
    counts = [1, 1, 1]
    assert aggregate_sum(means, counts) == 12.0

    # Floats
    means = [1.5, 2.5]
    counts = [1, 1]
    assert aggregate_sum(means, counts) == 4.0

    # Empty
    with pytest.raises(FedbiomedError):
        aggregate_sum([], [])


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
    variances = [2.0, 2.0]
    counts = [2, 2]

    expected_std = np.sqrt(20.0 / 3.0)
    res = aggregate_std(means, variances, counts)
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
    with pytest.raises(FedbiomedError):
        aggregate_histogram([hist1, hist3])

    # Empty list -> FedbiomedError
    with pytest.raises(FedbiomedError):
        aggregate_histogram([])


def test_aggregate_quantile():
    """Test aggregate_quantile function (histogram-based, range output)."""

    bin_edges = [0.0, 10.0, 20.0, 30.0]
    q_levels = [0.25, 0.5, 0.75]

    h1 = {"bin_edges": bin_edges, "counts": [5, 10, 5]}
    h2 = {"bin_edges": bin_edges, "counts": [5, 10, 5]}

    # 1. Normal case — two identical nodes
    res = aggregate_quantile([h1, h2], q_levels)
    assert set(res.keys()) == set(q_levels)
    for q in q_levels:
        entry = res[q]
        assert "value" in entry and "min" in entry and "max" in entry
        assert entry["min"] < entry["max"]
        assert entry["min"] <= entry["value"] <= entry["max"]

    # 2. Mismatched bin_edges -> FedbiomedError
    h3 = {"bin_edges": [0.0, 5.0, 20.0, 30.0], "counts": [5, 10, 5]}
    with pytest.raises(FedbiomedError):
        aggregate_quantile([h1, h3], q_levels)

    # 3. Empty histogram list -> FedbiomedError
    with pytest.raises(FedbiomedError):
        aggregate_quantile([], q_levels)

    # 4. Empty quantile list -> FedbiomedError
    with pytest.raises(FedbiomedError):
        aggregate_quantile([h1, h2], [])

    # 5. All-zero counts -> NaN values for every quantile level
    h_zero = {"bin_edges": bin_edges, "counts": [0, 0, 0]}
    res_zero = aggregate_quantile([h_zero], q_levels)
    assert set(res_zero.keys()) == set(q_levels)
    for q in q_levels:
        assert np.isnan(res_zero[q]["value"])

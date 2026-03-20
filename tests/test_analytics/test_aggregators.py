# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from fedbiomed.common.analytics._aggregators import (
    AGGREGATORS_MAP,
    aggregate_count,
    aggregate_histogram,
    aggregate_mean,
    aggregate_quantile,
    aggregate_std,
    aggregate_sum,
    aggregate_variance,
    aggregator,
)
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


def test_aggregator_allows_any_param_name():
    """Parameters are no longer restricted to Stats enum values."""

    @aggregator("test_stat")
    def _ok(not_a_stat_name: list):
        return not_a_stat_name

    assert _ok([1, 2]) == [1, 2]
    del AGGREGATORS_MAP["test_stat"]


def test_aggregator_validation_args():
    """Test aggregator validation decorator."""
    # Argument is not a list
    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_count(123)
    assert ErrorNumbers.FB633.value in str(excinfo.value)
    assert "must be a list" in str(excinfo.value)

    # Missing argument
    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_count()
    assert ErrorNumbers.FB633.value in str(excinfo.value)
    assert "Missing required argument" in str(excinfo.value)

    # Extra arguments are ignored (filtered out by decorator)
    # aggregate_count([10, 20], extra_arg=True) should not raise TypeError
    assert aggregate_count([10, 20], extra_arg=True) == 30


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


def test_aggregate_count_dicts():
    """Test aggregate_count with list of dicts (categorical counts)."""
    # Normal case: same keys, values are summed per key
    node1 = {"cat": 3, "dog": 5}
    node2 = {"cat": 2, "dog": 4}
    result = aggregate_count([node1, node2])
    assert result == {"cat": 5, "dog": 9}

    # Single node
    assert aggregate_count([{"a": 0, "b": 10}]) == {"a": 0, "b": 10}

    # Zero counts are valid
    assert aggregate_count([{"x": 0}, {"x": 0}]) == {"x": 0}

    # numpy integers are accepted as values
    node_np = {"a": np.int64(4), "b": np.int32(6)}
    result_np = aggregate_count([node_np, {"a": 1, "b": 2}])
    assert result_np == {"a": 5, "b": 8}

    # Different keys across nodes: missing keys default to 0 (union)
    result_union = aggregate_count([{"cat": 3, "dog": 5}, {"cat": 2, "bird": 1}])
    assert result_union == {"cat": 5, "dog": 5, "bird": 1}

    # Negative dict value raises FedbiomedError
    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_count([{"a": 3}, {"a": -1}])
    assert ErrorNumbers.FB633.value in str(excinfo.value)

    # Non-integer dict value raises FedbiomedError
    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_count([{"a": 1.5}, {"a": 2}])
    assert ErrorNumbers.FB633.value in str(excinfo.value)

    # Mixed list (int and dict) raises FedbiomedError
    with pytest.raises(FedbiomedError) as excinfo:
        aggregate_count([5, {"a": 3}])
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

    # Mismatched lengths
    with pytest.raises(FedbiomedError):
        aggregate_sum([1.0, 2.0], [1])


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

    # Mismatched lengths
    with pytest.raises(FedbiomedError):
        aggregate_mean([1.0, 2.0], [1])


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

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
    aggregate_std,
    aggregate_sum,
    aggregate_variance,
)
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


class TestAnalyticsAggregators:
    """Tests for analytic aggregator functions."""

    def test_aggregate_min(self):
        """Test aggregate_min function."""
        # Normal case
        assert aggregate_min([1.0, 2.0, 3.0]) == 1.0
        assert aggregate_min([-5.0, 0.0, 5.0]) == -5.0

        # Edge cases
        with pytest.raises(FedbiomedError):
            aggregate_min([])
        assert aggregate_min([10.0]) == 10.0
        assert aggregate_min([np.inf, 1.0]) == 1.0

    def test_aggregate_max(self):
        """Test aggregate_max function."""
        # Normal case
        assert aggregate_max([1.0, 2.0, 3.0]) == 3.0
        assert aggregate_max([-5.0, 0.0, 5.0]) == 5.0

        # Edge cases
        with pytest.raises(FedbiomedError):
            aggregate_max([])
        assert aggregate_max([10.0]) == 10.0
        assert aggregate_max([-np.inf, 1.0]) == 1.0

    def test_aggregate_count(self):
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

    def test_aggregate_sum(self):
        """Test aggregate_sum function."""
        # Normal case
        means = [2.0, 4.0, 6.0]
        counts = [10, 10, 10]
        # Sum = (2*10) + (4*10) + (6*10) = 20 + 40 + 60 = 120
        assert aggregate_sum(means, counts) == 120.0

        # Floats
        means = [1.5, 2.5]
        counts = [2, 2]
        assert aggregate_sum(means, counts) == 3.0 + 5.0

        # Empty
        with pytest.raises(FedbiomedError):
            aggregate_sum([], [])

        # Error cases
        with pytest.raises(FedbiomedError) as excinfo:
            aggregate_sum([1.0], [1, 2])
        assert ErrorNumbers.FB633.value in str(excinfo.value)

    def test_aggregate_mean(self):
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

    def test_aggregate_variance(self):
        """Test aggregate_variance function."""
        # Case using known population
        # Sample 1: [1, 3] -> mean=2, std=sqrt(2) approx 1.414, count=2. Var(s^2) = 2.
        # Sample 2: [5, 7] -> mean=6, std=sqrt(2), count=2. Var(s^2) = 2.
        # Combined: [1, 3, 5, 7] -> Mean=4.
        # Variance = ((1-4)^2 + (3-4)^2 + (5-4)^2 + (7-4)^2) / (4-1)
        # = (9 + 1 + 1 + 9) / 3 = 20/3 = 6.666...

        means = [2.0, 6.0]
        stds = [np.sqrt(2.0), np.sqrt(2.0)]
        counts = [2, 2]

        res = aggregate_variance(means, stds, counts)
        assert np.isclose(res, 20.0 / 3.0)

        # Single element total (variance undefined for N=1 if using sample variance)
        # If count=1, func returns nan
        assert np.isnan(aggregate_variance([1.0], [0.0], [1]))

        # Error cases
        with pytest.raises(FedbiomedError) as excinfo:
            aggregate_variance([1.0], [1.0], [])
        assert ErrorNumbers.FB633.value in str(excinfo.value)

    def test_aggregate_std(self):
        """Test aggregate_std function."""
        # Same logic as variance
        means = [2.0, 6.0]
        stds = [np.sqrt(2.0), np.sqrt(2.0)]
        counts = [2, 2]

        expected_std = np.sqrt(20.0 / 3.0)
        res = aggregate_std(means, stds, counts)
        assert np.isclose(res, expected_std)

    def test_aggregate_histogram(self):
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

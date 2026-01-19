# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
from typing import Dict, List, Union

import numpy as np

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger


def validate_aggregator_args(func):
    # Fetch signature once at decoration time for efficiency
    sig = inspect.signature(func)
    params = sig.parameters

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Filter kwargs to keep only arguments present in function signature
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}

        # Bind arguments to check for missing required arguments
        try:
            bound = sig.bind(*args, **filtered_kwargs)
        except TypeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: Missing required argument: {e}"
            ) from e

        for name, val in bound.arguments.items():
            if not isinstance(val, list):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB633.value}: Argument {name} must be a list"
                )
            if not val:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB633.value}: Argument {name} is empty"
                )
        return func(*bound.args, **bound.kwargs)

    return wrapper


@validate_aggregator_args
def aggregate_min(min: List[float]) -> float:
    """Aggregates minimum values.

    Args:
        min: List of minimum values from nodes.

    Returns:
        The global minimum.
    """
    return np.min(min)


@validate_aggregator_args
def aggregate_max(max: List[float]) -> float:
    """Aggregates maximum values.

    Args:
        max: List of maximum values from nodes.

    Returns:
        The global maximum.
    """
    return np.max(max)


@validate_aggregator_args
def aggregate_count(count: List[int]) -> int:
    """Aggregates count values.

    Args:
        count: List of counts from nodes.

    Returns:
        The total count.
    """
    if not all(isinstance(c, int) and c >= 0 for c in count):
        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: All count values must be non-negative integers."
        )
    return sum(count)


@validate_aggregator_args
def aggregate_sum(sum: List[float]) -> float:
    """Aggregates sums.

    Args:
        sum: List of sums from nodes.

    Returns:
        The total sum.
    """
    return np.sum(sum)


@validate_aggregator_args
def aggregate_mean(mean: List[float], count: List[int]) -> float:
    """Aggregates mean values using counts as weights.

    Args:
        mean: List of means from nodes.
        count: List of counts from nodes.

    Returns:
        The global mean.
    """
    total_count = aggregate_count(count)
    if total_count == 0:
        return np.nan
    total_sum = sum(m * c for m, c in zip(mean, count, strict=True))
    return total_sum / total_count


@validate_aggregator_args
def aggregate_variance(
    mean: List[float], variance: List[float], count: List[int]
) -> float:
    """Aggregates variance using means, variances, and counts.
    Assumes variance are sample variances (ddof=1).
    Returns sample variance.

    Args:
        mean: List of means from nodes.
        variance: List of variances from nodes.
        count: List of counts from nodes.

    Returns:
        The global sample variance.
    """
    if len(mean) != len(variance) or len(mean) != len(count):
        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: Mean, variance, and count lists must have the same length."
        )

    total_count = aggregate_count(count)
    if total_count <= 1:
        return np.nan

    # Calculate combined variance (sum( SS_within + SS_between ) / (N-1))
    # SS_within = sum( (n_i - 1) * s_i^2 )
    # SS_between = sum( n_i * (mean_i - global_mean)^2 )
    global_mean = aggregate_mean(mean, count)
    ss_within = sum((c - 1) * v for c, v in zip(count, variance, strict=True))
    ss_between = sum(
        c * ((m - global_mean) ** 2) for m, c in zip(mean, count, strict=True)
    )

    return (ss_within + ss_between) / (total_count - 1)


@validate_aggregator_args
def aggregate_std(mean: List[float], std: List[float], count: List[int]) -> float:
    """Aggregates standard deviation using means, stds, and counts.

    Args:
        mean: List of means from nodes.
        std: List of standard deviations from nodes.
        count: List of counts from nodes.

    Returns:
        The global sample standard deviation.
    """
    variance = [s**2 for s in std]
    var = aggregate_variance(mean, variance, count)
    return np.sqrt(var)


@validate_aggregator_args
def aggregate_histogram(
    histogram: List[Dict[str, Union[List[int], List[float]]]],
) -> Dict[str, Union[List[int], List[float]]]:
    """Aggregates histograms from nodes.

    Args:
        histograms: List of histograms from nodes, each as a dict with 'bin_edges' and 'counts'.

    Returns:
        The aggregated histogram as a dict with 'bin_edges' and 'counts'.
    """
    if not all(h["bin_edges"] == histogram[0]["bin_edges"] for h in histogram):
        logger.info("Bin edges do not match across histograms; cannot aggregate.")
        return None

    return {
        "bin_edges": histogram[0]["bin_edges"],
        "counts": list(np.sum([h["counts"] for h in histogram], axis=0)),
    }

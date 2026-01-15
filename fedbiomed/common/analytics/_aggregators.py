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
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        for name, val in bound.arguments.items():
            if name == "kwargs":
                continue
            if not isinstance(val, list):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB633.value}: Argument {name} must be a list"
                )
            if not val:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB633.value}: Argument {name} is empty"
                )
        return func(*args, **kwargs)

    return wrapper


@validate_aggregator_args
def aggregate_min(min: List[float], **kwargs) -> float:
    """Aggregates minimum values.

    Args:
        min: List of minimum values from nodes.

    Returns:
        The global minimum.
    """
    return np.min(min)


@validate_aggregator_args
def aggregate_max(max: List[float], **kwargs) -> float:
    """Aggregates maximum values.

    Args:
        max: List of maximum values from nodes.

    Returns:
        The global maximum.
    """
    return np.max(max)


@validate_aggregator_args
def aggregate_count(count: List[int], **kwargs) -> int:
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
def aggregate_sum(mean: List[float], count: List[int], **kwargs) -> float:
    """Aggregates sum using mean and count.

    Args:
        mean: List of means from nodes.
        count: List of counts from nodes.

    Returns:
        The total sum.
    """
    if len(mean) != len(count):
        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: Mean and count lists must have the same length."
        )

    total_sum = 0.0
    for m, c in zip(mean, count, strict=True):
        total_sum += m * c
    return total_sum


@validate_aggregator_args
def aggregate_mean(mean: List[float], count: List[int], **kwargs) -> float:
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

    total_sum = aggregate_sum(mean, count)
    return total_sum / total_count


@validate_aggregator_args
def aggregate_variance(
    mean: List[float], std: List[float], count: List[int], **kwargs
) -> float:
    """Aggregates variance using means, stds, and counts.
    Assumes std are sample standard deviations (ddof=1).
    Returns sample variance.

    Args:
        mean: List of means from nodes.
        std: List of standard deviations from nodes.
        count: List of counts from nodes.

    Returns:
        The global sample variance.
    """
    if len(mean) != len(std) or len(mean) != len(count):
        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: Mean, std, and count lists must have the same length."
        )

    total_count = aggregate_count(count)
    if total_count <= 1:
        return np.nan

    # Calculate combined variance (sum( SS_within + SS_between ) / (N-1))
    # SS_within = sum( (n_i - 1) * s_i^2 )
    # SS_between = sum( n_i * (mean_i - global_mean)^2 )
    global_mean = aggregate_mean(mean, count)
    ss_within = sum((c - 1) * (s**2) for c, s in zip(count, std, strict=True))
    ss_between = sum(
        c * ((m - global_mean) ** 2) for m, c in zip(mean, count, strict=True)
    )

    return (ss_within + ss_between) / (total_count - 1)


@validate_aggregator_args
def aggregate_std(
    mean: List[float], std: List[float], count: List[int], **kwargs
) -> float:
    """Aggregates standard deviation using means, stds, and counts.

    Args:
        mean: List of means from nodes.
        std: List of standard deviations from nodes.
        count: List of counts from nodes.

    Returns:
        The global sample standard deviation.
    """
    var = aggregate_variance(mean, std, count)
    return np.sqrt(var)


@validate_aggregator_args
def aggregate_histogram(
    histograms: List[Dict[str, Union[List[int], List[float]]]], **kwargs
) -> Dict[str, Union[List[int], List[float]]]:
    """Aggregates histograms from nodes.

    Args:
        histograms: List of histograms from nodes, each as a dict with 'bin_edges' and 'counts'.

    Returns:
        The aggregated histogram as a dict with 'bin_edges' and 'counts'.
    """
    if not all(h["bin_edges"] == histograms[0]["bin_edges"] for h in histograms):
        logger.info("Bin edges do not match across histograms; cannot aggregate.")
        return None

    return {
        "bin_edges": histograms[0]["bin_edges"],
        "counts": list(np.sum([h["counts"] for h in histograms], axis=0)),
    }

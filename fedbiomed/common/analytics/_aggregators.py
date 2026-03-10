# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
from typing import Callable, Dict, List, Union

import numpy as np

from fedbiomed.common.constants import ErrorNumbers, Stats
from fedbiomed.common.exceptions import FedbiomedError

# Maps every stat to its aggregator function.
AGGREGATORS_MAP: Dict[str, Callable] = {}


def aggregator(stat: str):
    """Register, validate, and wrap an aggregator function.

    Decoration time:
      - Raises FedbiomedError if any parameter name is not a Stats enum value.
      - Registers the wrapped function in AGGREGATORS_MAP under *stat*.

    Call time:
      - Filters out unknown kwargs.
    """

    def decorator(func):
        sig = inspect.signature(func)
        params = sig.parameters

        # Validation of parameter names against Stats enum values
        invalid = [p for p in params if p not in {s.value for s in Stats}]
        if invalid:
            raise FedbiomedError(
                f"Aggregator '{func.__name__}' has parameter(s) {invalid} "
                f"that are not valid Stats enum values. "
                f"Valid values: {sorted(s.value for s in Stats)}"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
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

        AGGREGATORS_MAP[stat] = wrapper
        return wrapper

    return decorator


@aggregator("min")
def aggregate_min(min: List[float]) -> float:
    """Aggregates minimum values.

    Args:
        min: List of minimum values from nodes.

    Returns:
        The global minimum.
    """
    return np.min(min)


@aggregator("max")
def aggregate_max(max: List[float]) -> float:
    """Aggregates maximum values.

    Args:
        max: List of maximum values from nodes.

    Returns:
        The global maximum.
    """
    return np.max(max)


@aggregator("count")
def aggregate_count(
    count: List[Union[int, Dict[str, int]]],
) -> Union[int, Dict[str, int]]:
    """Aggregates count values.

    Args:
        count: List of counts from nodes. Each element is either a non-negative
            integer or a dict mapping category labels to non-negative integer counts.

    Returns:
        The total count as an int, or a dict of summed counts.
    """
    if all(isinstance(c, (int, np.integer)) for c in count):
        if not all(c >= 0 for c in count):
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: All count values must be non-negative integers."
            )
        return int(np.sum(count))

    if all(isinstance(c, dict) for c in count):
        result: Dict[str, int] = {}
        for c in count:
            for k, v in c.items():
                if not isinstance(v, (int, np.integer)) or v < 0:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB633.value}: All count dict values must be non-negative integers."
                    )
                result[k] = result.get(k, 0) + int(v)
        return result

    raise FedbiomedError(
        f"{ErrorNumbers.FB633.value}: count must be a list of ints or a list of dicts."
    )


@aggregator("sum")
def aggregate_sum(mean: List[float], count: List[Union[int, np.integer]]) -> float:
    """Aggregates sum values using means and counts.

    Args:
        mean: List of means from nodes.
        count: List of counts from nodes.

    Returns:
        The total sum.
    """
    total_sum = sum(m * c for m, c in zip(mean, count, strict=True))
    return total_sum


@aggregator("mean")
def aggregate_mean(mean: List[float], count: List[Union[int, np.integer]]) -> float:
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


@aggregator("variance")
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


@aggregator("std")
def aggregate_std(mean: List[float], variance: List[float], count: List[int]) -> float:
    """Aggregates standard deviation using means, variances, and counts.

    Args:
        mean: List of means from nodes.
        variance: List of variances from nodes.
        count: List of counts from nodes.

    Returns:
        The global sample standard deviation.
    """
    var = aggregate_variance(mean, variance, count)
    return np.sqrt(var)


@aggregator("histogram")
def aggregate_histogram(
    histogram: List[Dict[str, Union[List[int], List[float]]]],
) -> Dict[str, Union[List[int], List[float]]]:
    """Aggregates histograms from nodes.

    Args:
        histogram: List of histograms from nodes, each as a dict with 'bin_edges' and 'counts'.

    Returns:
        The aggregated histogram as a dict with 'bin_edges' and 'counts'.
    """
    if not all(h["bin_edges"] == histogram[0]["bin_edges"] for h in histogram):
        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: Bin edges do not match across histograms; cannot aggregate."
        )

    return {
        "bin_edges": histogram[0]["bin_edges"],
        "counts": list(np.sum([h["counts"] for h in histogram], axis=0)),
    }


@aggregator("quantile")
def aggregate_quantile(
    histogram: List[Dict[str, Union[List[int], List[float]]]],
    quantile: List[float],
) -> Dict[float, Dict[str, float]]:
    """Estimates quantiles from per-node histograms.

    Args:
        histogram: List of per-node dicts, each with:
            - ``"bin_edges"``: monotonically increasing bin boundaries (length n+1)
            - ``"counts"``: per-bin counts (length n)
        quantile: Requested quantile levels in (0, 1] (e.g. ``[0.25, 0.5, 0.75]``).

    Returns:
        Dict keyed by each requested quantile level. Each value is a dict with:
        - ``"value"``: linearly interpolated point estimate within the target bin
        - ``"min"``:   lower bin-edge bound (range lower bound)
        - ``"max"``:   upper bin-edge bound (range upper bound)

    Raises:
        FedbiomedError: If ``bin_edges`` do not match across nodes.
    """
    aggregated_histogram = aggregate_histogram(histogram)
    total_counts = np.array(aggregated_histogram["counts"])
    total_n = int(total_counts.sum())

    if total_n == 0:
        return {q: {"value": np.nan, "min": np.nan, "max": np.nan} for q in quantile}

    edges = np.array(aggregated_histogram["bin_edges"])
    cumulative = np.cumsum(total_counts)

    result = {}
    for q in quantile:
        target = q * total_n
        bin_idx = int(np.searchsorted(cumulative, target, side="left"))
        bin_idx = min(bin_idx, len(total_counts) - 1)  # clamp to last bin

        lo, hi = float(edges[bin_idx]), float(edges[bin_idx + 1])
        prev_cum = float(cumulative[bin_idx - 1]) if bin_idx > 0 else 0.0
        bin_count = float(total_counts[bin_idx])
        frac = (target - prev_cum) / bin_count if bin_count > 0 else 0.5

        result[q] = {"value": lo + frac * (hi - lo), "min": lo, "max": hi}

    return result

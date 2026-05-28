# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
from typing import Callable, Dict, List, Union, cast

import numpy as np

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError

# Maps every stat to its aggregator function.
AGGREGATORS_MAP: Dict[str, Callable] = {}


def aggregator(stat: str):
    """Register and wrap an aggregator function.

    Decoration time:
      - Registers the wrapped function in AGGREGATORS_MAP under *stat*.

    Call time:
      - Filters out unknown kwargs.
      - Validates that all arguments are non-empty lists.
    """

    def decorator(func):
        sig = inspect.signature(func)
        params = sig.parameters

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
        int_counts = cast(List[int], count)
        if not all(c >= 0 for c in int_counts):
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: All count values must be non-negative integers."
            )
        return int(np.sum(int_counts))

    if all(isinstance(c, dict) for c in count):
        result: Dict[str, int] = {}
        for c in cast(List[Dict[str, int]], count):
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
def aggregate_sum(sum: List[float]) -> float:
    """Aggregates the summable ``sum`` wire primitive across nodes.

    Args:
        sum: List of per-node sums (Σ x per node).

    Returns:
        The global sum.
    """
    if not all(isinstance(s, (int, float, np.number)) for s in sum):
        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: sum must be a list of numeric values."
        )
    return float(np.sum(sum))


@aggregator("sum_sq")
def aggregate_sum_sq(sum_sq: List[float]) -> float:
    """Aggregates the summable ``sum_sq`` wire primitive (Σ x²) across nodes.

    Args:
        sum_sq: List of per-node sums of squares.

    Returns:
        The global sum of squares.
    """
    if not all(isinstance(s, (int, float, np.number)) for s in sum_sq):
        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: sum_sq must be a list of numeric values."
        )
    return float(np.sum(sum_sq))


@aggregator("mean")
def aggregate_mean(sum: List[float], count: List[Union[int, np.integer]]) -> float:
    """Aggregates global mean from sufficient statistics.

    Args:
        sum: List of per-node sums (Σ x per node).
        count: List of per-node counts.

    Returns:
        The global mean.
    """
    if (total_count := aggregate_count(count)) == 0:
        return np.nan
    total_sum = aggregate_sum(sum)
    return total_sum / total_count


@aggregator("variance")
def aggregate_variance(
    sum_sq: List[float], sum: List[float], count: List[int]
) -> float:
    """Aggregates global sample variance from sufficient statistics.

    Uses the computational formula:
        variance = (Σ x² − (Σ x)² / N) / (N − 1)

    Args:
        sum_sq: List of per-node sums of squares (Σ x² per node).
        sum: List of per-node sums (Σ x per node).
        count: List of per-node counts.

    Returns:
        The global sample variance (ddof=1), or ``nan`` when N ≤ 1.
    """

    if (total_count := aggregate_count(count)) <= 1:
        return np.nan
    total_sum = aggregate_sum(sum)
    total_sum_sq = aggregate_sum_sq(sum_sq)
    return (total_sum_sq - total_sum**2 / total_count) / (total_count - 1)


@aggregator("std")
def aggregate_std(sum_sq: List[float], sum: List[float], count: List[int]) -> float:
    """Aggregates global sample standard deviation from sufficient statistics.

    Args:
        sum_sq: List of per-node sums of squares (Σ x² per node).
        sum: List of per-node sums (Σ x per node).
        count: List of per-node counts.

    Returns:
        The global sample standard deviation.
    """
    return float(np.sqrt(aggregate_variance(sum_sq, sum, count)))


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

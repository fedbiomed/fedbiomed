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
    count: List[Union[int, float, Dict[str, Union[int, float]]]],
) -> Union[int, Dict[str, int]]:
    """Aggregates count values.

    Counts are conceptually non-negative integers, but secure aggregation
    encodes them as floats. Such values are rounded to the nearest integer
    before the non-negativity check, so values within ±0.5 clamp to zero.

    Args:
        count: List of counts from nodes. Each element is either a non-negative
            number or a dict mapping category labels to non-negative counts.
            Values may be ints or floats (the latter coming from secagg).

    Returns:
        The total count as an int, or a dict of summed counts.
    """
    if all(isinstance(c, (int, float, np.number)) for c in count):
        num_counts = cast(List[Union[int, float]], count)
        rounded_counts = [int(round(c)) for c in num_counts]
        if not all(c >= 0 for c in rounded_counts):
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: All count values must be non-negative."
            )
        return int(np.sum(rounded_counts))

    if all(isinstance(c, dict) for c in count):
        result: Dict[str, int] = {}
        for c in cast(List[Dict[str, Union[int, float]]], count):
            for k, v in c.items():
                if not isinstance(v, (int, float, np.number)):
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB633.value}: All count dict values must be numeric."
                    )
                rounded_v = int(round(v))
                if rounded_v < 0:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB633.value}: All count dict values must be non-negative."
                    )
                result[k] = result.get(k, 0) + rounded_v
        return result

    raise FedbiomedError(
        f"{ErrorNumbers.FB633.value}: count must be a list of numbers or a list of dicts."
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


@aggregator("sum_sq_centered")
def aggregate_sum_sq_centered(sum_sq_centered: List[float]) -> float:
    """Aggregates the summable ``sum_sq_centered`` wire primitive across nodes.

    Each node contributes Σ (x - μ)² where μ is the *global* mean. Summing these per-node
    centered second moments yields the total M2 from which variance and std are derived.

    Args:
        sum_sq_centered: List of per-node centered sums of squares (Σ (x - μ)²).

    Returns:
        The global centered sum of squares (total M2 about the global mean).
    """
    if not all(isinstance(s, (int, float, np.number)) for s in sum_sq_centered):
        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: sum_sq_centered must be a list of numeric values."
        )
    return float(np.sum(sum_sq_centered))


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
def aggregate_variance(sum_sq_centered: List[float], count: List[int]) -> float:
    """Aggregates global sample variance from centered sufficient statistics.

    Uses the numerically stable centered form: variance = Σ (x - μ)² / (N - 1)

    Args:
        sum_sq_centered: List of per-node centered sums of squares (Σ (x - μ)²).
        count: List of per-node counts.

    Returns:
        The global sample variance (ddof=1), or ``nan`` when N ≤ 1.
    """
    if (total_count := aggregate_count(count)) <= 1:
        return np.nan
    return aggregate_sum_sq_centered(sum_sq_centered) / (total_count - 1)


@aggregator("std")
def aggregate_std(sum_sq_centered: List[float], count: List[int]) -> float:
    """Aggregates global sample standard deviation from centered statistics.

    Args:
        sum_sq_centered: List of per-node centered sums of squares (Σ (x - μ)²).
        count: List of per-node counts.

    Returns:
        The global sample standard deviation.
    """
    return float(np.sqrt(aggregate_variance(sum_sq_centered, count)))


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

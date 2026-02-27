# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Scalar and 1D streaming accumulators: Count, Min, Max, Mean, Variance, ScalarBuffer, Histogram, Quantile."""

from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from fedbiomed.common.constants import FedbiomedError

from ._base import Accumulator

# =============================================================================
# BUFFER ACCUMULATOR
# =============================================================================


class ScalarBuffer(Accumulator):
    """Buffer-based accumulator with configurable statistics."""

    def __init__(
        self, length: int, stat_functions: Optional[Dict[str, Callable]] = None
    ):
        if not isinstance(length, (int, np.integer)) or length <= 0:
            raise FedbiomedError(f"'length' must be positive integer, got {length}")
        self.buffer = np.full(length, np.nan, dtype=np.float32)
        self.stat_functions = stat_functions or {}
        self._next_index = 0

    def __len__(self):
        return len(self.buffer)

    def update(self, val: Union[float, int]) -> None:
        if self._next_index >= len(self):
            raise FedbiomedError(f"Buffer full (size {len(self)})")
        self.buffer[self._next_index] = float(val)
        self._next_index += 1

    def set_stat_functions(self, stat_functions: Dict[str, Callable]) -> None:
        if not isinstance(stat_functions, dict):
            raise FedbiomedError(
                f"'stat_functions' must be dict, got {type(stat_functions)}"
            )
        for name, func in stat_functions.items():
            if not callable(func):
                raise FedbiomedError(f"Stat '{name}' is not callable: {type(func)}")
        self.stat_functions.update(stat_functions)

    def finalize(self) -> Dict[str, Any]:
        data = self.buffer[np.isfinite(self.buffer)]
        if len(data) == 0:
            return {name: np.nan for name in self.stat_functions}

        results = {}
        for name, func in self.stat_functions.items():
            try:
                results[name] = func(data)
            except Exception as e:
                raise FedbiomedError(f"Error computing stat '{name}': {e}") from e
        return results


# =============================================================================
# BASE CLASS
# =============================================================================


class BaseStatAccumulator(Accumulator):
    """Base for element-wise accumulators with shape validation."""

    def __init__(self):
        self._shape: Optional[tuple] = None

    def _validate_shape(self, val: np.ndarray) -> None:
        """Validate shape consistency."""
        if self._shape is None:
            self._shape = val.shape
        elif val.shape != self._shape:
            raise FedbiomedError(
                f"Shape mismatch: expected {self._shape}, got {val.shape}"
            )


# =============================================================================
# SIMPLE STATS
# =============================================================================


class CountAccumulator(BaseStatAccumulator):
    """Count non-NaN values element-wise."""

    def __init__(self):
        super().__init__()
        self.counts: Optional[np.ndarray] = None

    def update(self, val: np.ndarray) -> None:
        self._validate_shape(val)
        increment = np.isfinite(val).astype(np.int32)
        self.counts = increment if self.counts is None else self.counts + increment

    def finalize(self) -> Dict[str, Any]:
        return {"count": self.counts if self.counts is not None else 0}


class MinAccumulator(BaseStatAccumulator):
    """Compute minimum element-wise (ignores NaN, inf, -inf)."""

    def __init__(self):
        super().__init__()
        self.min_val: Optional[np.ndarray] = None

    def update(self, val: np.ndarray) -> None:
        self._validate_shape(val)
        if self.min_val is None:
            # Initialize: replace non-finite with +inf so any real value will be smaller
            self.min_val = np.where(np.isfinite(val), val, np.inf).astype(np.float32)
        else:
            self.min_val = np.fmin(self.min_val, val)

    def finalize(self) -> Dict[str, Any]:
        return {"min": self.min_val}


class MaxAccumulator(BaseStatAccumulator):
    """Compute maximum element-wise (ignores NaN, inf, -inf)."""

    def __init__(self):
        super().__init__()
        self.max_val: Optional[np.ndarray] = None

    def update(self, val: np.ndarray) -> None:
        self._validate_shape(val)
        if self.max_val is None:
            # Initialize: replace non-finite with -inf so any real value will be larger
            self.max_val = np.where(np.isfinite(val), val, -np.inf).astype(np.float32)
        else:
            self.max_val = np.fmax(self.max_val, val)

    def finalize(self) -> Dict[str, Any]:
        return {"max": self.max_val}


# =============================================================================
# COMPLEX STATS
# =============================================================================


class MeanAccumulator(BaseStatAccumulator):
    """Compute mean element-wise. Returns mean and count."""

    def __init__(self):
        super().__init__()
        self.sum_val: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None

    def update(self, val: np.ndarray) -> None:
        self._validate_shape(val)

        # Sum (replace non-finite with 0, initialize as float32)
        zeroed = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.sum_val = zeroed if self.sum_val is None else self.sum_val + zeroed

        # Count (finite values only)
        increment = np.isfinite(val).astype(np.int32)
        self.counts = increment if self.counts is None else self.counts + increment

    def finalize(self) -> Dict[str, Any]:
        if self.sum_val is None or self.counts is None:
            return {"mean": np.nan, "count": 0}
        mean = np.divide(
            self.sum_val,
            self.counts,
            out=np.full_like(self.sum_val, np.nan, dtype=np.float32),
            where=self.counts > 0,
        )
        return {"mean": mean, "count": self.counts}


class VarianceAccumulator(BaseStatAccumulator):
    """Compute variance element-wise using Welford's algorithm. Returns variance, mean, count."""

    def __init__(self):
        super().__init__()
        self.mean_val: Optional[np.ndarray] = None
        self.m2_val: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None

    def update(self, val: np.ndarray) -> None:
        self._validate_shape(val)
        mask = np.isfinite(val)

        if self.counts is None:
            self.counts = mask.astype(np.int32)
            self.mean_val = np.where(mask, val, 0.0).astype(np.float32)
            self.m2_val = np.zeros(val.shape, dtype=np.float32)
        else:
            delta = np.where(mask, val - self.mean_val, 0.0)
            self.counts += mask.astype(np.int32)
            mean_update = np.divide(
                delta,
                self.counts,
                out=np.zeros_like(delta),
                where=self.counts > 0,
            )
            self.mean_val = np.where(mask, self.mean_val + mean_update, self.mean_val)
            delta2 = np.where(mask, val - self.mean_val, 0.0)
            self.m2_val = np.where(mask, self.m2_val + delta * delta2, self.m2_val)

    def finalize(self) -> Dict[str, Any]:
        if self.counts is None or self.mean_val is None or self.m2_val is None:
            return {"variance": np.nan, "mean": np.nan, "count": 0}
        mean = np.where(self.counts > 0, self.mean_val, np.nan)
        variance = np.divide(
            self.m2_val,
            self.counts - 1,
            out=np.full_like(self.m2_val, np.nan, dtype=np.float32),
            where=self.counts > 1,
        )
        return {"variance": variance, "mean": mean, "count": self.counts}


# =============================================================================
# NOT VECTOR-AGGREGATED STATS
# =============================================================================


class HistogramAccumulator(Accumulator):
    """Streaming histogram accumulator for scalar values.

    Accumulates counts per bin given pre-specified bin edges. Using explicit
    bin_edges ensures consistent boundaries across federated nodes, which is
    required for histogram aggregation.

    Args:
        bin_edges: Monotonically increasing sequence defining bin boundaries.
            Must define at least _MIN_BINS bins (len >= _MIN_BINS + 1).
    """

    _MIN_BINS: int = 2

    def __init__(self, bin_edges: List[float]):
        edges = np.asarray(bin_edges, dtype=np.float32)

        # Validate bin edges: must be 1D, have at least _MIN_BINS bins, and be strictly increasing
        if edges.ndim != 1 or len(edges) <= self._MIN_BINS:
            raise FedbiomedError(
                f"'bin_edges' must define at least {self._MIN_BINS} bins"
            )
        if not np.all(np.diff(edges) > 0):
            raise FedbiomedError("'bin_edges' must be strictly increasing")

        self._bin_edges = edges
        self._counts = np.zeros(len(edges) - 1, dtype=np.int32)

    def update(self, value: Union[float, int]) -> None:
        v = float(value)

        # Validate value: must be finite and within bin edges
        if not np.isfinite(v):
            return
        if v < self._bin_edges[0] or v > self._bin_edges[-1]:
            return

        # Indexing: find the right bin for v and increment count
        idx = np.searchsorted(self._bin_edges, v, side="right") - 1
        if idx == len(self._counts):  # Handle rightmost edge explicitly
            idx -= 1
        self._counts[idx] += 1

    def finalize(self) -> Dict[str, Any]:
        return {
            "bin_edges": self._bin_edges.tolist(),
            "counts": self._counts.tolist(),
        }


# =============================================================================
# BUFFER-BACKED STATS
# =============================================================================


class QuantileAccumulator(Accumulator):
    """Quantile accumulator backed by ScalarBuffer.

    Buffers all scalar values and computes requested quantiles on finalization.
    Uses the (0, 1] convention (e.g. 0.5 = median).

    Args:
        quantiles: Non-empty list of distinct quantile levels in (0, 1].
        buffer_size: Maximum number of samples to store (typically len(dataset)).
    """

    def __init__(self, quantiles: List[float], buffer_size: int):
        if not quantiles:
            raise FedbiomedError("'quantiles' must not be empty")
        for q in quantiles:
            if not isinstance(q, (int, float)) or not (0 < q <= 1):
                raise FedbiomedError(f"Each quantile must be in (0, 1], got {q!r}")
        if len(set(quantiles)) != len(quantiles):
            raise FedbiomedError("'quantiles' must not contain duplicates")

        # Keep sorted list for deterministic key order when finalizing
        self._quantiles = sorted(quantiles)
        # scalar buffer simply stores values; quantile computation is handled
        # in finalize so we only need a plain buffer without stat functions.
        self._buffer = ScalarBuffer(length=buffer_size)

    def update(self, val: Union[float, int]) -> None:
        self._buffer.update(val)

    def finalize(self) -> Dict[str, Any]:
        # compute quantiles in a single numpy call to avoid repeated sorting
        data = self._buffer.buffer[np.isfinite(self._buffer.buffer)]
        if len(data) == 0:
            return {f"q_{q}": np.nan for q in self._quantiles}

        # numpy.quantile accepts a sequence of q values and returns all at once
        qvals = np.quantile(data, self._quantiles)
        # ensure all returned values are plain floats
        return {f"q_{q}": float(v) for q, v in zip(self._quantiles, qvals, strict=True)}


# =============================================================================
# IMAGE-SPECIFIC STATS
# =============================================================================


class ImageShapeAccumulator(Accumulator):
    """Accumulates the shape of images into a dictionary of tuples."""

    def __init__(self):
        self._shapes = {}

    def update(self, image: np.ndarray) -> None:
        """Update the accumulator with the shape of the given image.

        Args:
            image: Image as a numpy array of arbitrary shape
        """
        shape = tuple(image.shape)
        self._shapes[shape] = self._shapes.get(shape, 0) + 1

    def finalize(self) -> Dict[str, Any]:
        """Return the accumulated image shapes and their counts.

        Returns:
            A dictionary where keys are image shapes (as tuples) and values are
            the counts of images with those shapes.
        """
        return self._shapes


class ImageBaseAccumulator(Accumulator):
    """Abstract base for image statistics that reduce each image to a scalar."""

    _DESCRIBE_FUNCTIONS: Dict[str, Callable] = {
        "count": len,
        "mean": np.mean,
        "std": np.std,
        "q05": lambda data: np.quantile(data, 0.05),
        "q25": lambda data: np.quantile(data, 0.25),
        "q50": lambda data: np.quantile(data, 0.50),
        "q75": lambda data: np.quantile(data, 0.75),
        "q95": lambda data: np.quantile(data, 0.95),
    }

    def __init__(self, buffer_size: int):
        self._buffer = ScalarBuffer(buffer_size, self._DESCRIBE_FUNCTIONS)

    @abstractmethod
    def reduce(self, image: np.ndarray) -> float:
        pass

    def update(self, image: np.ndarray) -> None:
        try:
            self._buffer.update(self.reduce(image))
        except Exception as e:
            raise FedbiomedError(f"Error reducing image: {e}") from e

    def finalize(self) -> Dict[str, Any]:
        """Return descriptive statistics over all stored per-image scalars."""
        try:
            return self._buffer.finalize()
        except Exception as e:
            raise FedbiomedError(f"Error finalizing image statistics: {e}") from e


class ImageMeanAccumulator(ImageBaseAccumulator):
    """Accumulates the per-image pixel mean (``np.nanmean``)."""

    def reduce(self, image: np.ndarray) -> float:
        return float(np.nanmean(image))


class ImageVarianceAccumulator(ImageBaseAccumulator):
    """Accumulates the per-image pixel variance (``np.nanvar``)."""

    def reduce(self, image: np.ndarray) -> float:
        return float(np.nanvar(image))

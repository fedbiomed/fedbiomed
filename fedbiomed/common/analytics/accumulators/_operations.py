# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Statistic accumulators for federated analytics."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

from fedbiomed.common.constants import FedbiomedError

from ._base import Accumulator


class ScalarAccumulator(Accumulator):
    """Base marker for accumulators that consume scalar values."""

    @abstractmethod
    def finalize(self) -> Dict[str, Any]:
        pass


class ArrayAccumulator(Accumulator):
    """Base for element-wise array accumulators with shape validation.

    Subclasses must define:
        _key: output key name in finalize()
        _transform(): maps input array to the value to accumulate
    """

    _key: str
    _default: Any = np.nan

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Ensure subclasses define required class attribute
        if not hasattr(cls, "_key"):
            raise TypeError(f"{cls.__name__} must define class attribute '_key'")

    def __init__(self):
        self._shape: Optional[tuple] = None
        self._value: Optional[np.ndarray] = None

    def _validate_shape(self, val: np.ndarray) -> None:
        """Validate that incoming array has consistent shape across updates."""
        if self._shape is None:
            self._shape = val.shape
        elif val.shape != self._shape:
            raise FedbiomedError(
                f"Shape mismatch: expected {self._shape}, got {val.shape}"
            )

    @abstractmethod
    def _transform(self, val: np.ndarray) -> np.ndarray:
        """Transform input array to the value to accumulate."""
        pass

    def update(self, val: np.ndarray) -> None:
        self._validate_shape(val)
        t = self._transform(val)
        self._value = t if self._value is None else self._value + t

    def finalize(self) -> Dict[str, Any]:
        return {self._key: self._value if self._value is not None else self._default}


class CountAccumulator(ArrayAccumulator):
    """Count non-NaN values element-wise."""

    _key = "count"
    _default = 0

    def _transform(self, val: np.ndarray) -> np.ndarray:
        return np.isfinite(val).astype(np.int32)


class SumAccumulator(ArrayAccumulator):
    """Accumulate sum element-wise."""

    _key = "sum"

    def _transform(self, val: np.ndarray) -> np.ndarray:
        return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


class SumSqAccumulator(ArrayAccumulator):
    """Accumulate sum of squares element-wise."""

    _key = "sum_sq"

    def _transform(self, val: np.ndarray) -> np.ndarray:
        return (
            np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64) ** 2
        )


class HistogramAccumulator(ScalarAccumulator):
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
        if not np.isfinite(v):
            return
        if v < self._bin_edges[0] or v > self._bin_edges[-1]:
            return
        idx = np.searchsorted(self._bin_edges, v, side="right") - 1
        if idx == len(self._counts):
            idx -= 1
        self._counts[idx] += 1

    def finalize(self) -> Dict[str, Any]:
        return {
            "histogram": {
                "bin_edges": self._bin_edges.tolist(),
                "counts": self._counts.tolist(),
            },
        }

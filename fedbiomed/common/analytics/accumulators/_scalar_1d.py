# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Scalar and 1D streaming accumulators: Count, Min, Max, Mean, Variance, ScalarBuffer."""

from typing import Any, Callable, Dict, Optional, Union

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
# SIMPLE PRIMITIVES
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
# COMPLEX PRIMITIVES
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

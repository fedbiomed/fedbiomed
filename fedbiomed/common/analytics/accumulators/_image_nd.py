# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Image (N-D) streaming accumulator.

This module provides an accumulator tailored for image-like NumPy arrays
(`numpy.ndarray`). It computes basic descriptive statistics over a stream
of images:

- min
- max
- mean
- std
- count

Implementation notes
--------------------
- The default computation is element-wise with respect to the provided image shape.
    I.e. for images of shape (H, W, C), the resulting statistics are arrays with
    the same shape unless channel aggregation is enabled.
- Optional `aggregate_channels` support is implemented for *per-pixel outputs*
    by treating the last dimension as the channel axis and updating the underlying
    accumulators once per channel slice, effectively aggregating across channels
    and samples.
- In addition to per-pixel outputs, this accumulator also computes *per-channel*
    global summaries (scalar per channel) across all pixels and all samples.

The accumulator reuses the scalar/1D accumulators defined in `_scalar_1d.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from fedbiomed.common.constants import FedbiomedError

from ._base import Accumulator
from ._scalar_1d import (
    CountAccumulator,
    MaxAccumulator,
    MeanAccumulator,
    MinAccumulator,
    VarianceAccumulator,
)


class ImageNDAccumulator(Accumulator):
    """Streaming accumulator for NumPy image arrays.

    Args:
        config: Either an orchestrator-style configuration dict with a top-level
            "stats" field (e.g. {"type": IMAGE, "stats": {...}}) or directly a
            mapping of stat name to its argument dictionary.

            Supported stat args:
                - aggregate_channels (bool): if True, treat the last axis as
                  channels and aggregate statistics across channels.

    Raises:
        FedbiomedError: If config is invalid.
    """

    _SUPPORTED_STATS = {"min", "max", "mean", "variance", "std", "count"}

    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise FedbiomedError(
                f"ImageNDAccumulator expects config dict, got {type(config)}"
            )

        stats_cfg = config.get("stats", config)
        if not isinstance(stats_cfg, dict):
            raise FedbiomedError(
                f"ImageNDAccumulator expects 'stats' dict, got {type(stats_cfg)}"
            )

        # If no stats were provided, default to the basic image summary stats.
        if not stats_cfg:
            stats_cfg = {"min": {}, "max": {}, "mean": {}, "std": {}, "count": {}}

        # Normalize to {stat_name: args_dict}
        self._stats_cfg: Dict[str, Dict[str, Any]] = {}
        for stat_name, args in stats_cfg.items():
            if stat_name not in self._SUPPORTED_STATS:
                # Keep the implementation strict: this is a dedicated image ND accumulator.
                raise FedbiomedError(f"Unsupported image stat '{stat_name}'")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise FedbiomedError(
                    f"Args for stat '{stat_name}' must be dict, got {type(args)}"
                )
            self._stats_cfg[stat_name] = args

        # Instantiate the underlying primitive accumulators.
        # We always compute the full requested set, and ensure we have what's needed
        # to produce std (variance).
        self._acc_min = MinAccumulator() if self._wants("min") else None
        self._acc_max = MaxAccumulator() if self._wants("max") else None
        self._acc_count = CountAccumulator() if self._wants("count") else None
        self._acc_mean = MeanAccumulator() if self._wants("mean") else None

        wants_variance = self._wants("variance") or self._wants("std")
        self._acc_variance = VarianceAccumulator() if wants_variance else None

        # Cache per-stat aggregate_channels to avoid repeated dict lookups.
        self._aggregate_channels: Dict[str, bool] = {
            stat: bool(args.get("aggregate_channels", False))
            for stat, args in self._stats_cfg.items()
        }

        # --- Per-channel global summaries (scalar per channel) ---
        # These are computed across *all pixels* and *all samples*.
        # Channels are assumed to be the last axis when ndim >= 3.
        self._channels: Optional[int] = None
        self._ch_count: Optional[np.ndarray] = None  # int64
        self._ch_sum: Optional[np.ndarray] = None  # float64
        self._ch_sumsq: Optional[np.ndarray] = None  # float64
        self._ch_min: Optional[np.ndarray] = None  # float32
        self._ch_max: Optional[np.ndarray] = None  # float32

    def _wants(self, stat_name: str) -> bool:
        return stat_name in self._stats_cfg

    @staticmethod
    def _as_image_array(value: Any) -> np.ndarray:
        """Convert input to a NumPy array suitable for stats computation."""
        try:
            arr = np.asarray(value)
        except Exception as e:
            raise FedbiomedError(
                f"ImageNDAccumulator could not convert value to array: {e}"
            ) from e

        if not isinstance(arr, np.ndarray):
            raise FedbiomedError(
                f"ImageNDAccumulator expected numpy.ndarray, got {type(arr)}"
            )
        if arr.size == 0:
            raise FedbiomedError("ImageNDAccumulator does not support empty arrays")
        if not np.issubdtype(arr.dtype, np.number):
            raise FedbiomedError(
                f"ImageNDAccumulator expects numeric dtype, got {arr.dtype}"
            )

        # Use float32 for numeric stability / consistency with other accumulators.
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _iter_channel_slices(arr: np.ndarray) -> Optional[Tuple[np.ndarray, ...]]:
        """Returns channel slices if array looks like channel-last, else None."""
        if arr.ndim < 3:
            return None
        # Channel axis is assumed to be the last axis.
        # We treat each channel slice as one additional update.
        return tuple(arr[..., c] for c in range(arr.shape[-1]))

    def _ensure_channel_state(self, arr: np.ndarray) -> Tuple[int, bool]:
        """Initialize per-channel state and return (n_channels, has_channel_axis)."""
        has_channel_axis = arr.ndim >= 3
        n_channels = int(arr.shape[-1]) if has_channel_axis else 1

        if self._channels is None:
            self._channels = n_channels
            self._ch_count = np.zeros(n_channels, dtype=np.int64)
            self._ch_sum = np.zeros(n_channels, dtype=np.float64)
            self._ch_sumsq = np.zeros(n_channels, dtype=np.float64)
            self._ch_min = np.full(n_channels, np.inf, dtype=np.float32)
            self._ch_max = np.full(n_channels, -np.inf, dtype=np.float32)
        elif n_channels != self._channels:
            raise FedbiomedError(
                f"Channel count mismatch: expected {self._channels}, got {n_channels}"
            )

        return n_channels, has_channel_axis

    def _update_per_channel(self, arr: np.ndarray) -> None:
        """Update per-channel global summaries using raw pixel values."""
        if not (
            self._wants("min")
            or self._wants("max")
            or self._wants("mean")
            or self._wants("std")
            or self._wants("variance")
            or self._wants("count")
        ):
            return

        n_channels, has_channel_axis = self._ensure_channel_state(arr)

        # Scalar (single-channel) case: aggregate over all pixels.
        if not has_channel_axis:
            finite = np.isfinite(arr)
            if not np.any(finite):
                return
            vals = arr[finite].astype(np.float64, copy=False)
            self._ch_count[0] += int(vals.size)
            self._ch_sum[0] += float(np.sum(vals))
            self._ch_sumsq[0] += float(np.sum(vals * vals))
            self._ch_min[0] = float(min(self._ch_min[0], float(np.min(vals))))
            self._ch_max[0] = float(max(self._ch_max[0], float(np.max(vals))))
            return

        # Multi-channel case (channel-last).
        for c in range(n_channels):
            slc = arr[..., c]
            finite = np.isfinite(slc)
            if not np.any(finite):
                continue
            vals = slc[finite].astype(np.float64, copy=False)
            self._ch_count[c] += int(vals.size)
            self._ch_sum[c] += float(np.sum(vals))
            self._ch_sumsq[c] += float(np.sum(vals * vals))
            self._ch_min[c] = float(min(self._ch_min[c], float(np.min(vals))))
            self._ch_max[c] = float(max(self._ch_max[c], float(np.max(vals))))

    def _update_one(
        self, acc: Accumulator, arr: np.ndarray, *, aggregate_channels: bool
    ) -> None:
        if not aggregate_channels:
            acc.update(arr)
            return

        channel_slices = self._iter_channel_slices(arr)
        if channel_slices is None:
            acc.update(arr)
            return

        for slc in channel_slices:
            acc.update(slc)

    def update(self, value: Any) -> None:
        """Update all requested running statistics with a new image."""
        arr = self._as_image_array(value)

        # Per-channel global summaries (scalar per channel)
        self._update_per_channel(arr)

        if self._acc_min is not None:
            self._update_one(
                self._acc_min,
                arr,
                aggregate_channels=self._aggregate_channels.get("min", False),
            )
        if self._acc_max is not None:
            self._update_one(
                self._acc_max,
                arr,
                aggregate_channels=self._aggregate_channels.get("max", False),
            )
        if self._acc_count is not None:
            self._update_one(
                self._acc_count,
                arr,
                aggregate_channels=self._aggregate_channels.get("count", False),
            )
        if self._acc_mean is not None:
            self._update_one(
                self._acc_mean,
                arr,
                aggregate_channels=self._aggregate_channels.get("mean", False),
            )
        if self._acc_variance is not None:
            # std uses the variance accumulator; prefer std args if provided.
            agg = self._aggregate_channels.get(
                "std", self._aggregate_channels.get("variance", False)
            )
            self._update_one(self._acc_variance, arr, aggregate_channels=agg)

    @staticmethod
    def _extract(result: Dict[str, Any], key: str) -> Any:
        return result.get(key, np.nan)

    def finalize(self) -> Dict[str, Any]:
        """Finalize and return computed statistics."""
        results: Dict[str, Any] = {}

        if self._acc_min is not None:
            results["min"] = self._extract(self._acc_min.finalize(), "min")
        if self._acc_max is not None:
            results["max"] = self._extract(self._acc_max.finalize(), "max")
        if self._acc_count is not None:
            results["count"] = self._extract(self._acc_count.finalize(), "count")
        if self._acc_mean is not None:
            results["mean"] = self._extract(self._acc_mean.finalize(), "mean")

        if self._acc_variance is not None:
            var = self._extract(self._acc_variance.finalize(), "variance")
            if self._wants("variance"):
                results["variance"] = var
            if self._wants("std"):
                # std is derived from variance; preserve NaNs.
                results["std"] = np.sqrt(var)

        # Per-channel outputs are returned under a dedicated key.
        if self._ch_count is not None:
            ch_res: Dict[str, Any] = {}

            # count is number of finite pixel values aggregated per channel
            if self._wants("count"):
                ch_res["count"] = (
                    int(self._ch_count[0])
                    if self._channels == 1
                    else self._ch_count.astype(np.int64)
                )

            if self._wants("min"):
                ch_min = np.where(self._ch_count > 0, self._ch_min, np.nan)
                ch_res["min"] = float(ch_min[0]) if self._channels == 1 else ch_min

            if self._wants("max"):
                ch_max = np.where(self._ch_count > 0, self._ch_max, np.nan)
                ch_res["max"] = float(ch_max[0]) if self._channels == 1 else ch_max

            if self._wants("mean") or self._wants("variance") or self._wants("std"):
                mean = np.divide(
                    self._ch_sum,
                    self._ch_count,
                    out=np.full_like(self._ch_sum, np.nan, dtype=np.float64),
                    where=self._ch_count > 0,
                )

                if self._wants("mean"):
                    ch_res["mean"] = float(mean[0]) if self._channels == 1 else mean

                if self._wants("variance") or self._wants("std"):
                    # Sample variance (ddof=1) to match VarianceAccumulator behavior.
                    var_num = self._ch_sumsq - (self._ch_sum * self._ch_sum) / np.where(
                        self._ch_count > 0, self._ch_count, 1
                    )
                    var = np.divide(
                        var_num,
                        (self._ch_count - 1),
                        out=np.full_like(var_num, np.nan, dtype=np.float64),
                        where=self._ch_count > 1,
                    )

                    if self._wants("variance"):
                        ch_res["variance"] = (
                            float(var[0]) if self._channels == 1 else var
                        )
                    if self._wants("std"):
                        std = np.sqrt(var)
                        ch_res["std"] = float(std[0]) if self._channels == 1 else std

            results["per_channel"] = ch_res

        return results

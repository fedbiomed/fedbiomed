# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Image (N-D) streaming accumulator.

This module provides an accumulator tailored for image-like NumPy arrays
(`numpy.ndarray`). It computes *global* descriptive statistics over a stream
of images: i.e. a single scalar per statistic for the whole dataset.

Computed statistics (when requested):

- min
- max
- mean
- variance
- std
- count

Notes
-----
- Statistics are computed across all pixel values of all images seen so far.
- Images containing any non-finite values (NaN/inf) are rejected.
- Image shape may vary across updates; this accumulator does not require a fixed shape.
- Optional `aggregate_channels` arguments may appear in configs (registry support) but
  are not relevant when returning global scalars; they are accepted and ignored.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from fedbiomed.common.constants import FedbiomedError

from ._base import Accumulator


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

        # Global running state (over all finite pixels across all images).
        self._count: int = 0
        self._sum: float = 0.0
        self._sumsq: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    def _wants(self, stat_name: str) -> bool:
        return stat_name in self._stats_cfg

    def update(self, value: np.ndarray) -> None:
        """Update running global statistics with a new image."""
        finite = np.isfinite(value)
        # Reject images that contain any non-finite values (NaN/inf), including empty arrays.
        if value.size == 0 or not np.all(finite):
            raise FedbiomedError(
                "ImageNDAccumulator Error: Image contains non-numeric values"
            )

        vals = value[finite].astype(np.float64, copy=False)
        n = int(vals.size)
        self._count += n

        # Always track sum and sumsq (cheap and needed for mean/variance/std).
        self._sum += float(np.sum(vals))
        self._sumsq += float(np.sum(vals * vals))

        # Track min/max.
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmin < self._min:
            self._min = vmin
        if vmax > self._max:
            self._max = vmax

    def finalize(self) -> Dict[str, Any]:
        """Finalize and return requested statistics as global scalars."""
        results: Dict[str, Any] = {}
        if self._count == 0 or self._count == 1:
            raise FedbiomedError(
                "ImageNDAccumulator Error: No valid pixel values were accumulated"
            )

        if self._wants("count"):
            results["count"] = int(self._count)
        if self._wants("min"):
            results["min"] = float(self._min)
        if self._wants("max"):
            results["max"] = float(self._max)

        if self._wants("mean"):
            mean = self._sum / self._count
            results["mean"] = float(mean)

        if self._wants("variance") or self._wants("std"):
            var_num = self._sumsq - (self._sum * self._sum) / self._count
            var = var_num / (self._count - 1)

            if self._wants("variance"):
                results["variance"] = float(var) if np.isfinite(var) else np.nan
            if self._wants("std"):
                std = np.sqrt(var) if np.isfinite(var) else np.nan
                results["std"] = float(std) if np.isfinite(std) else np.nan

        return results

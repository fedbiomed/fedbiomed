# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence, Union

import numpy as np

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError

from ._analytics_strategy import AnalyticsStrategy


class ImageAnalytics(AnalyticsStrategy):
    """Mixin class for computing analytics on image folder datasets"""

    def mean(self, **kwargs) -> Dict:
        return super().mean(**kwargs)

    def histogram(
        self,
        bin_edges: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """
        Compute histogram counts for pixel values across all images in the dataset.

        Args:
            bin_edges: Global bin edges broadcast by the researcher. Can be:
                - 1D numpy array: applied to all pixel values
                - Dict mapping column/feature names to bin_edges

        Returns:
            counts: counts is a numpy array of shape (B,) with histogram counts for all pixel values

        Raises:
            FedbiomedError: if bin_edges is not a 1D array of length >= 2
            FedbiomedError: if bin_edges are not strictly increasing
        """
        # If bin_edges is a dict, extract the single array
        if isinstance(bin_edges, dict):
            bin_edges = next(iter(bin_edges.values()))

        # Convert to numpy array if it's a list
        if isinstance(bin_edges, list):
            bin_edges = np.array(bin_edges)

        # Validate bin_edges
        if bin_edges.ndim != 1 or bin_edges.size < 2:
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: bin_edges must be a 1D array of length >= 2"
            )
        if not np.all(np.diff(bin_edges) > 0):
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: bin_edges must be strictly increasing"
            )

        # Initialize histogram counts
        num_bins = bin_edges.size - 1
        counts = np.zeros(num_bins, dtype=np.int64)

        # Single pass through dataset
        for data, _ in self:
            # Flatten pixel values
            pixel_values = data.flatten()

            # Convert to float for processing
            pixel_values = pixel_values.astype(np.float64)

            # Skip non-finite values
            pixel_values = pixel_values[np.isfinite(pixel_values)]

            if pixel_values.size > 0:
                clipped = np.clip(pixel_values, bin_edges[0], bin_edges[-1])
                counts += np.histogram(clipped, bins=bin_edges)[0]

        return {"pixel_values": counts}

    def min_max(self, **kwargs) -> Dict[str, tuple]:
        """Calculate min and max pixel values across all images in the dataset.

        Returns:
            Dictionary with key "pixel_values" containing (min, max) tuple

        Raises:
            FedbiomedError: if dataset is empty
        """

        # TODO: Check for min number of pixels on images.
        if len(self) == 0:
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: Cannot calculate min_max of an empty dataset."
            )

        pixel_min = float("inf")
        pixel_max = float("-inf")

        for data, _ in self:
            pixel_values = data.flatten()

            pixel_values = pixel_values.astype(np.float64)

            # Find min/max excluding non-finite values
            finite_values = pixel_values[np.isfinite(pixel_values)]
            if finite_values.size > 0:
                pixel_min = min(pixel_min, np.min(finite_values))
                pixel_max = max(pixel_max, np.max(finite_values))

        return {
            "pixel_values": (
                pixel_min if pixel_min != float("inf") else np.nan,
                pixel_max if pixel_max != float("-inf") else np.nan,
            )
        }

    def quantile(
        self,
        bin_edges: np.ndarray,
        q: Union[float, Sequence[float]],
    ) -> Dict[float, float]:
        """
        Compute quantiles from the histogram of pixel values.

        Args:
            bin_edges: Global bin edges. 1D array, length B+1.
            q: Quantile(s) to compute. Scalar in [0, 1] or sequence of values in [0, 1].
               0.5 = median, 0.25 = first quartile, 0.75 = third quartile, etc.

        Returns:
            Dictionary mapping quantile values to computed quantiles.
            e.g., {0.5: median_value} or {0.25: q1_value, 0.5: median_value, 0.75: q3_value}
            Linear interpolation is used within bins.

        Raises:
            FedbiomedError: if q values are not in [0, 1]
        """

        # If bin_edges is a dict, extract the single array
        if isinstance(bin_edges, dict):
            bin_edges = next(iter(bin_edges.values()))

        # Normalize q to array
        q_arr = np.atleast_1d(q)

        # Validate q values
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: Quantile values must be in [0, 1]"
            )

        # Get histogram counts
        counts = self.histogram(bin_edges).get("pixel_values")

        # Compute cumulative distribution
        total_count = np.sum(counts)
        if total_count == 0:
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: No data available for quantile computation"
            )

        cumsum = np.cumsum(counts)

        # Compute quantiles
        quantiles = np.zeros_like(q_arr, dtype=np.float64)
        bin_widths = np.diff(bin_edges)

        for i, quantile in enumerate(q_arr):
            # Find the target count
            target_count = quantile * total_count

            # Find the bin containing this quantile
            bin_idx = np.searchsorted(cumsum, target_count, side="left")
            bin_idx = np.clip(bin_idx, 0, len(counts) - 1)

            # Linear interpolation within the bin
            bin_left = bin_edges[bin_idx]
            bin_width = bin_widths[bin_idx]

            # Count in previous bins
            count_before = cumsum[bin_idx - 1] if bin_idx > 0 else 0

            # Fraction within current bin
            count_in_bin = counts[bin_idx]
            if count_in_bin > 0:
                fraction = (target_count - count_before) / count_in_bin
                fraction = np.clip(fraction, 0, 1)
            else:
                fraction = 0.5

            quantiles[i] = bin_left + fraction * bin_width

        # Return dictionary mapping quantile values to computed quantiles
        return {
            "pixel_values": {
                float(q_val): float(quant)
                for q_val, quant in zip(q_arr, quantiles, strict=True)
            }
        }

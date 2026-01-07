# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence, Union

import numpy as np
import torch

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

        # Validate bin_edges
        if bin_edges.ndim != 1 or bin_edges.size < 2:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: bin_edges must be a 1D array of length >= 2"
            )
        if not np.all(np.diff(bin_edges) > 0):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: bin_edges must be strictly increasing"
            )

        # Initialize histogram counts
        num_bins = bin_edges.size - 1
        counts = np.zeros(num_bins, dtype=np.int64)

        # Single pass through dataset
        for idx in range(len(self)):
            data, _ = self[idx]

            # Convert to numpy array if necessary
            if isinstance(data, torch.Tensor):
                pixel_values = data.detach().cpu().numpy().flatten()
            elif isinstance(data, np.ndarray):
                pixel_values = data.flatten()

            # Convert to float for processing
            pixel_values = pixel_values.astype(np.float64)

            # Skip non-finite values
            pixel_values = pixel_values[np.isfinite(pixel_values)]

            if pixel_values.size > 0:
                # Determine bin indices (like numpy.histogram)
                bin_indices = np.searchsorted(bin_edges, pixel_values, side="right") - 1
                # Clamp to valid range
                bin_indices = np.clip(bin_indices, 0, num_bins - 1)
                # Accumulate counts
                np.add.at(counts, bin_indices, 1)

        return {"pixel_values": counts}

    def minmax(self, **kwargs) -> Dict[str, tuple]:
        """Calculate min and max pixel values across all images in the dataset.

        Returns:
            Dictionary with key "pixel_values" containing (min, max) tuple

        Raises:
            FedbiomedError: if dataset is empty
        """
        if len(self) == 0:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Cannot calculate minmax of an empty dataset."
            )

        pixel_min = float("inf")
        pixel_max = float("-inf")

        for idx in range(len(self)):
            data, _ = self[idx]

            # Convert to numpy array if necessary
            if isinstance(data, torch.Tensor):
                pixel_values = data.detach().cpu().numpy().flatten()
            elif isinstance(data, np.ndarray):
                pixel_values = data.flatten()
            else:
                pixel_values = np.array(data).flatten()

            pixel_values = pixel_values.astype(np.float64)

            # Find min/max excluding non-finite values
            finite_values = pixel_values[np.isfinite(pixel_values)]
            if finite_values.size > 0:
                pixel_min = min(pixel_min, np.min(finite_values))
                pixel_max = max(pixel_max, np.max(finite_values))

        return {
            "pixel_values": (
                pixel_min if pixel_min != float("inf") else None,
                pixel_max if pixel_max != float("-inf") else None,
            )
        }

    def quantile(
        self,
        bin_edges: np.ndarray,
        q: Union[float, Sequence[float]],
    ) -> Union[float, np.ndarray]:
        """
        Compute quantiles from the histogram of pixel values.

        Args:
            bin_edges: Global bin edges. 1D array, length B+1.
            q: Quantile(s) to compute. Scalar in [0, 1] or sequence of values in [0, 1].
               0.5 = median, 0.25 = first quartile, 0.75 = third quartile, etc.

        Returns:
            Quantile value(s). Scalar if q is scalar, array if q is sequence.
            Linear interpolation is used within bins.

        Raises:
            FedbiomedError: if q values are not in [0, 1]
        """
        # Normalize q to array
        q_is_scalar = np.isscalar(q)
        q_arr = np.atleast_1d(q)

        # Validate q values
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Quantile values must be in [0, 1]"
            )

        # Get histogram counts
        counts = self.histogram(bin_edges)

        # Compute cumulative distribution
        total_count = np.sum(counts)
        if total_count == 0:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: No data available for quantile computation"
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

        # Return scalar if input was scalar
        return quantiles[0] if q_is_scalar else quantiles

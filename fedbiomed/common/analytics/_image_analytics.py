# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

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
        bin_edges: np.ndarray,
    ) -> np.ndarray:
        """
        Compute histogram counts for pixel values across all images in the dataset.

        Args:
            bin_edges: Global bin edges broadcast by the researcher. 1D array, length B+1.

        Returns:
            counts: counts is a numpy array of shape (B,) with histogram counts for all pixel values

        Raises:
            FedbiomedError: if bin_edges is not a 1D array of length >= 2
            FedbiomedError: if bin_edges are not strictly increasing
        """
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

        return counts

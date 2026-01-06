# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
from typing import Dict

import numpy as np

from ._analytics_strategy import AnalyticsStrategy


class TabularAnalytics(AnalyticsStrategy):
    """Mixin class for computing analytics on tabular datasets"""

    def mean(self, **kwargs) -> Dict:
        """Calculate mean of features across the dataset. Only considers numeric columns.
        Iterates through dataset items (tuples of `data:np.ndarrays` and `target=None`).

        Returns:
            Dictionary by column names with their respective calculated value.
        """
        data, _ = self[0]

        # Initialize accumulators
        data_sum = np.zeros_like(data, dtype=float)
        data_count = np.zeros_like(data, dtype=int)

        # Iterate through dataset to accumulate
        for idx in range(len(self)):
            data, _ = self[idx]
            mask = [isinstance(_, (int, float)) and np.isfinite(_) for _ in data]
            data_sum[mask] += data[mask].astype(float)  # Cast for sanity
            data_count += mask

        # Calculate means by dividing by count
        data_mean = data_sum / data_count

        # Build result dict by column names
        return {col: data_mean[i] for i, col in enumerate(self._input_columns)}

    def max(self, **kwargs) -> Dict:
        """Calculate max of features across the dataset. Only considers numeric columns.
        Iterates through dataset items (tuples of `data:np.ndarrays` and `target=None`).

        Returns:
            Dictionary by column names with their respective calculated value.
        """
        data, _ = self[0]

        # Initialize accumulators
        data_max = np.full_like(data, -np.inf, dtype=float)

        # Iterate through dataset to accumulate
        for idx in range(len(self)):
            data, _ = self[idx]
            mask = [isinstance(_, (int, float)) and np.isfinite(_) for _ in data]
            data_max[mask] = np.maximum(data_max[mask], data[mask].astype(float))

        # Build result dict by column names
        return {col: data_max[i] for i, col in enumerate(self._input_columns)}

    def min(self, **kwargs) -> Dict:
        """Calculate min of features across the dataset. Only considers numeric columns.
        Iterates through dataset items (tuples of `data:np.ndarrays` and `target=None`).

        Returns:
            Dictionary by column names with their respective calculated value.
        """
        data, _ = self[0]

        # Initialize accumulators
        data_min = np.full_like(data, np.inf, dtype=float)

        # Iterate through dataset to accumulate sums, mins, maxs, and counts
        for idx in range(len(self)):
            data, _ = self[idx]
            mask = [isinstance(_, (int, float)) and np.isfinite(_) for _ in data]
            data_min[mask] = np.minimum(data_min[mask], data[mask].astype(float))

        # Build result dict by column names
        return {col: data_min[i] for i, col in enumerate(self._input_columns)}

    def variance(self, **kwargs) -> Dict:
        """Calculate variance of features across the dataset. Only considers numeric columns.
        Iterates through dataset items (tuples of `data:np.ndarrays` and `target=None`).

        Returns:
            Dictionary by column names with their respective calculated value.
        """
        data, _ = self[0]

        # Get means first
        data_mean = np.array(list(self.mean().values()))

        # Initialize accumulators
        data_variance = np.zeros_like(data, dtype=float)
        data_count = np.zeros_like(data, dtype=int)

        # Iterate through dataset to accumulate sums, mins, maxs, and counts
        for idx in range(len(self)):
            data, _ = self[idx]
            mask = [isinstance(_, (int, float)) and np.isfinite(_) for _ in data]
            data_variance[mask] += np.square(data[mask].astype(float) - data_mean[mask])
            data_count += mask

        # Calculate means by dividing by count
        data_variance /= data_count

        # Build result dict by column names
        return {col: data_variance[i] for i, col in enumerate(self._input_columns)}

    def std(self, **kwargs) -> Dict:
        """Calculate std of features across the dataset. Only considers numeric columns.
        Iterates through dataset items (tuples of `data:np.ndarrays` and `target=None`).

        Returns:
            Dictionary by column names with their respective calculated value.
        """
        # Build result dict by column names from variance function
        return {col: np.sqrt(val) for col, val in self.variance().items()}

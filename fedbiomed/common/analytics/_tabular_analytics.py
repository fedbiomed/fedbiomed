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
        data_sum = None
        count = None

        # Iterate through dataset to accumulate
        for data, _ in self:
            if data_sum is None:
                # Initialize accumulators on first iteration
                data_sum = np.zeros_like(data, dtype=float)
                count = np.zeros_like(data, dtype=int)

            mask = [isinstance(val, (int, float)) and np.isfinite(val) for val in data]
            data_sum[mask] += data[mask].astype(float)
            count[mask] += 1

        # Calculate means by dividing by count
        data_mean = np.divide(
            data_sum, count, out=np.full_like(data_sum, np.nan), where=count != 0
        )

        # Build result dict by column names
        return {col: data_mean[i] for i, col in enumerate(self._input_columns)}

    def max(self, **kwargs) -> Dict:
        """Calculate max of features across the dataset. Only considers numeric columns.
        Iterates through dataset items (tuples of `data:np.ndarrays` and `target=None`).

        Returns:
            Dictionary by column names with their respective calculated value.
        """
        data_max = None

        # Iterate through dataset to accumulate
        for data, _ in self:
            if data_max is None:
                # Initialize accumulators on first iteration
                data_max = np.full_like(data, -np.inf, dtype=float)

            mask = [isinstance(val, (int, float)) and np.isfinite(val) for val in data]
            data_max[mask] = np.maximum(data_max[mask], data[mask].astype(float))

        # Build result dict by column names
        return {col: data_max[i] for i, col in enumerate(self._input_columns)}

    def min(self, **kwargs) -> Dict:
        """Calculate min of features across the dataset. Only considers numeric columns.
        Iterates through dataset items (tuples of `data:np.ndarrays` and `target=None`).

        Returns:
            Dictionary by column names with their respective calculated value.
        """
        data_min = None

        # Iterate through dataset to accumulate
        for data, _ in self:
            if data_min is None:
                # Initialize accumulators on first iteration
                data_min = np.full_like(data, np.inf, dtype=float)

            mask = [isinstance(val, (int, float)) and np.isfinite(val) for val in data]
            data_min[mask] = np.minimum(data_min[mask], data[mask].astype(float))

        # Build result dict by column names
        return {col: data_min[i] for i, col in enumerate(self._input_columns)}

    def variance(self, **kwargs) -> Dict:
        """Calculate variance of features across the dataset. Only considers numeric columns.
        Uses Welford's online algorithm for numerical stability and single-pass efficiency.
        Iterates through dataset items (tuples of `data:np.ndarrays` and `target=None`).

        Returns:
            Dictionary by column names with their respective calculated value.
        """
        mean = None
        m2 = None
        count = None

        # Welford's online algorithm for variance in a single pass
        for data, _ in self:
            if mean is None:
                # Initialize accumulators on first iteration
                mean = np.zeros_like(data, dtype=float)
                m2 = np.zeros_like(data, dtype=float)
                count = np.zeros_like(data, dtype=int)

            mask = [isinstance(val, (int, float)) and np.isfinite(val) for val in data]
            data_float = data[mask].astype(float)
            count[mask] += 1
            delta = data_float - mean[mask]
            mean[mask] += delta / count[mask]
            delta2 = data_float - mean[mask]
            m2[mask] += delta * delta2

        # Calculate variance, using nan for zero counts
        data_variance = np.divide(
            m2, count, out=np.full_like(m2, np.nan), where=count != 0
        )

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

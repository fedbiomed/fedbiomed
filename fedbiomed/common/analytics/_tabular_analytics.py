# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
from typing import Dict

import numpy as np

from ._analytics_strategy import AnalyticsStrategy


class TabularAnalytics(AnalyticsStrategy):
    """Mixin class for computing analytics on tabular datasets"""

    def basic_stats(self, only_min_max: bool = False) -> Dict:
        """Calculate statistics across the dataset.

        Args:
            only_min_max: If True, only calculate min and max. If False, calculate min, max, count, mean, std.

        Returns:
            Dictionary with structure: {variable_name: {statistic_name: value}}
        """
        data_min = None
        data_max = None
        count = None
        mean = None
        m2 = None

        for data, _ in self:
            # Mask generation handles mixed types or NaNs
            mask = [
                isinstance(val, (int, float, np.number)) and np.isfinite(val)
                for val in data
            ]
            masked_data = data[mask].astype(float)

            # Initialize accumulators on first presence of data (assuming consistent shape)
            if data_min is None:
                data_min = np.full_like(data, np.inf, dtype=float)
                data_max = np.full_like(data, -np.inf, dtype=float)
                if not only_min_max:
                    count = np.zeros_like(data, dtype=int)
                    mean = np.zeros_like(data, dtype=float)
                    m2 = np.zeros_like(data, dtype=float)

            # Update min/max
            data_min[mask] = np.minimum(data_min[mask], masked_data)
            data_max[mask] = np.maximum(data_max[mask], masked_data)
            if not only_min_max:
                count[mask] += 1
                # Welford's online algorithm for mean/variance
                delta = masked_data - mean[mask]
                mean[mask] += delta / count[mask]
                delta2 = masked_data - mean[mask]
                m2[mask] += delta * delta2

        # Prepare final values
        results = {}

        if data_min is None:  # No data processed
            return results

        for i, col in enumerate(self._input_columns):
            stats = {
                "min": data_min[i] if np.isfinite(data_min[i]) else np.nan,
                "max": data_max[i] if np.isfinite(data_max[i]) else np.nan,
            }
            if not only_min_max:
                stats["count"] = count[i] if count[i] > 0 else np.nan
                stats["mean"] = mean[i] if count[i] > 0 else np.nan
                stats["std"] = np.sqrt(m2[i] / count[i]) if count[i] > 0 else np.nan

            results[col] = stats

        return results

    def min_max(self) -> Dict:
        """Returns min and max across the dataset."""
        return self.basic_stats(only_min_max=True)

    def mean(self) -> Dict:
        """Returns mean across the dataset."""
        stats = self.basic_stats()
        return {col: stat["mean"] for col, stat in stats.items()}

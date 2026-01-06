# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Union

import numpy as np

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError

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

<<<<<<< HEAD
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
=======
        Returns:
            Dictionary by column names with their respective calculated value.
        """
        # Build result dict by column names from variance function
        return {col: np.sqrt(val) for col, val in self.variance().items()}
    
    
    def histogram(
        self,
        bin_edges: Union[np.ndarray, Dict[Union[str, int], np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-feature histogram counts for a tabular dataset on a node.

        Args:
            bin_edges:
                Global bin edges broadcast by the researcher. Can be either:
                - 1D numpy array (length B+1): applied to all columns
                - Dict mapping column names to bin edges: column-specific bins

        Returns:
            dict: {column_name: counts}
                counts is a numpy array of shape (B,) for each column

        Raises:
            FedbiomedError: if bin_edges are invalid or missing for any column
        """

        # Validate and prepare bin_edges for each column
        col_bin_edges = {}
        for col_idx, col in enumerate(self._input_columns):
            if isinstance(bin_edges, dict):
                # Try to get edges by column name first, then by index
                edges = bin_edges.get(col) or bin_edges.get(col_idx)
                if edges is None:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Column '{col}' (index {col_idx}) not found in bin_edges dict"
                    )
            else:
                edges = bin_edges

            # Validate bin_edges
            if edges.ndim != 1 or edges.size < 2:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value} bin_edges must be a 1D array of length >= 2"
                )
            if not np.all(np.diff(edges) > 0):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value} bin_edges must be strictly increasing"
                )

            col_bin_edges[col] = edges

        # Initialize result counts for each column
        result: Dict[str, np.ndarray] = {
            col: np.zeros(edges.size - 1, dtype=np.int64)
            for col, edges in col_bin_edges.items()
        }

        # Single pass through dataset
        for idx in range(len(self)):
            data, _ = self[idx]

            # Process each column (for each sample)
            for i, col in enumerate(self._input_columns):
                value = data[i]

                # Skip non-finite values
                if not np.isfinite(value):
                    continue

                edges = col_bin_edges[col]
                # Determine bin index: rightmost edge <= value minus 1 (like numpy.histogram)
                bin_idx = np.searchsorted(edges, value, side="right") - 1
                # Clamp to valid range
                bin_idx = np.clip(bin_idx, 0, edges.size - 2)
                result[col][bin_idx] += 1

        return result
>>>>>>> 0ddec5e6 (Initial draft for Histogram Logic for Tabular Dataset)

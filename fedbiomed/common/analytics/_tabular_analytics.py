# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence, Union

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

        print("Computing histogram for database with columns:", self._input_columns)

        # Validate and prepare bin_edges for each column
        col_bin_edges = {}
        for col_idx, col in enumerate(self._input_columns):
            if isinstance(bin_edges, dict):
                # Try to get edges by column name first, then by index
                if col in bin_edges:
                    edges = bin_edges[col]
                elif col_idx in bin_edges:
                    edges = bin_edges[col_idx]
                else:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Column '{col}' (index {col_idx}) not found in bin_edges dict"
                    )
            else:
                print("Using global bin edges for column:", col)
                edges = bin_edges

            # Convert to numpy array if it's a list
            if isinstance(edges, list):
                edges = np.array(edges)

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

        print("Calculating histogram counts for each column")
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
<<<<<<< HEAD
>>>>>>> 0ddec5e6 (Initial draft for Histogram Logic for Tabular Dataset)
=======

    def quantile(
        self,
        bin_edges: Union[np.ndarray, Dict[str, np.ndarray]],
        q: Union[float, Sequence[float]],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute quantiles from the histogram of column values.

        Parameters
        ----------
        bin_edges:
            Global bin edges. Can be either:
            - 1D numpy array (length B+1): applied to all columns
            - Dict mapping column names to bin edges: column-specific bins
        q:
            Quantile(s) to compute. Scalar in [0, 1] or sequence of values in [0, 1].
            0.5 = median, 0.25 = first quartile, 0.75 = third quartile, etc.

        Returns
        -------
        dict: {column_name: quantile_value(s)}
            Quantile value(s) for each column. Scalar if q is scalar, array if q is sequence.
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

        # Get histogram counts for all columns
        counts_dict = self.histogram(bin_edges)

        result = {}

        # Process each column
        for col in self._input_columns:
            counts = counts_dict[col]

            # Determine bin_edges for this column
            if isinstance(bin_edges, dict):
                col_idx = (
                    self._input_columns.index(col)
                    if isinstance(self._input_columns, list)
                    else list(self._input_columns).index(col)
                )
                edges = bin_edges.get(col) or bin_edges.get(col_idx)
                if edges is None:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Column '{col}' not found in bin_edges dict"
                    )
            else:
                edges = bin_edges

            # Compute cumulative distribution
            total_count = np.sum(counts)
            if total_count == 0:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: No data available for quantile computation in column '{col}'"
                )

            cumsum = np.cumsum(counts)

            # Compute quantiles for this column
            quantiles = np.zeros_like(q_arr, dtype=np.float64)
            bin_widths = np.diff(edges)

            for i, quantile in enumerate(q_arr):
                # Find the target count
                target_count = quantile * total_count

                # Find the bin containing this quantile
                bin_idx = np.searchsorted(cumsum, target_count, side="left")
                bin_idx = np.clip(bin_idx, 0, len(counts) - 1)

                # Linear interpolation within the bin
                bin_left = edges[bin_idx]
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

            # Store result for this column (scalar if input was scalar)
            result[col] = quantiles[0] if q_is_scalar else quantiles

        return result
<<<<<<< HEAD
>>>>>>> 27505247 (Add draft quantile function that uses histogram for image and tabular datasets)
=======

    def minmax(self, **kwargs) -> Dict[str, tuple]:
        """Calculate min and max values of features across the dataset.

        Returns:
            Dictionary keyed by column names with (min, max) tuples for each column.

        Raises:
            FedbiomedError: if dataset is empty
        """
        num_samples = len(self)

        if num_samples == 0:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Cannot calculate minmax of an empty dataset."
            )

        result = {}

        for col_idx, col in enumerate(self._input_columns):
            col_min = float("inf")
            col_max = float("-inf")

            for idx in range(num_samples):
                data, _ = self[idx]
                # Extract value for this specific column
                value = float(data[col_idx])

                if np.isfinite(value):
                    col_min = min(col_min, value)
                    col_max = max(col_max, value)

            result[col] = (
                col_min if col_min != float("inf") else None,
                col_max if col_max != float("-inf") else None,
            )

        return result
>>>>>>> dd3aa21e (First working draft for histogram for Tabular Dataset)

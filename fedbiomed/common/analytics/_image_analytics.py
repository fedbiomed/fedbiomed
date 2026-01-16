# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence, Union

import numpy as np

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError

from ._analytics_strategy import AnalyticsStrategy


class ImageAnalytics(AnalyticsStrategy):
    """Mixin class for computing analytics on image folder datasets"""

    def _get_channel_axis(
        self, data: np.ndarray, channel: Union[int, str, None]
    ) -> Union[int, None]:
        """Determine the channel axis based on user input and data shape.

        Args:
            data: Image data array
            channel: Channel selection mode ('auto', int index, or None)

        Returns:
            Channel axis index or None if no channel dimension
        """
        if channel is None:
            return None

        if isinstance(channel, int):
            if channel >= data.ndim or channel < -data.ndim:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB633.value}: Channel index {channel} out of range for data with {data.ndim} dimensions"
                )
            return channel if channel >= 0 else data.ndim + channel

        if channel == "auto":
            # Find dimension with size < 4 (likely channel dimension)
            # Check first dimension
            if data.shape[0] < 4:
                return 0
            # Check last dimension
            if data.shape[-1] < 4:
                return -1
            return None

        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: Invalid channel value '{channel}'. Must be None, 'auto', or an integer index"
        )

    def _extract_feature_data(
        self,
        data: np.ndarray,
        feature: str,
        channel: Union[int, str, None] = None,
        per_channel: bool = False,
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """Extract data for a specific feature from image data.

        Args:
            data: Image data (2D, 3D, or 4D array)
            feature: Feature name:
                - 'intensity': Pixel/voxel values
                - 'height': Image height dimension
                - 'width': Image width dimension
                - 'depth': Image depth dimension
            channel: Channel selection for intensity:
                - None: flatten all pixel values
                - 'auto': automatically detect channel dimension
                - int: specific channel index

        Returns:
            1D array of values for the feature, or a dict of channel-indexed arrays
        """
        data = np.asarray(data, dtype=np.float64)

        if feature == "intensity":
            # Return pixel intensity values
            if per_channel:
                channel_axis = self._get_channel_axis(
                    data, channel if channel is not None else "auto"
                )
                if channel_axis is None:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB633.value}: Cannot determine channel axis for per-channel intensity."
                    )
                channel_count = data.shape[channel_axis]
                return {
                    idx: np.take(data, idx, axis=channel_axis).ravel()
                    for idx in range(channel_count)
                }

            return data.ravel()

        if feature == "per_channel":
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: Feature 'per_channel' is not supported; use per_channel=True for intensity."
            )

        if feature == "height":
            # Return image height (second-to-last dimension)
            if data.ndim >= 2:
                height = data.shape[-2]
                return np.array([float(height)])
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: Cannot extract height from data with {data.ndim} dimensions."
            )

        if feature == "width":
            # Return image width (last dimension)
            if data.ndim >= 2:
                width = data.shape[-1]
                return np.array([float(width)])
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: Cannot extract width from data with {data.ndim} dimensions."
            )

        if feature == "depth":
            # Return image depth (first spatial dimension)
            if data.ndim >= 3:
                channel_axis = self._get_channel_axis(data, channel)
                depth = data.shape[0] if channel_axis != 0 else data.shape[1]
                return np.array([float(depth)])
            return np.array([np.nan])

        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: Unknown feature '{feature}'. "
            f"Valid features are: 'height', 'width', 'depth', 'intensity'"
        )

    def _iter_feature_values(
        self,
        data: np.ndarray,
        feature: str,
        channel: Union[int, str, None],
        per_channel: bool,
    ):
        extracted = self._extract_feature_data(
            data, feature, channel=channel, per_channel=per_channel
        )
        if isinstance(extracted, dict):
            for channel_idx, values in extracted.items():
                yield f"{feature}_channel_{channel_idx}", values
        else:
            yield feature, extracted

    def basic_stats(
        self,
        features: list = None,
        channel: Union[int, str, None] = None,
        per_channel: bool = False,
        **kwargs,
    ) -> Dict:
        """Compute basic statistics (min, max, mean, std, count) for each feature.

        Stats are accumulated incrementally to avoid loading the entire dataset.
        """
        if not features:
            features = ["height", "width", "depth", "intensity"]

        # Initialize containers per feature
        stats = {}
        for feature in features:
            if feature == "intensity" and per_channel:
                # Each channel will be dynamically added later
                continue
            stats[feature] = {
                "count": 0,
                "mean": 0.0,
                "M2": 0.0,  # sum of squares of differences from current mean (Welford)
                "min": np.inf,
                "max": -np.inf,
            }

        # Iterate over dataset
        for data, _ in self:
            for feature in features:
                for out_feature, values in self._iter_feature_values(
                    data, feature, channel, per_channel
                ):
                    values = np.asarray(values, dtype=np.float64)
                    values = values[np.isfinite(values)]
                    if values.size == 0:
                        continue

                    fstats = stats.setdefault(
                        out_feature,
                        {
                            "count": 0,
                            "mean": 0.0,
                            "M2": 0.0,
                            "min": np.inf,
                            "max": -np.inf,
                        },
                    )

                    # Incremental update (Welford’s method)
                    n_old = fstats["count"]
                    n_new = n_old + len(values)
                    delta = np.mean(values) - fstats["mean"]
                    fstats["mean"] += delta * (len(values) / n_new)
                    fstats["M2"] += np.sum((values - fstats["mean"]) ** 2)
                    fstats["count"] = n_new
                    fstats["min"] = min(fstats["min"], float(np.min(values)))
                    fstats["max"] = max(fstats["max"], float(np.max(values)))

        # Finalize stats
        result = {}
        for name, s in stats.items():
            count = s["count"]
            if count == 0:
                result[name] = {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                    "count": 0,
                }
                continue
            std = np.sqrt(s["M2"] / count) if count > 1 else 0.0
            result[name] = {
                "min": float(s["min"]),
                "max": float(s["max"]),
                "mean": float(s["mean"]),
                "std": float(std),
                "count": count,
            }

        return result

    def histogram(
        self,
        bin_edges: Union[np.ndarray, Dict[str, np.ndarray]],
        features: list = None,
        channel: Union[int, str, None] = None,
        per_channel: bool = False,
    ) -> Dict[str, Dict[str, list]]:
        """
        Compute histogram counts for image features across all images in the dataset.

        Args:
            bin_edges: Global bin edges broadcast by the researcher. Can be:
                - 1D numpy array: applied to all features
                - Dict mapping feature names to bin_edges
            features: List of features to compute histograms for. Valid values:
                - 'intensity': Pixel intensity values
                - 'height': Height dimension
                - 'width': Width dimension
                - 'depth': Depth dimension
                If empty or None, compute for 'intensity' only.
            channel: Channel selection for intensity:
                - None: flatten all pixel values
                - 'auto': automatically detect channel dimension
                - int: specific channel index
            per_channel: If True, compute intensity histograms per channel.

        Returns:
            Dictionary mapping feature names to {"bin_edges": [...], "counts": [...]}

        Raises:
            FedbiomedError: if bin_edges is not a 1D array of length >= 2
            FedbiomedError: if bin_edges are not strictly increasing
        """
        if features is None or len(features) == 0:
            features = ["intensity"]

        # Parse bin_edges
        if isinstance(bin_edges, dict):
            bin_edges_dict = bin_edges
        else:
            # Single bin_edges array applies to all features
            if isinstance(bin_edges, list):
                bin_edges = np.array(bin_edges)
            bin_edges_dict = {feature: bin_edges for feature in features}

        # Validate bin_edges
        for feature, edges in bin_edges_dict.items():
            if isinstance(edges, list):
                edges = np.array(edges)
            if edges.ndim != 1 or edges.size < 2:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB633.value}: bin_edges for '{feature}' must be a 1D array of length >= 2"
                )
            if not np.all(np.diff(edges) > 0):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB633.value}: bin_edges for '{feature}' must be strictly increasing"
                )

        histograms: Dict[str, np.ndarray] = {}
        histogram_edges: Dict[str, np.ndarray] = {}

        # Single pass through dataset
        for data, _ in self:
            for feature in features:
                for out_feature, feature_values in self._iter_feature_values(
                    data, feature, channel, per_channel
                ):
                    feature_values = feature_values[np.isfinite(feature_values)]
                    if feature_values.size == 0:
                        continue

                    edges = bin_edges_dict.get(out_feature)
                    if edges is None:
                        edges = bin_edges_dict.get(feature)
                    if edges is None:
                        raise FedbiomedError(
                            f"{ErrorNumbers.FB633.value}: bin_edges missing for feature '{out_feature}'"
                        )
                    edges = np.array(edges)

                    if out_feature not in histograms:
                        histograms[out_feature] = np.zeros(
                            edges.size - 1, dtype=np.int64
                        )
                        histogram_edges[out_feature] = edges

                    clipped = np.clip(feature_values, edges[0], edges[-1])
                    histograms[out_feature] += np.histogram(clipped, bins=edges)[0]

        return {
            feature: {
                "bin_edges": list(histogram_edges[feature]),
                "counts": list(counts),
            }
            for feature, counts in histograms.items()
        }

    def min_max(
        self,
        features: list = None,
        channel: Union[int, str, None] = None,
        per_channel: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate min and max values for image features across all images in the dataset.

        Args:
            features: List of features to compute min/max for. Valid values:
                - 'intensity': Pixel intensity values
                - 'height': Height dimension
                - 'width': Width dimension
                - 'depth': Depth dimension
                If empty or None, compute for 'intensity' only.
            channel: Channel selection for intensity:
                - None: flatten all pixel values
                - 'auto': automatically detect channel dimension
                - int: specific channel index
            per_channel: If True, compute intensity min/max per channel.

        Returns:
            Dictionary mapping feature names to {"min": ..., "max": ...}

        Raises:
            FedbiomedError: if dataset is empty
        """
        if features is None or len(features) == 0:
            features = ["intensity"]

        if len(self) == 0:
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: Cannot calculate min_max of an empty dataset."
            )

        minmax_dict: Dict[str, tuple] = {}

        for data, _ in self:
            for feature in features:
                for out_feature, feature_values in self._iter_feature_values(
                    data, feature, channel, per_channel
                ):
                    finite_values = feature_values[np.isfinite(feature_values)]

                    if finite_values.size > 0:
                        curr_min, curr_max = minmax_dict.get(
                            out_feature, (float("inf"), float("-inf"))
                        )
                        minmax_dict[out_feature] = (
                            min(curr_min, np.min(finite_values)),
                            max(curr_max, np.max(finite_values)),
                        )

        # Convert inf values to None for features with no data
        result = {}
        for feature, (min_val, max_val) in minmax_dict.items():
            result[feature] = {
                "min": min_val if min_val != float("inf") else np.nan,
                "max": max_val if max_val != float("-inf") else np.nan,
            }

        for feature in features:
            if feature == "intensity" and per_channel:
                continue
            if feature not in result:
                result[feature] = {"min": np.nan, "max": np.nan}

        return result

    def quantile(
        self,
        bin_edges: Union[np.ndarray, Dict[str, np.ndarray]],
        q: Union[float, Sequence[float]],
        features: list = None,
        channel: Union[int, str, None] = None,
        per_channel: bool = False,
    ) -> Dict[str, Dict[float, float]]:
        """
        Compute quantiles for image features from their histograms.

        Args:
            bin_edges: Global bin edges. Can be:
                - 1D array: applied to all features
                - Dict mapping feature names to bin_edges
            q: Quantile(s) to compute. Scalar in [0, 1] or sequence of values in [0, 1].
               0.5 = median, 0.25 = first quartile, 0.75 = third quartile, etc.
            features: List of features to compute quantiles for. Valid values:
                - 'intensity': Pixel intensity values
                - 'height': Height dimension
                - 'width': Width dimension
                - 'depth': Depth dimension
                If empty or None, compute for 'intensity' only.
            channel: Channel selection for intensity:
                - None: flatten all pixel values
                - 'auto': automatically detect channel dimension
                - int: specific channel index
            per_channel: If True, compute intensity quantiles per channel.

        Returns:
            Dictionary mapping feature names to dicts of quantile values and computed quantiles.
            e.g., {'intensity': {0.5: median_value, 0.25: q1_value, ...}}

        Raises:
            FedbiomedError: if q values are not in [0, 1]
        """
        if features is None or len(features) == 0:
            features = ["intensity"]

        # Parse bin_edges
        if isinstance(bin_edges, dict):
            bin_edges_dict = bin_edges
        else:
            # Single bin_edges array applies to all features
            if isinstance(bin_edges, list):
                bin_edges = np.array(bin_edges)
            bin_edges_dict = {feature: bin_edges for feature in features}

        # Normalize q to array
        q_arr = np.atleast_1d(q)

        # Validate q values
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: Quantile values must be in [0, 1]"
            )

        # Get histograms and compute quantiles for each feature
        quantiles_dict = {}
        histograms = self.histogram(
            bin_edges_dict, features, channel, per_channel=per_channel
        )

        for out_feature, hist in histograms.items():
            counts = np.array(hist["counts"])
            base_feature = (
                "intensity"
                if out_feature.startswith("intensity_channel_")
                else out_feature
            )
            edges = bin_edges_dict.get(out_feature)
            if edges is None:
                edges = bin_edges_dict.get(base_feature)
            if edges is None:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB633.value}: bin_edges missing for feature '{out_feature}'"
                )
            edges = np.array(edges)

            # Compute cumulative distribution
            total_count = np.sum(counts)
            if total_count == 0:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB633.value}: No data available for quantile computation on feature '{out_feature}'"
                )

            cumsum = np.cumsum(counts)

            # Compute quantiles
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

            # Return dictionary mapping quantile values to computed quantiles
            quantiles_dict[out_feature] = {
                float(q_val): float(quant)
                for q_val, quant in zip(q_arr, quantiles, strict=True)
            }

        return quantiles_dict

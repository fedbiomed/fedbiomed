# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

import numpy as np

from fedbiomed.common.constants import AnalyticsTypes, DatasetTypes, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


class AnalyticsResult:
    """Generic result wrapper for federated analytics outputs."""

    _METHOD_MAP = {
        (DatasetTypes.TABULAR.value, AnalyticsTypes.BASIC_STATS.value): {
            "aggregate": "_aggregate_tabular_basic_stats",
            "values": "_values_tabular_basic_stats",
        },
        (DatasetTypes.TABULAR.value, AnalyticsTypes.HISTOGRAM.value): {
            "aggregate": "_aggregate_tabular_histogram",
            "values": "_values_tabular_histogram",
        },
        (DatasetTypes.IMAGES.value, AnalyticsTypes.HISTOGRAM.value): {
            "aggregate": "_aggregate_image_histogram",
            "values": "_values_image_histogram",
        },
    }

    def __init__(
        self,
        dataset_type: DatasetTypes,
        analytics_type: AnalyticsTypes,
        node_results: Dict[str, Any],
        errors: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._dataset_type = dataset_type
        self._analytics_type = analytics_type
        self._node_results = node_results or {}
        self._errors = errors or {}

    @property
    def errors(self) -> Dict[str, Any]:
        return self._errors

    def aggregate(self) -> Dict[str, Any]:
        handler = self._resolve_mapping("aggregate")
        return handler()

    def values(self) -> Dict[str, Any]:
        handler = self._resolve_mapping("values")
        return handler()

    def _resolve_mapping(self, method_type: str):
        key = (self._dataset_type.value, self._analytics_type.value)
        mapping = self._METHOD_MAP.get(key)
        if mapping is None or method_type not in mapping:
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: "
                f"No '{method_type}' handler for analytics result '{key}'."
            )
        return getattr(self, mapping[method_type])

    @staticmethod
    def _is_finite(value: Any) -> bool:
        try:
            return value is not None and np.isfinite(value)
        except TypeError:
            return False

    def _values_by_node(self) -> Dict[str, Any]:
        return {
            node_id: self._extract_output(reply)
            for node_id, reply in self._node_results.items()
        }

    @staticmethod
    def _extract_output(reply: Any) -> Any:
        return reply.output if hasattr(reply, "output") else reply

    def _aggregate_tabular_basic_stats(self) -> Dict[str, Dict[str, float]]:
        values_by_node = self._values_by_node()
        accumulators: Dict[str, Dict[str, float]] = {}

        for output in values_by_node.values():  # for each node's output
            for col, stats in output.items():  # for each column in the output
                acc = accumulators.setdefault(
                    col,
                    {
                        "min": None,
                        "max": None,
                        "count": 0.0,
                        "mean": 0.0,
                        "m2": 0.0,
                    },
                )

                node_min = stats.get("min")
                node_max = stats.get("max")
                if self._is_finite(node_min):
                    acc["min"] = (
                        node_min if acc["min"] is None else min(acc["min"], node_min)
                    )
                if self._is_finite(node_max):
                    acc["max"] = (
                        node_max if acc["max"] is None else max(acc["max"], node_max)
                    )

                node_count = stats.get("count")
                if not self._is_finite(node_count) or float(node_count) <= 0:
                    continue

                node_mean = stats.get("mean")
                if not self._is_finite(node_mean):
                    continue

                node_std = stats.get("std")
                node_m2 = (
                    float(node_std) ** 2 * float(node_count)
                    if self._is_finite(node_std)
                    else 0.0
                )

                self._combine_mean_m2(
                    acc, float(node_count), float(node_mean), float(node_m2)
                )

        aggregated: Dict[str, Dict[str, float]] = {}
        for col, acc in accumulators.items():
            count = acc["count"]
            min_val = acc["min"] if acc["min"] is not None else np.nan
            max_val = acc["max"] if acc["max"] is not None else np.nan

            if count > 0:
                std = float(np.sqrt(acc["m2"] / count))
                aggregated[col] = {
                    "min": float(min_val),
                    "max": float(max_val),
                    "count": int(count),
                    "mean": float(acc["mean"]),
                    "std": std,
                }
            else:
                aggregated[col] = {
                    "min": float(min_val),
                    "max": float(max_val),
                    "count": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                }

        return aggregated

    @staticmethod
    def _combine_mean_m2(
        acc: Dict[str, float],
        node_count: float,
        node_mean: float,
        node_m2: float,
    ) -> None:
        if acc["count"] == 0:
            acc["count"] = node_count
            acc["mean"] = node_mean
            acc["m2"] = node_m2
            return

        delta = node_mean - acc["mean"]
        total_count = acc["count"] + node_count
        acc["mean"] += delta * node_count / total_count
        acc["m2"] += node_m2 + (delta**2) * acc["count"] * node_count / total_count
        acc["count"] = total_count

    def _aggregate_histogram(self) -> Dict[str, np.ndarray]:
        values_by_node = self._values_by_node()
        aggregated: Dict[str, np.ndarray] = {}
        for output in values_by_node.values():  # for each node's output
            for name, counts in output.items():  # for each histogram in the output
                counts_array = np.array(counts)
                if name not in aggregated:
                    aggregated[name] = counts_array
                else:
                    aggregated[name] += counts_array
        return aggregated

    def _aggregate_tabular_histogram(self) -> Dict[str, np.ndarray]:
        return self._aggregate_histogram()

    def _aggregate_image_histogram(self) -> Dict[str, np.ndarray]:
        return self._aggregate_histogram()

    def _values_tabular_basic_stats(self) -> Dict[str, Any]:
        return self._values_by_node()

    def _values_tabular_histogram(self) -> Dict[str, Any]:
        return self._values_by_node()

    def _values_image_histogram(self) -> Dict[str, Any]:
        return self._values_by_node()

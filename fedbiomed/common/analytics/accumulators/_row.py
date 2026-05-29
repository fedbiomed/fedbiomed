# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Tuple, Type

import numpy as np

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_types import DatasetElementType
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._base import Accumulator
from ._registry import AnalyticsRegistry


class RowAccumulator(Accumulator):
    """Accumulator for tabular row data.

    - Simple statistics (no extra args) are batched across all requested columns
    using a single NumPy-sliced accumulator per stat type (vectorized path).
    - Parameterized statistics get their own accumulator instance per column
    (independent path).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: dict with ``conf`` (column→stats mapping) and ``schema_columns`` (ordered column names).
        """
        self.column_configs = config.get("conf", {})
        self.schema_columns = config.get("schema_columns")

        if not self.schema_columns:
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: RowAccumulator requires 'schema_columns' in config "
                "to map column names to data indices."
            )
        if not all(col in self.schema_columns for col in self.column_configs):
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: All columns in 'conf' must be present in 'schema_columns'."
            )
        if not self.column_configs:
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: RowAccumulator requires a non-empty 'conf' in config."
            )

        self.col_map: Dict[str, int] = {
            name: i for i, name in enumerate(self.schema_columns)
        }

        col_indices, seen_classes, self.independent_accumulators = (
            self._classify_stats()
        )
        (
            self.vectorized_indices,
            self.vectorized_output_map,
            self.vectorized_accumulators,
        ) = self._build_vectorized_accumulators(col_indices, seen_classes)

        logger.debug(
            f"RowAccumulator initialized with schema_columns: {self.schema_columns}"
        )

    def _classify_stats(
        self,
    ) -> Tuple[
        Dict[str, List[int]],
        Dict[str, Dict[Type[Accumulator], None]],
        Dict[int, List[Tuple[str, Accumulator]]],
    ]:
        """Classify each stat as vectorized (no args) or independent (has args).

        Returns:
            As col_indices, per-stat list of column indices for vectorized stats.
            As seen_classes, per-stat insertion-ordered set of accumulator classes.
            As independent_accumulators, per-column list of (stat_name, accumulator) pairs.
        """
        col_indices: Dict[str, List[int]] = {}
        seen_classes: Dict[str, Dict[Type[Accumulator], None]] = {}
        independent: Dict[int, List[Tuple[str, Accumulator]]] = {}

        for col_name, stats in self.column_configs.items():
            idx = self.col_map[col_name]
            for stat, args in stats.items():
                if not args:
                    col_indices.setdefault(stat, []).append(idx)
                    for acc_cls in AnalyticsRegistry.get_accumulators(
                        stat, DatasetElementType.ROW
                    ):
                        seen_classes.setdefault(stat, {})[acc_cls] = None
                else:
                    for acc_cls in AnalyticsRegistry.get_accumulators(
                        stat, DatasetElementType.ROW
                    ):
                        acc = acc_cls(**args)
                        independent.setdefault(idx, []).append((stat, acc))

        return col_indices, seen_classes, independent

    def _build_vectorized_accumulators(
        self,
        col_indices: Dict[str, List[int]],
        seen_classes: Dict[str, Dict[Type[Accumulator], None]],
    ) -> Tuple[
        Dict[str, List[int]],
        Dict[str, Dict[int, int]],
        Dict[str, List[Accumulator]],
    ]:
        """Build deduped index arrays and shared accumulator instances for vectorized stats.

        Args:
            col_indices: Per-stat list of column indices.
            seen_classes: Per-stat insertion-ordered set of accumulator classes.

        Returns:
            As vectorized_indices, per-stat sorted list of column indices.
            As vectorized_output_map, per-stat mapping from column index to position in packed output.
            As vectorized_accumulators, per-stat list of instantiated accumulators (deterministic order).
        """
        v_indices: Dict[str, List[int]] = {}
        v_output_map: Dict[str, Dict[int, int]] = {}
        v_accumulators: Dict[str, List[Accumulator]] = {}

        for stat, indices in col_indices.items():
            sorted_idx = sorted(set(indices))
            v_indices[stat] = sorted_idx
            v_output_map[stat] = {
                col_idx: pos for pos, col_idx in enumerate(sorted_idx)
            }
            v_accumulators[stat] = [
                acc_cls() for acc_cls in seen_classes.get(stat, {}).keys()
            ]

        return v_indices, v_output_map, v_accumulators

    def update(self, value: np.ndarray) -> None:
        """value must be 1D with one element per schema column."""
        value = np.asarray(value)

        if value.ndim != 1:
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: RowAccumulator.update: Expected 1D array, got ndim={value.ndim}"
            )
        if len(value) != len(self.schema_columns):
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: RowAccumulator.update: Expected {len(self.schema_columns)} "
                f"elements, got {len(value)}"
            )

        for stat, indices in self.vectorized_indices.items():
            for acc in self.vectorized_accumulators[stat]:
                acc.update(value[indices])

        for col_idx, acc_list in self.independent_accumulators.items():
            for _, acc in acc_list:
                acc.update(value[col_idx])

    def finalize(self) -> Dict[str, Dict[str, Any]]:
        """Returns ``{col_name: {stat_name: value}}``."""
        arrays_per_stat: Dict[str, Dict[str, Any]] = {}
        for stat_name, acc_list in self.vectorized_accumulators.items():
            output_arrays: Dict[str, Any] = {}
            for acc in acc_list:
                output_arrays.update(acc.finalize())
            arrays_per_stat[stat_name] = output_arrays

        results: Dict[str, Dict[str, Any]] = {}

        for col_name, col_stats in self.column_configs.items():
            col_idx = self.col_map[col_name]
            col_results: Dict[str, Any] = {}

            for stat_name in col_stats:
                col_position_map = self.vectorized_output_map.get(stat_name)
                if col_position_map is not None and col_idx in col_position_map:
                    col_pos = col_position_map[col_idx]
                    for output_key, packed_array in arrays_per_stat[stat_name].items():
                        col_results[output_key] = packed_array[col_pos]

            for stat_name, acc in self.independent_accumulators.get(col_idx, []):
                col_results[stat_name] = acc.finalize()

            results[col_name] = col_results

        return results

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Tuple

import numpy as np

from fedbiomed.common.constants import FedbiomedError
from fedbiomed.common.dataset_types import DatasetElementType
from fedbiomed.common.logger import logger

from ._base import Accumulator
from ._registry import AnalyticsRegistry


class RowAccumulator(Accumulator):
    """Accumulator explicitly designed for Tabular/Row data processing.

    This class employs a hybrid strategy for efficiency:
    1. Vectorization: Standard primitives (min, max, sum, count, sum_squares) are grouped
       and computed across all columns simultaneously using NumPy vectorization.
    2. Independent Handling: Complex statistics (currently independent) or those with
       column-specific arguments are handled via dedicated accumulators per column.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary coming from Orchestrator.
        """
        # Initialize instance attributes
        self.vectorized_indices: Dict[str, List[int]] = {}
        self.vectorized_accumulators: Dict[str, Accumulator] = {}
        self.vectorized_accumulators_classes: Dict[str, type] = {}
        self.independent_accumulators: Dict[int, List[Tuple[str, Accumulator]]] = {}
        # Pre-calculated map for O(1) lookups: stat -> {col_idx: vector_idx}
        self.vectorized_output_map: Dict[str, Dict[int, int]] = {}

        logger.info(f"Initializing RowAccumulator with config: {config}")
        self.column_configs = config.get("conf", {})
        self.column_order = config.get("columns")

        if not self.column_order:
            raise FedbiomedError(
                "RowAccumulator requires 'columns' in config to map column names to indices."
            )
        if not all(col in self.column_configs for col in self.column_order):
            raise FedbiomedError(
                "All columns in 'columns' must have corresponding entries in 'conf'."
            )

        # O(1) lookup for column index
        self.col_map = {name: i for i, name in enumerate(self.column_order)}

        for col_name, stats in self.column_configs.items():
            idx = self.col_map.get(col_name)
            if idx is None:
                continue

            for stat, stat_args in stats.items():
                # Get accumulator class from registry
                accumulator_class = AnalyticsRegistry.get_accumulator_class(
                    stat, DatasetElementType.ROW
                )

                # Case A: Standard Vectorizable Primitive (from registry)
                if accumulator_class and not stat_args:
                    if stat not in self.vectorized_accumulators_classes:
                        self.vectorized_accumulators_classes[stat] = accumulator_class
                    if stat not in self.vectorized_indices:
                        self.vectorized_indices[stat] = []
                    self.vectorized_indices[stat].append(idx)
                # Case B: Complex or Argument-Dependent Stat
                else:
                    self._add_independent_accumulator(idx, stat, stat_args)

        # Instantiate vectorized accumulators and build output map
        for stat, indices in self.vectorized_indices.items():
            if indices:
                sorted_indices = sorted(indices)
                self.vectorized_indices[stat] = sorted_indices

                # Build map: stat -> {col_idx: vector_pos}
                self.vectorized_output_map[stat] = {
                    col_idx: pos for pos, col_idx in enumerate(sorted_indices)
                }

                if stat in self.vectorized_accumulators_classes:
                    self.vectorized_accumulators[stat] = (
                        self.vectorized_accumulators_classes[stat]()
                    )

        logger.info("Vectorized accumulators initialized")

    def _add_independent_accumulator(
        self, col_idx: int, stat_name: str, args: Dict[str, Any]
    ):
        """Factory for independent/complex accumulators."""
        acc = None
        if stat_name == "histogram":
            # Placeholder for histogram
            pass

        if acc:
            if col_idx not in self.independent_accumulators:
                self.independent_accumulators[col_idx] = []
            self.independent_accumulators[col_idx].append((stat_name, acc))

    def update(self, value: np.ndarray) -> None:
        """Update state with a new row sample."""
        # Ensure NumPy array
        value = np.asarray(value)

        if value.ndim != 1:
            raise FedbiomedError(
                f"RowAccumulator.update: Expected 1D array, got ndim={value.ndim}"
            )

        # Vectorized Updates
        for stat, indices in self.vectorized_indices.items():
            if stat in self.vectorized_accumulators:
                self.vectorized_accumulators[stat].update(value[indices])

        # Independent Updates
        for col_idx, acc_list in self.independent_accumulators.items():
            if col_idx < len(value):
                val = value[col_idx]
                for _, acc in acc_list:
                    acc.update(val)

    def finalize(self) -> Dict[str, Dict[str, Any]]:
        """Return the final state by column."""
        results: Dict[str, Dict[str, Any]] = {}

        # Retrieve all vectorized results once
        vec_results = {
            stat: acc.finalize() for stat, acc in self.vectorized_accumulators.items()
        }

        # Re-distribute to columns
        for col_name, col_conf in self.column_configs.items():
            idx = self.col_map.get(col_name)
            if idx is None:
                continue

            col_res = {}

            # Process stats defined for this column
            for stat in col_conf:
                # Check Vectorized
                vec_indices_map = self.vectorized_output_map.get(stat)
                if vec_indices_map and idx in vec_indices_map:
                    pos = vec_indices_map[idx]
                    val_result = vec_results[stat]

                    if isinstance(val_result, dict):
                        for key, val_array in val_result.items():
                            if np.isscalar(val_array):
                                col_res[key] = val_array
                            else:
                                col_res[key] = val_array[pos]
                    else:
                        col_res[stat] = (
                            val_result if np.isscalar(val_result) else val_result[pos]
                        )

            # Process Independent
            if idx in self.independent_accumulators:
                for stat_name, accumulator in self.independent_accumulators[idx]:
                    col_res[stat_name] = accumulator.finalize()

            results[col_name] = col_res

        return results

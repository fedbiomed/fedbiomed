# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from fedbiomed.common.analytics import AnalyticsStrategy


class TabularAnalytics(AnalyticsStrategy):
    """Mixin class for computing analytics on tabular datasets"""

    def mean(self, **kwargs) -> Dict:
        """Calculate mean of features across the dataset

        Iterates through dataset items (tuples of data and target tensors/numpy arrays)
        and computes the mean of both data and target features manually.

        Returns:
            Dictionary keyed by column names (input_columns + target_columns) with their
            respective means. Preserves the input data type (Tensor or numpy array).
        """
        data_sum = 0
        target_sum = 0
        count = 0

        for idx in range(len(self)):
            data, target = self[idx]
            data_sum += data
            count += 1
            if target is not None:
                target_sum += target

        if count == 0:
            # Return dict with None values for all columns
            result = {}
            for col in self._input_columns:
                result[col] = None
            if self._target_columns is not None:
                for col in self._target_columns:
                    result[col] = None
            return result

        # Calculate means by dividing by count
        data_mean = data_sum / count
        target_mean = target_sum / count if target_sum != 0 else None

        # Build result dict keyed by column names
        result = {}

        # Add input column means
        for i, col in enumerate(self._input_columns):
            result[col] = (
                data_mean[i] if hasattr(data_mean, "__getitem__") else data_mean
            )

        # Add target column means (only if target_columns is not None)
        if self._target_columns is not None:
            for i, col in enumerate(self._target_columns):
                result[col] = (
                    target_mean[i]
                    if hasattr(target_mean, "__getitem__")
                    else target_mean
                )

        return result

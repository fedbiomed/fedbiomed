# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError

from ._analytics_strategy import AnalyticsStrategy


class TabularAnalytics(AnalyticsStrategy):
    """Mixin class for computing analytics on tabular datasets"""

    def mean(self, **kwargs) -> Dict:
        """Calculate mean of features across the dataset

        Iterates through dataset items (tuples of data and target tensors/numpy arrays)
        and computes the mean of both data and target features manually.
        Only considers numeric columns.

        Returns:
            Dictionary keyed by column names (input_columns + target_columns) with their
            respective means. Preserves the input data type (Tensor or numpy array).

        Raises:
            FedbiomedError: if dataset is empty
        """
        # Filter out non-numeric columns
        numeric_inputs, numeric_targets = self._filter_numeric_columns()

        if not numeric_inputs and not numeric_targets:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: No numeric columns found for mean calculation."
            )

        # Start calculating mean
        data_sum = 0
        target_sum = 0
        num_samples = len(self)

        # Raise error if no data is found
        if num_samples == 0:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Cannot calculate mean of an empty dataset."
            )

        for idx in range(num_samples):
            data, target = self[idx]
            data_sum += data[numeric_inputs]
            if target is not None:
                target_sum += target[numeric_targets]

        # Calculate means by dividing by count
        data_mean = data_sum / num_samples
        target_mean = target_sum / num_samples if target_sum != 0 else None

        # Build result dict keyed by column names
        # Get the column names of the numeric columns
        result = {}
        input_columns = (self._input_columns[i] for i in numeric_inputs)
        target_columns = (
            (self._target_columns[i] for i in numeric_targets)
            if self._target_columns is not None
            else None
        )

        # Add input column means
        for i, col in enumerate(input_columns):
            result[col] = (
                data_mean[i] if hasattr(data_mean, "__getitem__") else data_mean
            )

        # Add target column means (only if target_columns is not None)
        if target_columns is not None:
            for i, col in enumerate(target_columns):
                result[col] = (
                    target_mean[i]
                    if hasattr(target_mean, "__getitem__")
                    else target_mean
                )

        return result

    def _filter_numeric_columns(self):
        """Filter columns to keep only numeric ones

        Returns:
            Column numbers of the numeric columns
        """
        controller = self._controller
        schema = controller._reader.data.schema

        input_cols = controller.normalize_columns(self._input_columns)
        target_cols = (
            controller.normalize_columns(self._target_columns)
            if self._target_columns is not None
            else []
        )

        numeric_inputs = [
            i for i, col in enumerate(input_cols) if schema[col].is_numeric()
        ]

        numeric_targets = [
            i for i, col in enumerate(target_cols) if schema[col].is_numeric()
        ]

        return (numeric_inputs, numeric_targets)

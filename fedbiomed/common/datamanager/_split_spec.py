# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data Splitting class
"""

from numbers import Integral
from typing import Any, Iterable, List, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger


class SplitSpec:
    """Specification object describing dataset splitting"""

    def __init__(
        self,
        dataset: Union[Dataset, Any],
        train_indices: Iterable[int],
        validation_indices: Iterable[int],
    ):
        """Constructor of SplitSpec

        Args:
            dataset: Dataset to be split
            train_indices: List of indices for training subset
            validation_indices: List of indices for validation subset
        """
        self.dataset = dataset
        self.splits = {
            "train": self._to_list(train_indices, "train"),
            "validation": self._to_list(validation_indices, "validation"),
        }
        self._validate()

    def _to_list(self, indices: Iterable[int], name: str) -> List[int]:
        """Convert input to list and normalize errors.

        Args:
            indices: Input indices to convert to list
            name: Name of the split (for error messages)
        """
        try:
            converted = list(indices)
        except TypeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                f"{name} indices must be an iterable of integers."
            ) from e
        return converted

    def _validate(self) -> None:
        """Validate the split specification for consistency and correctness."""

        dataset_size = len(self.dataset)

        # Per-split checks
        for name, indices in self.splits.items():
            if len(indices) == 0:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: "
                    f"{name.capitalize()} split cannot be empty."
                )

            if not all(isinstance(i, Integral) for i in indices):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: "
                    f"{name.capitalize()} indices must be integers."
                )

            if any(i < 0 for i in indices):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: "
                    f"Negative {name} indices are not allowed."
                )

            if len(indices) != len(set(indices)):
                logger.warning(f"Duplicate {name} indices detected.")

            if max(indices) >= dataset_size:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: "
                    f"{name.capitalize()} indices exceed dataset size."
                )

        # Cross-split validation
        split_names = list(self.splits.keys())

        for i, name_a in enumerate(split_names):
            for name_b in split_names[i + 1 :]:
                overlap = set(self.splits[name_a]) & set(self.splits[name_b])
                if overlap:
                    logger.warning(
                        f"{name_a.capitalize()} and {name_b} splits overlap."
                    )

    @property
    def train_indices(self) -> List[int]:
        """Get the training indices.

        Returns:
            List of training indices.
        """
        return self.splits["train"]

    @property
    def validation_indices(self) -> List[int]:
        """Get the validation indices.

        Returns:
            List of validation indices.
        """
        return self.splits["validation"]

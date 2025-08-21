# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_types import DataReturnFormat, DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError


class Dataset(ABC):
    _controller_cls: type = None
    _controller = None
    _to_format: DataReturnFormat = None

    # === Properties ===
    @property
    def to_format(self) -> DataReturnFormat:
        return self._to_format

    @to_format.setter
    def to_format(self, to_format_input: DataReturnFormat):
        if not isinstance(to_format_input, DataReturnFormat):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: `to_format` is not `DataReturnFormat` type"
            )
        self._to_format = to_format_input

    # === Abstract functions ===
    @abstractmethod
    def complete_initialization(self) -> None:
        pass

    @abstractmethod
    def _apply_transforms(self, sample: Dict[str, Any]) -> Tuple[Any, Any]:
        pass

    # === Functions ===
    def _init_controller(self, controller_kwargs: Dict[str, Any]) -> None:
        """Initializes self._controller

        Args:
            controller_kwargs: arguments necessary to initialize the controller

        Raises:
            FedbiomedError: if `controller_kwargs` is not a `dict`
            FedbiomedError: if there is a problem instantiating `_controller`
        """
        if not isinstance(controller_kwargs, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected `controller_kwargs` to be a "
                f"`dict`, got {type(controller_kwargs).__name__}"
            )

        try:
            # Instantiate controller
            self._controller = self._controller_cls(**controller_kwargs)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to create Controller. {e}"
            ) from e

    def __getitem__(self, idx: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Apply transforms to sample and returns it"""
        return self._apply_transforms(self._controller._get_nontransformed_item(idx))

    def __len__(self) -> int:
        return len(self._controller)

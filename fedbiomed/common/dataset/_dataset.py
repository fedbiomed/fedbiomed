# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch

from fedbiomed.common.constants import ErrorNumbers, _BaseEnum
from fedbiomed.common.dataset_controller import (
    ImageFolderController,
    MedicalFolderController,
    MedNistController,
    MnistController,
)
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

# === Constants ===

# Nota: `key` must match 'name' given by `value._controller_args`
CONTROLLER_REGISTRY = {
    "ImageFolder": ImageFolderController,
    "MedicalFolder": MedicalFolderController,
    "MedNIST": MedNistController,
    "MNIST": MnistController,
}

# === Enums ===


# Nota: `value` serves in `isinstance` call for validation of `transforms`
class DataReturnFormat(_BaseEnum):
    SKLEARN = (np.ndarray, pd.DataFrame, pd.Series)
    TORCH = torch.Tensor


class Dataset(ABC):
    _allowed_controllers: Union[str, Tuple[str, ...]] = None
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
    def _init_controller(self, controller_kwargs: Dict):
        """Initializes self._controller

        Args:
            controller_kwargs: arguments necessary to initialize the controller

        Raises:
            FedbiomedError: if `controller_kwargs` is not a `dict`
            FedbiomedError: if key 'name' is not present in `controller_kwargs`
            FedbiomedError: if controller 'name' is not in `_allowed_controllers`
            FedbiomedError: if there is a problem instantiating `_controller`
        """
        if not isinstance(controller_kwargs, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected `controller_kwargs` to be a "
                f"`dict`, got {type(controller_kwargs).__name__}"
            )
        if "name" not in controller_kwargs:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: 'name' not found in `controller_kwargs`"
            )
        if (
            self._allowed_controllers is None
            or (
                isinstance(self._allowed_controllers, str)
                and controller_kwargs["name"] != self._allowed_controllers
            )
            or (
                isinstance(self._allowed_controllers, dict)
                and controller_kwargs["name"] not in self._allowed_controllers
            )
        ):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Controller name not found in allowed "
                f"controllers for this Dataset"
            )

        try:
            # Instantiate controller
            self._controller = CONTROLLER_REGISTRY[controller_kwargs["name"]](
                **{_k: _v for _k, _v in controller_kwargs.items() if _k != "name"}
            )
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to create Controller. {e}"
            ) from e

    def _get_nontransformed_item(self, index: int) -> Dict[str, Any]:
        """Retrieve a data sample directly from `self._controller`"""
        try:
            item = self._controller._get_nontransformed_item(index=index)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item from controller"
            ) from e
        return item

    def __getitem__(self, idx: int):
        """Apply transforms to sample and returns it"""
        return self._apply_transforms(self._get_nontransformed_item(idx))

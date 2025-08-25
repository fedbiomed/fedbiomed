from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as T

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller import (
    ImageFolderController,
    MedNistController,
    MnistController,
)
from fedbiomed.common.dataset_types import DataReturnFormat, DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

from ._dataset import Dataset


class SimpleDataset(Dataset):
    "Dataset where data and target are implicitly predefined by the controller"

    _native_to_framework_transform = {
        DataReturnFormat.SKLEARN: np.array,
        DataReturnFormat.TORCH: T.ToTensor(),
    }
    _native_to_framework_target_transform = {
        DataReturnFormat.SKLEARN: np.array,
        DataReturnFormat.TORCH: torch.tensor,
    }

    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        if type(self) is SimpleDataset:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: "
                "`SimpleDataset` cannot be instantiated directly"
            )
        self.transform = transform
        self.target_transform = target_transform

    # === Properties ===
    @property
    def native_to_framework_transform(self) -> Callable:
        return self._native_to_framework_transform[self._to_format]

    @property
    def native_to_framework_target_transform(self) -> Callable:
        return self._native_to_framework_target_transform[self._to_format]

    @property
    def transform(self) -> Callable:
        return self._transform

    @transform.setter
    def transform(self, transform_input: Optional[Callable]):
        """Raises FedbiomedValueError if transform_input is not valid"""
        if transform_input is None:
            self._transform = lambda x: x
        elif callable(transform_input):
            self._transform = transform_input
        else:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for `transform`"
            )

    @property
    def target_transform(self) -> Callable:
        return self._target_transform

    @target_transform.setter
    def target_transform(self, transform_input: Optional[Callable]):
        """Raises FedbiomedValueError if transform_input is not valid"""
        if transform_input is None:
            self._target_transform = lambda x: x
        elif callable(transform_input):
            self._target_transform = transform_input
        else:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for `target_transform`"
            )

    # === Functions ===
    def _validate_transform(
        self,
        item: Any,
        transform: Optional[Callable],
        is_target: bool = False,
    ):
        """Called once per `transform` from `complete_initialization`

        Args:
            item: from `self._controller._get_nontransformed_item`
            transform: `Callable` given at instantiation of cls
            is_target: To identify `transform` from `target_transform`

        Raises:
            FedbiomedError: if there is a problem applying transforms
            FedbiomedError: if transforms do not return expected type
        """
        native_to_framework_transform = (
            self.native_to_framework_target_transform
            if is_target is True
            else self.native_to_framework_transform
        )

        try:
            item = native_to_framework_transform(item)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Unable to apply "
                f"`native_to_framework_{'target_' if is_target else ''}transform`"
            ) from e

        if not isinstance(item, self._to_format.value):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected "
                f"`native_to_framework_{'target_' if is_target else ''}transform` "
                f"to return `{self._to_format.value}`, got {type(item).__name__}"
            )

        try:
            item = transform(item)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Unable to apply "
                f"`{'target_' if is_target else ''}transform` to "
                f"`{'target' if is_target else 'data'}`"
            ) from e

        if not isinstance(item, self._to_format.value):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected "
                f"`{'target_' if is_target else ''}transform` to return "
                f"`{self._to_format.value}`, got {type(item).__name__}"
            )

    def complete_initialization(
        self,
        controller_kwargs: Dict[str, Any],
        to_format: DataReturnFormat,
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            controller_kwargs: arguments to create controller
            to_format: format associated to expected return format
        """
        self.to_format = to_format
        self._init_controller(controller_kwargs=controller_kwargs)

        # Recover sample and validate consistency of transforms
        sample = self._controller._get_nontransformed_item(0)
        self._validate_transform(sample["data"], transform=self.transform)
        self._validate_transform(
            sample["target"],
            transform=self.target_transform,
            is_target=True,
        )

    def __getitem__(self, idx: int) -> tuple[DatasetDataItem, DatasetDataItem]:
        sample = self._controller._get_nontransformed_item(idx)

        data = self.native_to_framework_transform(sample["data"])
        try:
            data = self.transform(data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply `transform` to `data` "
                f"from sample (index={idx}) in {self._to_format.value} format."
            ) from e

        target = self.native_to_framework_target_transform(sample["target"])
        try:
            target = self.target_transform(target)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply `target_transform` to "
                f"`target` from sample (index={idx}) in {self._to_format.value} format."
            ) from e

        return data, target


class ImageFolderDataset(SimpleDataset):
    _controller_cls = ImageFolderController


class MedNistDataset(SimpleDataset):
    _controller_cls = MedNistController


class MnistDataset(SimpleDataset):
    _controller_cls = MnistController

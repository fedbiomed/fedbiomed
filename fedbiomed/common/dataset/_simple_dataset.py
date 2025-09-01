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
        self._transform = self._validate_transform(transform)
        self._target_transform = self._validate_transform(target_transform)

    # === Functions ===
    def _validate_transform(self, transform_input: Optional[Callable]):
        """Raises FedbiomedValueError if transform_input is not valid"""
        if transform_input is None:
            return lambda x: x
        elif callable(transform_input):
            return transform_input
        else:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for transform input"
            )

    def _validate_pipeline(
        self,
        data: Any,
        transform: Optional[Callable],
        is_target: bool = False,
    ):
        """Called once per `transform` from `complete_initialization`

        Args:
            data: from `self._controller._get_nontransformed_item`
            transform: `Callable` given at instantiation of cls
            is_target: To identify `transform` from `target_transform`

        Raises:
            FedbiomedError: if there is a problem applying transforms
            FedbiomedError: if transforms do not return expected type
        """
        native_to_framework_transform = (
            self._native_to_framework_target_transform[self._to_format]
            if is_target is True
            else self._native_to_framework_transform[self._to_format]
        )

        try:
            data = native_to_framework_transform(data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Unable to perform type conversion of "
                f"{'target' if is_target else 'data'} to {self._to_format.value}"
            ) from e

        if not isinstance(data, self._to_format.value):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                f"Expected type conversion of {'target' if is_target else 'data'}"
                f"to return `{self._to_format.value}`, got {type(data).__name__}"
            )

        try:
            data = transform(data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Unable to apply "
                f"`{'target_' if is_target else ''}transform` to "
                f"`{'target' if is_target else 'data'}`"
            ) from e

        if not isinstance(data, self._to_format.value):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected "
                f"`{'target_' if is_target else ''}transform` to return "
                f"`{self._to_format.value}`, got {type(data).__name__}"
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
        self._validate_pipeline(sample["data"], transform=self._transform)
        self._validate_pipeline(
            sample["target"],
            transform=self._target_transform,
            is_target=True,
        )

    def __getitem__(self, idx: int) -> tuple[DatasetDataItem, DatasetDataItem]:
        if self._controller is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Dataset object has not completed "
                "initialization. It is not ready to use yet."
            )

        sample = self._controller._get_nontransformed_item(idx)

        try:
            data = self._transform(
                self._native_to_framework_transform[self._to_format](sample["data"])
            )
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply `transform` to `data` "
                f"from sample (index={idx}) in {self._to_format.value} format."
            ) from e

        try:
            target = self._target_transform(
                self._native_to_framework_target_transform[self._to_format](
                    sample["target"]
                )
            )
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

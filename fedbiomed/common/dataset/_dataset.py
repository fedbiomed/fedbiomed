# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller import Controller
from fedbiomed.common.dataset_types import DataReturnFormat, DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError


class Dataset(ABC):
    _native_to_framework: Dict[DataReturnFormat, Callable]
    _controller_cls: Type[Controller] = None
    _controller: Controller = None
    _to_format: DataReturnFormat = None

    transform: Optional[Union[Callable, Dict[str, Callable]]] = None
    target_transform: Optional[Union[Callable, Dict[str, Callable]]] = None

    @property
    def to_format(self) -> DataReturnFormat:
        return self._to_format

    @to_format.setter
    def to_format(self, format: DataReturnFormat):
        """Setter for `to_format` property

        Args:
            format: expected format of the data returned by `__getitem__`

        Raises:
            FedbiomedValueError: if `format` is not of type `DataReturnFormat`
        """
        if not isinstance(format, DataReturnFormat):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: `to_format` is not `DataReturnFormat` type"
            )
        self._to_format = format

    # === Abstract functions ===
    @abstractmethod
    def complete_initialization(self) -> None:
        """Finalize initialization of object to be able to recover items"""
        # Recover sample and validate consistency of transforms
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[DatasetDataItem, DatasetDataItem]:
        pass

    def _get_format_conversion_callable(self) -> Callable:
        """Get conversion function from data to expected format

        Returns:
            Callable function to convert data to expected format
        """
        if not self._to_format:
            raise AttributeError(
                f"{ErrorNumbers.FB632.value}: `to_format` is not set. "
                "Please set it before using the dataset. E.g., `dataset.to_format = "
                "fedbiomed.common.dataset_types.DataReturnFormat.TORCH`"
            )

        return self._native_to_framework[self._to_format]

    def _validate_transform(self, transform: Optional[Callable]) -> Callable:
        """Validates `transform` input

        Args:
            transform: transform to validate

        Returns:
            Validated transform

        Raises:
            FedbiomedValueError: if transform is not None or Callable
        """
        if transform is None:
            return lambda x: x

        if callable(transform):
            return transform
        else:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for transform input proived by user."
            )

    def _validate_format_conversion(
        self, data: Any, extra_info: Optional[str] = None
    ) -> Any:
        """Validates format conversion

        Args:
            data: from `self._controller.get_sample`
            extra_info: info to add to error message to indicate concerned variables

        Returns:
            Transformed data
        """
        extra_info = "" if extra_info is None else extra_info
        converter = self._get_format_conversion_callable()

        try:
            data = converter(data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Unable to perform type conversion of "
                f"data to {self._to_format.value}. {extra_info}"
            ) from e

        if not isinstance(data, self._to_format.value):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                f"Expected type conversion for the data to return "
                f"`{self._to_format.value}`, got {type(data).__name__}. {extra_info}"
            )

        return data

    def _validate_transformation(
        self, data: Any, transform: Optional[Callable], extra_info: Optional[str] = None
    ) -> Any:
        """Validates and applies `transform`

        Args:
            data: from `self._controller.get_sample`
            transform: `Callable` given at instantiation of cls
            extra_info: info to add to error message to indicate concerned variables

        Returns:
            Transformed data
        """
        extra_info = "" if extra_info is None else extra_info
        try:
            data = transform(data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Unable to apply transform. "
                f"{extra_info if extra_info else ''}"
            ) from e

        if not isinstance(data, self._to_format.value):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected "
                f"`transform` to return `{self._to_format.value}`, "
                f"got {type(data).__name__}. {extra_info}"
            )

        return data

    def _validate_format_and_transformations(
        self, data: Any, transform: Optional[Callable], label: Optional[str] = None
    ) -> Any:
        """Validates and applies format conversion and `transform`

        Args:
            data: from `self._controller.get_sample`
            transform: `Callable` given at instantiation of cls
            label: to add to error message to indicate concerned variables

        Returns:
            Transformed data
        """
        by = " " if label is None else f" by '{label}' "
        data = self._validate_format_conversion(
            data,
            extra_info=f"Error raised{by}in format conversion step.",
        )
        data = self._validate_transformation(
            data,
            transform,
            extra_info=f"Error raised{by}when applying associated transform",
        )
        return data

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

    def apply_transforms(self, sample: Dict[str, Any]) -> None:
        """Apply transforms to sample in place

        Args:
            sample: sample returned by `self._controller.get_sample`

        Raises:
            FedbiomedError: if there is a problem applying `transform` or `target_transform`
        """
        try:
            sample["data"] = self._transform(
                self._get_format_conversion_callable()(sample["data"])
            )
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply `transform` to `data` "
                f"in sample in {self._to_format.value} format."
            ) from e

        try:
            sample["target"] = self._target_transform(
                self._get_format_conversion_callable()(sample["target"])
            )
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply `target_transform` to "
                f"`target` in sample in {self._to_format.value} format."
            ) from e

        return sample

    def __len__(self) -> int:
        return len(self._controller)

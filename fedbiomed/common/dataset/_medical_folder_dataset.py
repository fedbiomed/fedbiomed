# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller import MedicalFolderController
from fedbiomed.common.dataset_types import DataReturnFormat, DatasetDataItem, Transform
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

from ._dataset import Dataset


class MedicalFolderDataset(Dataset):
    _controller_cls = MedicalFolderController

    _native_to_framework = {
        DataReturnFormat.SKLEARN: lambda x: x.get_fdata(),
        DataReturnFormat.TORCH: lambda x: torch.from_numpy(x.get_fdata()),
    }

    def __init__(
        self,
        data_modalities: Union[str, Iterable[str]],
        target_modalities: Optional[Union[str, Iterable[str]]],
        transform: Transform = None,
        target_transform: Transform = None,
    ):
        """Initializes the MedicalFolderDataset.

        Args:
            data_modalities (Union[str, Iterable[str]]): The data modalities to use.
            target_modalities (Optional[Union[str, Iterable[str]]]): The target modalities to use.
            transform (Transform, optional): The transform to apply to the data. Defaults to None.
            target_transform (Transform, optional): The transform to apply to the target data. Defaults to None.

        Raises:
            FedbiomedValueError:
            - If the input modalities are not valid.
            - If `data_modalities` is empty.
            - If `target_transform` is given but `target_modalities` is None\
        """
        if not data_modalities:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: `data_modalities` cannot be empty"
            )

        self._data_modalities = self._normalize_modalities(data_modalities)
        self._target_modalities = (
            None
            if target_modalities is None
            else self._normalize_modalities(target_modalities)
        )

        self._transform = self._validate_transform(
            transform=transform,
            modalities=self._data_modalities,
        )

        if self._target_modalities is None:
            if target_transform is not None:
                raise FedbiomedValueError(
                    f"{ErrorNumbers.FB632.value}: `target_transform` provided but "
                    "`target_modalities` is None"
                )
            else:
                self._target_transform = None
        else:
            self._target_transform = self._validate_transform(
                transform=target_transform,
                modalities=self._target_modalities,
            )

    # === Functions ===
    @staticmethod
    def _normalize_modalities(modalities: Union[str, Iterable[str]]) -> set[str]:
        """Validates `modalities` and returns them in type `set`

        Raises:
            FedbiomedError: If the input does not math the types expected
        """
        if isinstance(modalities, str):
            return set([modalities])
        if (
            not isinstance(modalities, dict)
            and isinstance(modalities, Iterable)
            and all(isinstance(item, str) for item in modalities)
        ):
            return set(modalities)
        raise FedbiomedError(
            f"{ErrorNumbers.FB613.value}: Unexpected type for modalities. "
            f"Expected `str` or `Iterable[str]`, got {type(modalities).__name__}"
        )

    @staticmethod
    def _validate_transform(
        transform: Transform,
        modalities: set[str],
    ) -> Union[Callable, Dict[str, Callable]]:
        """Validates `transform` given at instantiation"""
        transform = {} if transform is None else transform
        if isinstance(transform, dict):
            if not set(transform.keys()).issubset(modalities):
                raise FedbiomedValueError(
                    f"{ErrorNumbers.FB632.value}: Unexpected keys in transform. "
                    f"Expected keys to be in modalities, got {list(transform.keys())}"
                )
            if not all(callable(_v) for _v in transform.values()):
                raise FedbiomedValueError(
                    f"{ErrorNumbers.FB632.value}: dict must map strings to callables"
                )
            # Ensure all modalities have a transform (identity if not given)
            return {
                _k: (transform[_k] if _k in transform else lambda x: x)
                for _k in modalities
            }
        elif callable(transform):
            return transform
        else:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for `transform`"
            )

    def _validate_format_and_transformations(
        self,
        data: Dict[str, Any],
        transform: Transform,
        is_target: bool = False,
    ) -> Dict[str, Callable]:
        """Called once per `transform` from `complete_initialization`

        Args:
            data: from `self._controller.get_sample`
            transform: `Callable` given at instantiation of cls
            is_target: To identify `transform` from `target_transform`

        Raises:
            FedbiomedValueError: if `data` is not a `dict`
            FedbiomedError:
            - if there is a problem applying `transform`
            - if `transform` do not return expected type
        """
        if not isinstance(data, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected "
                f"`{'target' if is_target else 'data'}` to be a `dict`, got "
                f"{type(data).__name__}"
            )
        modalities = set(data.keys())

        for modality in (_mod for _mod in modalities if _mod != "demographics"):
            data[modality] = self._validate_format_conversion(
                data[modality], extra_info=f"Modality: {modality}"
            )

        # Apply and validate transforms
        if isinstance(transform, dict):
            for modality in modalities:
                if modality != "demographics":
                    self._validate_transformation(
                        data=data[modality],
                        transform=transform[modality],
                        extra_info=f"Error raised by modality: '{modality}'",
                    )
                else:
                    try:
                        _ = transform["demographics"](data["demographics"])
                    except Exception as e:
                        raise FedbiomedError(
                            f"{ErrorNumbers.FB632.value}: Failed to apply transform "
                            f"to 'demographics' in {self._to_format.value} format."
                        ) from e
        else:
            # transform: apply to the entire data dict
            try:
                data = transform(data)
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply "
                    f"{'target_transform' if is_target else 'transform'} to entire data dict"
                ) from e

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
        sample = self._controller.get_sample(0)
        self._validate_format_and_transformations(
            {modality: sample[modality] for modality in self._data_modalities},
            transform=self._transform,
        )
        if self._target_modalities is not None:
            self._validate_format_and_transformations(
                {modality: sample[modality] for modality in self._target_modalities},
                transform=self._target_transform,
                is_target=True,
            )

    def _process_sample_data(
        self,
        sample: dict,
        modalities: set[str],
        transform: Transform,
        idx: int,
        is_target: bool = False,
    ) -> DatasetDataItem:
        """Process sample with format conversion and transforms"""
        transform_type = "target_transform" if is_target else "transform"

        # Format conversion
        data = {
            modality: (
                self._get_default_types_callable()(
                    self._get_format_conversion_callable()(sample[modality])
                )
                if modality != "demographics"
                else sample[modality]
            )
            for modality in modalities
        }

        # Apply transforms
        if isinstance(transform, dict):
            for modality in modalities:
                try:
                    data[modality] = transform[modality](data[modality])
                except Exception as e:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Failed to apply "
                        f"`{transform_type}` to modality '{modality}' from sample "
                        f"(index={idx}) in {self._to_format.value} format."
                    ) from e

                try:
                    data[modality] = self._get_default_types_callable()(data[modality])
                except Exception as e:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Failed to apply "
                        f"`default training plan types to modality '{modality}' from sample "
                        f"(index={idx}) in {self._to_format.value} format."
                    ) from e
        else:
            try:
                data = transform(data)
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply `{transform_type}` "
                    f"to sample (index={idx}) in {self._to_format.value} format."
                ) from e

            try:
                for modality in modalities:
                    data[modality] = self._get_default_types_callable()(data[modality])
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply default training plan types "
                    f"to sample (index={idx}) in {self._to_format.value} format."
                ) from e
        return data

    def __getitem__(self, idx: int) -> tuple[DatasetDataItem, DatasetDataItem]:
        """Get a sample from the dataset with proper preprocessing and transforms"""
        if self._controller is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Dataset object has not completed "
                "initialization. It is not ready to use yet."
            )

        sample = self._controller.get_sample(idx)

        # Process data
        data = self._process_sample_data(
            sample, self._data_modalities, self._transform, idx
        )

        # Process target if any
        target = (
            self._process_sample_data(
                sample,
                self._target_modalities,
                self._target_transform,
                idx,
                is_target=True,
            )
            if self._target_modalities is not None
            else None
        )

        return data, target

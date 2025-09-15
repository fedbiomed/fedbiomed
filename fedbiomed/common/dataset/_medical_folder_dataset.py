# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import inspect
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
        demographics_transform: Optional[Callable] = None,
    ):
        self._data_modalities = self._normalize_modalities(data_modalities)
        self._target_modalities = (
            None
            if target_modalities is None
            else self._normalize_modalities(target_modalities)
        )

        self._transform = self._validate_transform(
            transform=transform,
            modalities=self._data_modalities,
            demographics_transform=demographics_transform,
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
            return {modalities}
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
    def _is_whole_dict_transform(transform):
        """Detects if a transform expects a dict (whole sample) or single modality"""
        if not callable(transform):
            return False
        sig = inspect.signature(transform)
        params = sig.parameters
        return len(params) == 1

    @staticmethod
    def _validate_transform(
        transform: Transform,
        modalities: set[str],
        demographics_transform: Optional[Callable] = None,
    ) -> Union[Callable, Dict[str, Callable]]:
        """Turns `transform_input` into a `dict` that matches `modalities`

        Raises:
            FedbiomedValueError:
                - if input is not in `[None, Callable, Dict[str, Callable]`
                - if input is `dict` and not all `modalities` are present
                - if both `transform` as whole-dict callable and `demographics_transform` are given
                - if `demographics_transform` is not a `Callable`
                - if `demographics_transform` is given but `demographics` not in `modalities`
        """
        if callable(transform) and MedicalFolderDataset._is_whole_dict_transform(
            transform
        ):
            if demographics_transform is not None:
                raise FedbiomedValueError(
                    f"{ErrorNumbers.FB632.value}: Cannot use both `transform` as a "
                    "whole-dict callable and `demographics_transform`. Modify "
                    "`transform` to handle `demographics` if needed."
                )
            return transform  # Return as a single callable

        if transform is None:
            transform = {_k: (lambda x: x) for _k in modalities}
        elif isinstance(transform, dict):
            if not all(callable(_v) for _v in transform.values()):
                raise FedbiomedValueError(
                    ErrorNumbers.FB632.value
                    + ": `transform` dictionary must map strings to callables"
                )
            if "demographics" in transform and demographics_transform is not None:
                raise FedbiomedValueError(
                    f"{ErrorNumbers.FB632.value}: Redundancy found, cannot use both "
                    "`transform['demographics']` and `demographics_transform`. "
                )
        else:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for `transform`"
            )

        if demographics_transform is not None:
            if not callable(demographics_transform):
                raise FedbiomedValueError(
                    f"{ErrorNumbers.FB632.value}: `demographics_transform` must be "
                    f"a callable, got {type(demographics_transform).__name__}"
                )
            if "demographics" not in modalities:
                raise FedbiomedValueError(
                    f"{ErrorNumbers.FB632.value}: `demographics_transform` provided but "
                    f"key `demographics` not in `data_modalities`"
                )
            transform["demographics"] = demographics_transform

        missing_transforms = modalities.difference(transform.keys())
        if missing_transforms:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Missing modalities in "
                f"`transform`: {missing_transforms}"
            )

        return transform

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
                self._validate_transformation(
                    data=data[modality],
                    transform=transform[modality],
                    extra_info=f"Modality: '{modality}'",
                )
        else:
            # Whole-dict transform: apply to the entire data dict
            try:
                data = transform(data)
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply whole-dict "
                    f"{'target_transform' if is_target else 'transform'}"
                ) from e

            if not isinstance(data, self._to_format.value):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Expected "
                    f"transform to return `{self._to_format.value}`, "
                    f"got {type(data).__name__}."
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

    def __getitem__(self, idx: int) -> tuple[DatasetDataItem, DatasetDataItem]:
        if self._controller is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Dataset object has not completed "
                "initialization. It is not ready to use yet."
            )

        sample = self._controller.get_sample(idx)

        # Prepare data and target dicts
        data = {
            modality: (
                self._get_format_conversion_callable()(sample[modality])
                if modality != "demographics"
                else sample[modality]
            )
            for modality in self._data_modalities
        }

        target = (
            None
            if self._target_modalities is None
            else {
                modality: self._get_format_conversion_callable()(sample[modality])
                for modality in self._target_modalities
            }
        )

        # Apply transforms
        # If transform is a dict, apply per-modality; if callable, apply to whole dict
        if isinstance(self._transform, dict):
            for modality in self._data_modalities:
                try:
                    data[modality] = self._transform[modality](data[modality])
                except Exception as e:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Failed to apply `transform` "
                        f"to modality '{modality}' from sample (index={idx}) in "
                        f"{self._to_format.value} format."
                    ) from e
        else:
            try:
                data = self._transform(data)
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply whole-dict `transform` "
                    f"to sample (index={idx}) in {self._to_format.value} format."
                ) from e

        if self._target_modalities is not None:
            if isinstance(self._target_transform, dict):
                for modality in self._target_modalities:
                    try:
                        target[modality] = self._target_transform[modality](
                            target[modality]
                        )
                    except Exception as e:
                        raise FedbiomedError(
                            f"{ErrorNumbers.FB632.value}: Failed to apply `target_transform` "
                            f"to modality '{modality}'from sample (index={idx}) in "
                            f"{self._to_format.value} format."
                        ) from e
            else:
                try:
                    target = self._target_transform(target)
                except Exception as e:
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB632.value}: Failed to apply whole-dict `target_transform` "
                        f"to sample (index={idx}) in {self._to_format.value} format."
                    ) from e

        return data, target

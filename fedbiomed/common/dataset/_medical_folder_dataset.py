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
        target_modalities: Union[str, Iterable[str]],
        transform: Transform = None,
        target_transform: Transform = None,
        demographics_transform: Optional[Callable] = None,
    ):
        self._data_modalities = self._normalize_modalities(data_modalities)
        self._target_modalities = self._normalize_modalities(target_modalities)

        self._transform = self._validate_transform(
            transform=transform,
            modalities=self._data_modalities,
        )
        self._target_transform = self._validate_transform(
            transform=target_transform,
            modalities=self._target_modalities,
        )

        if demographics_transform is not None and callable(demographics_transform):
            if "demographics" in self._data_modalities:
                self._data_modalities.add("demographics")
                self._transform["demographics"] = demographics_transform

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
    def _validate_transform(transform: Transform, modalities: set[str]):
        """Turns `transform_input` into a `dict` that matches `modalities`

        Raises:
            FedbiomedValueError:
                - if input is not in `[None, Callable, Dict[str, Callable]`
                - if input is `dict` and not all `modalities` are present
        """
        if transform is None:
            return {_k: (lambda x: x) for _k in modalities}

        elif callable(transform):
            return {_k: transform for _k in modalities}

        elif isinstance(transform, dict):
            if not all(callable(_v) for _v in transform.values()):
                raise FedbiomedValueError(
                    ErrorNumbers.FB632.value
                    + ": `transform` dictionary must map strings to callables"
                )

            missing_transforms = modalities.difference(transform.keys())
            if missing_transforms:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Missing modalities in "
                    f"`transform`: {missing_transforms}"
                )

            return transform
        else:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for `transform`"
            )

    def _validate_pipeline(
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
                data[modality], for_=f"modality {modality}"
            )

        for modality in modalities:
            _name = (
                "demographics_"
                if modality == "demographics"
                else ("target_" if is_target is True else "")
            )
            self._validate_transformation(
                transform=transform[modality],
                data=data[modality],
                extra_info=f"for modality '{modality}'",
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
        self._validate_pipeline(
            {modality: sample[modality] for modality in self._data_modalities},
            transform=self._transform,
        )
        self._validate_pipeline(
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

        data = {}
        for modality in self._data_modalities:
            data[modality] = (
                self._get_format_conversion_callable()(sample[modality])
                if modality != "demographics"
                else sample[modality]
            )
            try:
                data[modality] = self._transform[modality](data[modality])
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply `transform` "
                    f"to modality '{modality}' from sample (index={idx}) in "
                    f"{self._to_format.value} format."
                ) from e

        target = {}
        for modality in self._target_modalities:
            target[modality] = self._get_format_conversion_callable()(sample[modality])
            try:
                target[modality] = self._target_transform[modality](target[modality])
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply `target_transform` "
                    f"to modality '{modality}'from sample (index={idx}) in "
                    f"{self._to_format.value} format."
                ) from e

        return data, target

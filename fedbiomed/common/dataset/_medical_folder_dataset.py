from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller import MedicalFolderController
from fedbiomed.common.dataset_types import DataReturnFormat, DatasetDataItem, Transform
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

from ._dataset import Dataset


class MedicalFolderDataset(Dataset):
    _controller_cls = MedicalFolderController
    # To go from metatensor to `np.ndarray` and `torch.Tensor`
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
        self.data_modalities = data_modalities
        self.target_modalities = target_modalities
        self.transform = transform
        self.target_transform = target_transform

        if demographics_transform is not None and callable(demographics_transform):
            if "demographics" in self.data_modalities:
                self._data_modalities.add("demographics")
                self._transform["demographics"] = demographics_transform

    # === Properties ===
    @property
    def data_modalities(self):
        return self._data_modalities

    @data_modalities.setter
    def data_modalities(self, modalities: Union[str, Iterable[str]]) -> set:
        self._data_modalities = self._controller_cls._normalize_modalities(modalities)

    @property
    def target_modalities(self):
        return self._target_modalities

    @target_modalities.setter
    def target_modalities(self, modalities: Union[str, Iterable[str]]) -> set:
        self._target_modalities = self._controller_cls._normalize_modalities(modalities)

    @property
    def modalities(self):
        return self._data_modalities.union(self._target_modalities)

    @property
    def native_to_framework(self):
        return self._native_to_framework[self._to_format]

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform_input: Transform):
        """Turns `transform_input` into a `dict` that matches `self.data_modalities`

        Raises:
            FedbiomedValueError:
                - if input is not in `[None, Callable, Dict[str, Callable]`
                - if input is `dict` and not all `self.data_modalities` are present
        """
        if transform_input is None:
            self._transform = {_k: (lambda x: x) for _k in self.data_modalities}

        elif callable(transform_input):
            self._transform = {_k: transform_input for _k in self.data_modalities}

        elif isinstance(transform_input, dict):
            if not all(callable(_v) for _v in transform_input.values()):
                raise FedbiomedValueError(
                    ErrorNumbers.FB632.value
                    + ": `transform` dictionary must map strings to callables"
                )

            missing_transforms = self.data_modalities - transform_input.keys()
            if missing_transforms:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Missing modalities in "
                    f"`transform`: {missing_transforms}"
                )

            self._transform = transform_input
        else:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for `transform`"
            )

    @property
    def target_transform(self):
        return self._target_transform

    @target_transform.setter
    def target_transform(self, transform_input: Transform):
        """Turns `transform_input` into a `dict` that matches `self.target_modalities`

        Raises:
            FedbiomedValueError:
                - if input is not in `[None, Callable, Dict[str, Callable]`
                - if input is `dict` and not all `self.target_modalities` are present
        """
        if transform_input is None:
            self._target_transform = {
                _k: (lambda x: x) for _k in self.target_modalities
            }

        elif callable(transform_input):
            self._target_transform = {
                _k: transform_input for _k in self.target_modalities
            }

        elif isinstance(transform_input, dict):
            if not all(callable(_v) for _v in transform_input.values()):
                raise FedbiomedValueError(
                    ErrorNumbers.FB632.value
                    + ": `target_transform` dictionary must map strings to callables"
                )

            missing_transforms = self.target_modalities - transform_input.keys()
            if missing_transforms:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Missing modalities in "
                    f"`target_transform`: {missing_transforms}"
                )

            self._target_transform = transform_input
        else:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for `target_transform`"
            )

    @property
    def demographics_transform(self):
        if "demographics" in self._data_modalities:
            return self._transform["demographics"]
        return None

    # === Functions ===
    def _validate_transform(
        self,
        item: Dict[str, Any],
        transform: Transform,
        is_target: bool = False,
    ) -> Dict[str, Callable]:
        """Called once per `transform` from `complete_initialization`

        Args:
            item: from `self._controller._get_nontransformed_item`
            transform: `Callable` given at instantiation of cls
            is_target: To identify `transform` from `target_transform`

        Raises:
            FedbiomedValueError:
            - if `item` is not a `dict`
            FedbiomedError:
            - if there is a problem applying `transform`
            - if `transform` do not return expected type
        """
        if not isinstance(item, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected "
                f"`{'target' if is_target else 'data'}` to be a `dict`, got "
                f"{type(item).__name__}"
            )
        modalities = set(item.keys())

        for modality in (_mod for _mod in modalities if _mod != "demographics"):
            try:
                item[modality] = self.native_to_framework(item[modality])
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Unable to apply `native_to_framework`"
                ) from e

            if not isinstance(item[modality], self._to_format.value):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Expected `native_to_framework` "
                    f"to return `{self._to_format.value}`, got "
                    f"{type(item[modality]).__name__} for modality: '{modality}'"
                )

        for modality in modalities:
            _name = (
                "demographics_"
                if modality == "demographics"
                else ("target_" if is_target is True else "")
            )
            try:
                item[modality] = transform[modality](item[modality])
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Unable to apply "
                    f"`{_name}transform` to `{modality}`"
                ) from e
            if not isinstance(item[modality], self._to_format.value):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Expected "
                    f"`{_name}transform` to return `{self._to_format.value}`, got "
                    f"{type(item[modality]).__name__} for modality: '{modality}'"
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
        self._validate_transform(
            {modality: sample[modality] for modality in self.data_modalities},
            transform=self.transform,
        )
        self._validate_transform(
            {modality: sample[modality] for modality in self.target_modalities},
            transform=self.target_transform,
            is_target=True,
        )

    def __getitem__(self, idx: int) -> tuple[DatasetDataItem, DatasetDataItem]:
        sample = self._controller._get_nontransformed_item(idx)

        data = {}
        for modality in self.data_modalities:
            data[modality] = (
                self.native_to_framework(data["modality"])
                if modality != "demographics"
                else sample[modality]
            )
            try:
                data[modality] = self.transform[modality](data[modality])
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply `transform` "
                    f"to modality '{modality}' from sample (index={idx}) in "
                    f"{self._to_format.value} format."
                ) from e

        target = {}
        for modality in self.target_modalities:
            try:
                target[modality] = self.target_transform[modality](sample[modality])
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply `target_transform` "
                    f"to modality '{modality}'from sample (index={idx}) in "
                    f"{self._to_format.value} format."
                ) from e

        return data, target

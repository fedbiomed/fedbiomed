from typing import Any, Callable, Dict, Iterable, Union

import torchvision.transforms as T
from monai.transforms import ToNumpy

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller import MedicalFolderController
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

from ._dataset import Dataset
from ._dataset_types import DataReturnFormat, Transform


class MedicalFolderDataset(Dataset):
    _controller_cls = MedicalFolderController
    # To go from metatensor to `np.ndarray` and `torch.Tensor`
    _native_to_framework_transform = {
        DataReturnFormat.SKLEARN: ToNumpy(),
        DataReturnFormat.TORCH: lambda x: x.as_tensor(),
    }

    def __init__(
        self,
        data_modalities: Union[str, Iterable[str]],
        target_modalities: Union[str, Iterable[str]],
        transform: Transform = None,
        target_transform: Transform = None,
        # demographics_transform: Optional[Callable] = None,
    ):
        self.data_modalities = data_modalities
        self.target_modalities = target_modalities
        self.transform = transform
        self.target_transform = target_transform

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
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform_input: Transform):
        if transform_input is not None:
            self._validate_transform_input(transform_input)
        self._transform = transform_input

    @property
    def target_transform(self):
        return self._target_transform

    @target_transform.setter
    def target_transform(self, transform_input: Transform):
        if transform_input is not None:
            self._validate_transform_input(transform_input)
        self._target_transform = transform_input

    def _validate_transform_input(self, transform_input: Transform) -> None:
        """Validates transform inputs

        Raises:
            FedbiomedValueError:
                - if input is not `Callable` or `Dict[str, Callable]`
                - if input is `dict` and not all expected 'modalities' are present
        """
        if not callable(transform_input):
            if not isinstance(transform_input, dict):
                raise FedbiomedValueError(
                    f"{ErrorNumbers.FB632.value}: Unexpected `Transform` input"
                )
            elif not all(
                isinstance(_k, str) and callable(_v)
                for _k, _v in transform_input.items()
            ):
                raise FedbiomedValueError(
                    ErrorNumbers.FB632.value
                    + ": Transform dictionary must map strings to callables"
                )

    # === Functions ===
    def _update_transform(
        self,
        data: Dict[str, Any],
        transform: Transform,
        is_target: bool = False,
    ) -> Dict[str, Callable]:
        """Called once per `transform` from `complete_initialization`

        Args:
            data: from `self._controller._get_nontransformed_item`
            transform: `Callable` given at instantiation of cls
            is_target: To identify `transform` from `target_transform`

        Raises:
            FedbiomedValueError:
            - if `data` is not a `dict`
            FedbiomedError:
            - if `transform` is a `dict` and does not contain all modalities in `data`
            - if there is a problem applying `transform`
            - if `transform` do not return expected type

        Returns:
            `transform` ready to use in `__getitem__`
        """
        if not isinstance(data, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected "
                f"`{'target' if is_target else 'data'}` to be a `dict`, got "
                f"{type(data).__name__}"
            )
        modalities = set(data.keys())

        native_to_framework_transform = {
            modality: self._native_to_framework_transform[self._to_format]
            for modality in modalities
        }

        if transform is None:
            transform = native_to_framework_transform
        elif callable(transform):
            transform = {modality: transform for modality in modalities}
        elif isinstance(transform, dict):
            missing_transforms = modalities.difference(set(transform.keys()))
            if missing_transforms:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Missing modalities in "
                    f"`{'target_' if is_target else ''}transform`: {missing_transforms}"
                )
        else:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for "
                f"`{'target_' if is_target else ''}transform`"
            )

        for modality in modalities:
            for attempt in range(2):
                try:
                    item_modality = transform[modality](data[modality])
                    break
                except Exception as e:
                    if attempt == 0:
                        transform[modality] = T.Compose(
                            [
                                native_to_framework_transform[modality],
                                transform[modality],
                            ]
                        )
                    else:
                        raise FedbiomedError(
                            f"{ErrorNumbers.FB632.value}: Unable to apply "
                            f"`{'target_' if is_target else ''}transform` to "
                            f"'{modality}': {e}"
                        ) from e

            if not isinstance(item_modality, self._to_format.value):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Expected "
                    f"`{'target_' if is_target else ''}transform` to return "
                    f"`{self._to_format.value}`, got {type(item_modality).__name__} "
                    f"for modality '{modality}'"
                )

        return transform

    def _apply_transforms(
        self, sample: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        # TODO: demographics
        data = {}
        for modality in self.data_modalities:
            try:
                data[modality] = self.transform[modality](sample[modality])
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply `transform` "
                    f"to modality '{modality}'. {e}"
                ) from e

        target = {}
        for modality in self.target_modalities:
            try:
                target[modality] = self.target_transform[modality](sample[modality])
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply `target_transform` "
                    f"to modality '{modality}'. {e}"
                ) from e

        return data, target

    def complete_initialization(
        self,
        controller_kwargs: Dict[str, Any],
        to_format: DataReturnFormat,
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            controller_kwargs: arguments to create controller
            to_format: format associated to expected return format

        Raises:
            FedbiomedError: if `sample` returned by `_controller` is not a `dict`
            KeyError: if `sample` does not include all modalities
        """
        self.to_format = to_format
        self._init_controller(controller_kwargs=controller_kwargs)

        sample = self._controller._get_nontransformed_item(0)
        # update transforms if necessary --------------------------------------
        self.transform = self._update_transform(
            {modality: sample[modality] for modality in self.data_modalities},
            transform=self.transform,
        )
        self.target_transform = self._update_transform(
            {modality: sample[modality] for modality in self.target_modalities},
            transform=self.target_transform,
            is_target=True,
        )

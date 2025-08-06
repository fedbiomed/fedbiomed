from typing import Any, Callable, Dict, Iterable, Optional, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

from ._dataset import Dataset, Transform


class MedicalFolderDataset(Dataset):
    def __init__(
        self,
        data_modalities: Optional[Union[str, Iterable[str]]] = "T1",
        target_modalities: Optional[Union[str, Iterable[str]]] = "label",
        transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
        target_transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
        # demographics_transform: Optional[Callable] = None,
    ):
        self.data_modalities = data_modalities
        self.target_modalities = target_modalities
        self.transform = transform
        self.target_transform = target_transform

    @property
    def data_modalities(self):
        return self._data_modalities

    @data_modalities.setter
    def data_modalities(self, modalities: Union[str, Iterable[str]]) -> set:
        self._data_modalities = self._normalize_modalities(modalities)

    @property
    def target_modalities(self):
        return self._target_modalities

    @target_modalities.setter
    def target_modalities(self, modalities: Union[str, Iterable[str]]) -> set:
        self._target_modalities = self._normalize_modalities(modalities)

    def _normalize_modalities(self, modalities: Union[str, Iterable[str]]) -> set:
        """Validates `modalities`

        Returns:
            `modalities` in type `set`

        Raises:
            FedbiomedValueError: If the input does not math the types expected
        """
        if isinstance(modalities, str):
            return {modalities}
        if (
            not isinstance(modalities, dict)
            and isinstance(modalities, Iterable)
            and all(isinstance(item, str) for item in modalities)
        ):
            return set(modalities)
        raise FedbiomedValueError(
            f"{ErrorNumbers.FB613.value}: "
            "Unexpected type for modalities. Expected str or Iterable[str]"
        )

    @property
    def modalities(self):
        return self._data_modalities.union(self._target_modalities)

    @property
    def transform(self) -> Optional[Callable]:
        return self._transform

    @transform.setter
    def transform(self, transform_input: Optional[Callable]):
        self._validate_transform_input(transform_input)
        self._transform = transform_input

    @property
    def target_transform(self) -> Optional[Callable]:
        return self._target_transform

    @target_transform.setter
    def target_transform(self, transform_input: Optional[Callable]):
        self._validate_transform_input(transform_input)
        self._target_transform = transform_input

    def _normalize_transform_input(self, transform_input: Optional[Callable]) -> None:
        """Raises FedbiomedValueError if transform_input is not valid"""
        if not (transform_input is None or callable(transform_input)):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected `Transform` input"
            )

    def _validate_transform_input(self, transform_input: Transform) -> None:
        if transform_input is None or callable(transform_input):
            return transform_input
        elif isinstance(transform_input, dict):
            if not all(
                isinstance(k, str) and callable(v) for k, v in transform_input.items()
            ):
                raise FedbiomedError(
                    ErrorNumbers.FB632.value
                    + ": Transform dict must map strings to callables"
                )

        raise FedbiomedValueError(
            f"{ErrorNumbers.FB632.value}: Unexpected `Transform` input"
        )

    def _apply_transforms(self, sample: Dict[str, Any]):
        try:
            data = (
                sample["data"]
                if self.transform is None
                else self.transform(sample["data"])
            )
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply `transform`. {e}"
            ) from e

        try:
            target = (
                sample["target"]
                if sample["target"] is None or self.target_transform is None
                else self.target_transform(sample["target"])
            )
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply `target_transform`. {e}"
            ) from e

        return data, target

    def _validate(self, index: int):
        if self._controller is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Controller has not been initialized"
            )

        sample = self._get_nontransformed_item(index)
        if not isinstance(sample, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected `sample` to be a `dict`, got "
                f"{type(sample).__name__}"
            )
        if not all(_k in sample for _k in self.modalities):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Missing keys in `sample`. Expected "
                ", ".join(f"'{modality}'" for modality in self.modalities)
            )

        data, target = self._apply_transforms(sample)
        if not isinstance(data, dict) or not all(
            isinstance(_v, self._to_format.value) for _v in data.values()
        ):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected `transform` to return "
                f"`{self._to_format.value.__name__}`, got: "
                ", ".join(f"{_k}: {type(_v).__name__}" for _k, _v in data.items())
            )
        if not isinstance(target, dict) or not all(
            isinstance(_v, self._to_format.value) for _v in target.values()
        ):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected `target_transform` to return "
                f"`{self._to_format.value.__name__}`, got: "
                ", ".join(f"{_k}: {type(_v).__name__}" for _k, _v in target.items())
            )

        return data, target

from typing import Any, Callable, Dict, Optional

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

from ._dataset import DataReturnFormat, Dataset


class SimpleDataset(Dataset):
    "Dataset where data and target are implicitly predefined by the controller"

    def __init__(
        self,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        to_format: DataReturnFormat = DataReturnFormat.TORCH,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.to_format = to_format

    @property
    def transform(self) -> Optional[Callable]:
        return self._transform

    @transform.setter
    def transform(self, transform_input: Optional[Callable]):
        """Raises FedbiomedValueError if transform_input is not valid"""
        if not (transform_input is None or callable(transform_input)):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for `transform`"
            )
        self._transform = transform_input

    @property
    def target_transform(self) -> Optional[Callable]:
        return self._target_transform

    @target_transform.setter
    def target_transform(self, transform_input: Optional[Callable]):
        """Raises FedbiomedValueError if transform_input is not valid"""
        if not (transform_input is None or callable(transform_input)):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for `target_transform`"
            )
        self._target_transform = transform_input

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

    def _validate(self):
        if self._controller is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Controller has not been initialized"
            )

        sample = self._get_nontransformed_item(0)
        if not isinstance(sample, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected `sample` to be a `dict`, got "
                f"{type(sample).__name__}"
            )
        if not all(_k in sample for _k in ["data", "target"]):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Missing keys in `sample`. Expected "
                "'data' and 'target'"
            )

        data, target = self._apply_transforms(sample)
        if not isinstance(data, self._to_format.value):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected `transform` to return "
                f"`{self._to_format.value.__name__}`, got {type(data).__name__}"
            )
        if sample["target"] is not None and not isinstance(
            target, self._to_format.value
        ):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected `target_transform` to return "
                f"`{self._to_format.value.__name__}`, got {type(target).__name__}"
            )


class MnistDataset(SimpleDataset):
    pass


class ImageFolderDataset(SimpleDataset):
    pass

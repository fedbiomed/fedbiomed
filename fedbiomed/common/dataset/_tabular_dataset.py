from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Tuple, Union

if TYPE_CHECKING:
    import numpy as np
    import torch

import polars as pl

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset._dataset import Dataset
from fedbiomed.common.dataset_controller._tabular_controller import TabularController
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError


class TabularDataset(Dataset):
    _controller_cls: type = TabularController

    _native_to_framework = {
        DataReturnFormat.SKLEARN: lambda x: x.to_numpy(),
        DataReturnFormat.TORCH: lambda x: x.to_torch(),
    }

    def __init__(
        self,
        input_columns: Iterable | int | str,
        target_columns: Iterable | int | str,
        transform: Optional[Callable] = None,
    ) -> None:
        self._transform = self._validate_transform(transform_input=transform)
        self._input_columns = input_columns
        self._target_columns = target_columns

    # === Functions ===
    def _get_format_conversion_callable(self):
        return self._native_to_framework[self._to_format]

    def _validate_transform(self, transform_input: Optional[Callable]):
        """Turns `transform_input` into a `dict` that matches `modalities`

        Raises:
            FedbiomedValueError:
                - if input is not in `[None, Callable, Dict[str, Callable]`
                - if input is `dict` and not all `modalities` are present
        """
        if transform_input is None:
            return lambda x: x

        elif callable(transform_input):
            return transform_input

        else:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Unexpected type for `transform`"
            )

    def _validate_pipeline(
        self,
        data: Dict[str, pl.DataFrame],
        transform: Optional[Callable],
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

        try:
            data = self._get_format_conversion_callable()(data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Unable to perform type conversion to "
                f"{self._to_format.value}"
            ) from e

        if not isinstance(data, self._to_format.value):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected type conversion "
                f"to return `{self._to_format.value}`, got "
                f"{type(data).__name__}"
            )

        try:
            data = transform(data)  # type: ignore
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Unable to apply transform"
            ) from e

        if not isinstance(data, self._to_format.value):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected "
                f"`transform` to return `{self._to_format.value}`, "
                f"got {type(data).__name__} "
            )

    def complete_initialization(  # type: ignore
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
        controller_kwargs.update(
            {
                "input_columns": self._input_columns,
                "target_columns": self._target_columns,
            }
        )
        self._init_controller(controller_kwargs=controller_kwargs)

        # Recover sample and validate consistency of transforms
        sample = self._controller._get_nontransformed_item(0)  # type: ignore
        self._validate_pipeline(sample["data"], transform=self._transform)
        self._validate_pipeline(sample["target"], transform=self._transform)

    def _apply_transforms(self, sample: Dict[str, Any]) -> Tuple[Any, Any]:
        try:
            data = self._transform(
                self._get_format_conversion_callable()(sample["data"])
            )
            target = self._transform(
                self._get_format_conversion_callable()(sample["target"])
            )
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply transforms. {e}"
            ) from e

        return data, target

    def __getitem__(self, idx: int) -> Dict[str, Union["np.array", "torch.Tensor"]]:
        if self._controller is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: This dataset object has not completed initialization."
            )

        sample = self._controller._get_nontransformed_item(idx)  # type: ignore

        data = self._get_format_conversion_callable()(sample["data"])
        try:
            data = self._transform(data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply `transform` to `data` "
                f"from sample (index={idx}) in {self._to_format.value} format."
            ) from e

        target = self._get_format_conversion_callable()(sample["target"])
        try:
            target = self._transform(target)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply `_transform` to "
                f"`target` from sample (index={idx}) in {self._to_format.value} format."
            ) from e

        return {"data": data, "target": target}

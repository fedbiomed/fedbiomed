from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import polars as pl

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset._dataset import Dataset
from fedbiomed.common.dataset_reader import CsvReader
from fedbiomed.common.dataset_types import DataReturnFormat, DatasetDataItem
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError


class TabularDataset(Dataset):
    # _controller_cls: type = None
    # _controller = None
    # _to_format: DataReturnFormat = None

    _dataset_path: str | Path
    _reader: CsvReader
    _data: pl.DataFrame
    _target: pl.DataFrame

    def __init__(
        self,
        root: str | Path,
        input_labels: Iterable | int | str,
        target_labels: Iterable | int | str,
    ) -> None:
        super().__init__()
        self._reader = CsvReader(self.root)  # type: ignore
        all_row_indexes = list(range(self._reader._len))
        self._data = self._reader.get(all_row_indexes, input_labels)
        self._target = self._reader.get(all_row_indexes, target_labels)

    # === Properties ===
    @property
    def to_format(self) -> DataReturnFormat:
        return self._to_format

    @to_format.setter
    def to_format(self, to_format_input: DataReturnFormat):
        if not isinstance(to_format_input, DataReturnFormat):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: `to_format` is not `DataReturnFormat` type"
            )
        self._to_format = to_format_input

    def __getitem__(self, idx: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        if idx >= self.__len__():
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item at index {idx}"
            )
        return ({"data": self._data[idx]}, {"data": self._target[idx]})

    def __len__(self) -> int:
        return self._reader.len()

    # TODO: Change it with the accurate return type
    def shape(self) -> Dict:
        return self._reader.shape()

    def _apply_transforms(self, sample: Dict[str, Any]) -> Tuple[Any, Any]:
        return super()._apply_transforms(sample)

    def complete_initialization(self) -> None:
        return super().complete_initialization()

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

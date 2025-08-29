from pathlib import Path
from typing import Any, Dict, Iterable, Union

import polars as pl

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller._controller import Controller
from fedbiomed.common.dataset_reader._csv_reader import CsvReader
from fedbiomed.common.exceptions import FedbiomedError


class TabularController(Controller):
    _reader: CsvReader
    _data: pl.DataFrame
    _target: pl.DataFrame

    def __init__(
        self,
        root: Union[str, Path],
        input_labels: Iterable | int | str,
        target_labels: Iterable | int | str,
    ) -> None:
        """Constructor of the class

        Args:
            root: Root directory path

        Raises:
            FedbiomedError: if `root` does not exist
        """
        self.root = root
        self._reader = CsvReader(self.root)
        all_row_indexes = list(range(self._reader._len))
        self._data = self._reader.get(all_row_indexes, input_labels)
        self._target = self._reader.get(all_row_indexes, target_labels)

        self._controller_kwargs = {
            "root": str(self.root),
        }

    def _get_nontransformed_item(self, index: int) -> Dict[str, Any]:
        """Retrieve a data sample without applying transforms"""
        if index >= self.__len__():
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item at index {index}"
            )
        return {
            "data": self._data[index].to_numpy(),
            "target": self._target[index].to_numpy(),
        }

    def __len__(self) -> int:
        return self._reader.len()

    # TODO: Change it with the accurate return type
    def shape(self) -> Dict:
        return self._reader.shape()

from pathlib import Path
from typing import Dict, Iterable, Union

import polars as pl

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller._controller import Controller
from fedbiomed.common.dataset_reader._csv_reader import CsvReader
from fedbiomed.common.exceptions import FedbiomedError


class TabularController(Controller):
    _reader: CsvReader

    def __init__(
        self,
        root: Union[str, Path],
        input_columns: Iterable | int | str,
        target_columns: Iterable | int | str,
    ) -> None:
        """Constructor of the class

        Args:
            root: Root directory path

        Raises:
            FedbiomedError: if `root` does not exist
        """
        self.root = root
        self._reader = CsvReader(self.root)
        self._input_columns = input_columns
        self._target_columns = target_columns

    def _get_nontransformed_item(self, index: int) -> Dict[str, pl.DataFrame]:
        """Retrieve a data sample without applying transforms"""
        if index >= self.__len__():
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item at index {index}"
            )
        return {
            "data": self._reader.get(index, self._input_columns),
            "target": self._reader.get(index, self._target_columns),
        }

    def __len__(self) -> int:
        return self._reader.len()

    # TODO: Change it with the accurate return type
    def shape(self) -> Dict:
        return self._reader.shape()

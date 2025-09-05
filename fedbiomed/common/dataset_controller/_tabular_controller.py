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
    ) -> None:
        """Constructor of the class

        Args:
            root: Root directory path

        Raises:
            FedbiomedError: if `root` does not exist
        """
        self.root = root
        self._reader = CsvReader(self.root)

    def get_sample(self, index: int) -> pl.DataFrame:
        """Retrieve a data sample without applying transforms"""
        if index >= self.__len__():
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item at index {index}"
            )
        return self._reader.get(index)

    def normalize_columns(self, columns: Union[Iterable, int, str]) -> list[str]:
        """Validate and normalize `columns` to a list of column names

        Args:
            columns: Columns to normalize

        Raises:
            FedbiomedError: if `columns` is not valid

        Returns:
            List of column names
        """
        return self._reader.normalize_columns(columns=columns)

    def __len__(self) -> int:
        return self._reader.len()

    # TODO: Change it with the accurate return type
    def shape(self) -> Dict:
        return self._reader.shape()

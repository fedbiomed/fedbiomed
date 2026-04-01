# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader implementation for CSV file
"""
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from nipoppy import NipoppyDataRetriever

from fedbiomed.common.exceptions import FedbiomedError


class NipoppyReader:
    def __init__(
        self,
        path: Path,
    ) -> None:
        """Constructs the csv reader.

        Args:
            path: The path of the csv file that contains the dataset.
            has_header: Boolean to indicate whether the file has a header or not.
                By default it is set as 'auto', which is the case that the reader tries to
                detect itself whether the file has a header or not.
            delimiter: The delimiter used in the csv file.
                By default it is set as None, which is the case that the reader tries to
                detect itself whether the file has a delimiter or not.
        """
        self._path = path
        self._data: Optional[pd.DataFrame] = None  # lazy loaded

    def _read(self,
              phenotypes: Optional[List[str]] = None,
              derivatives: Optional[List[Tuple[str, str, str]]] = None) -> pd.DataFrame:
        """Reads all dataset and returns the dataframe.

        Returns:
            Polars DataFrame: The content of the CSV file.
        Raises:
            FedbiomedError: if the CSV file cannot be read due to inconsistent lines
        """
        return NipoppyDataRetriever(self._path).get_tabular_data(phenotypes=phenotypes, derivatives=derivatives)

    def shape(self):
        """Returns the shape of the csv dataset.

        Computed before applying transforms or conversion to other format.

        Returns:
            Dictionary with the shape and other necessary info for the dataset
        """
        if self._data is None:
            return {"nipoppy": (-1, -1)}
        return {"nipoppy": self._data.shape}

    def get(
        self,
        indexes: int | Iterable[int],
        phenotypes: Optional[Iterable | int | str] = None,
        derivatives: Optional[Iterable | int | str] = None,
    ) -> pd.DataFrame:
        """Gets the specified rows and columns in the dataset.

        Args:
            indexes: Row indexes to retrieve.
            columns: (Optional) list of columns to retrieve.
        Returns:
            Polars DataFrame: The specified dataframe.
        """
        # Convert indexes to an iterable if it is not already
        if not isinstance(indexes, Iterable) or isinstance(indexes, int):
            indexes = [indexes]

        if self._data is None:
            self._data = self._read(phenotypes=phenotypes, derivatives=derivatives)
            
        return self._data.iloc[indexes]

    def to_pandas(self) -> pd.DataFrame:
        """Returns the data as a Pandas Dataframe."""
        return self._data

    def to_numpy(self):
        """Returns the data as a Numpy ndarray."""
        return self._data.values

    def len(self) -> int:
        """Get number of samples"""
        return self.shape()["nipoppy"][0]

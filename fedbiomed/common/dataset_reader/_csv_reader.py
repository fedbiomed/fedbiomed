# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader implementation for CSV file
"""

import csv
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import polars as pl

from fedbiomed.common.exceptions import FedbiomedError, FedbiomedUserInputError


class CsvReader:
    _BEGINING_OF_FILE: int = 32768  # 32 kB data

    def __init__(
        self,
        path: Path,
        has_header: str | bool = "auto",
        delimiter: Optional[str] = None,
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

        self._delimiter = delimiter
        self.header: bool | None = None if has_header == "auto" else has_header

        # Pre-parse the CSV file to determine its delimiter and header
        # Note: this will read the first line of the file
        self._pre_parse()

        # Initialize the data and the column names
        self.data = self._read()
        self.columns = list(self.data.columns)

        # Initialize shape and length
        # Defer costly operations
        self._shape = self.data.shape
        self._len = self._shape[0]

    def _read(self, **kwargs) -> pl.DataFrame:
        """Reads all dataset and returns the dataframe.

        Returns:
            Polars DataFrame: The content of the CSV file.
        Raises:
            FedbiomedError: if the CSV file cannot be read due to inconsistent lines
        """

        try:
            df = pl.read_csv(
                self._path,
                separator=self._delimiter,
                has_header=bool(self.header),
                **kwargs,
            )
        except pl.exceptions.ComputeError as err:
            msg = f"cannot read csv file {self._path} due to inconsistent lines: see details"
            raise FedbiomedError(msg) from err

        return df

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def validate(self) -> None:
        """Validate the path of the CSV file.

        Raises:
            FedbiomedError: If the path is invalid
        """
        if not os.path.isfile(self._path):
            raise FedbiomedError(f"error: cannot find csv file {self._path}")

    def shape(self):
        """Returns the shape of the csv dataset.

        Computed before applying transforms or conversion to other format.

        Returns:
            Dictionary with the shape and other necessary info for the dataset
        """
        return {"csv": self._shape}

    def get(
        self,
        indexes: int | Iterable,
        columns: Optional[Iterable | int | str] = None,
    ) -> pl.DataFrame:
        """Gets the specified rows and columns in the dataset.

        Args:
            indexes: Row indexes to retrieve.
            columns: (Optional) list of columns to retrieve.
        Returns:
            Polars DataFrame: The specified dataframe.
        """
        return self._get_entry(indexes=indexes, columns=columns)

    def to_pandas(self) -> pd.DataFrame:
        """Returns the data as a Pandas Dataframe."""
        return self.data.to_pandas()

    def to_numpy(self):
        """Returns the data as a Numpy ndarray."""
        return self.data.to_numpy()

    def unsafe_to_torch(self):
        """This is an unsafe method that returns the data as a Torch Tensor.

        Warning: This method requires that columns have homogeneous data types. Havinng
        mixed types will raise an error.
        """
        self.data.to_torch()

    def _get_entry(
        self,
        indexes: int | Iterable,
        columns: Optional[Iterable] = None,
    ) -> pl.DataFrame:
        """Internal function to get the specified rows and columns in the dataset.

        Args:
            indexes: Row indexes to retrieve.
            columns: (Optional) list of columns to retrieve.
        Returns:
            Polars DataFrame: The specified dataframe.
        """

        # Convert indexes to an iterable if it is not already
        if not isinstance(indexes, Iterable) or isinstance(indexes, int):
            indexes = [indexes]

        _inter = list(filter(lambda x: 0 > x or x > self._len, set(indexes)))
        if any(_inter):
            raise FedbiomedUserInputError(
                f"Row index(es) {_inter} is out of range (0 to {self._len - 1})"
            )

        # Only retrieve specified columns if provided
        if columns is not None:
            columns = self.normalize_columns(columns)
            # if columns is a single string or integer, convert it to a list
            subset = self.data.select(columns)
        else:
            # case where columns=[], it will return the whole dataset: should we fix that?
            subset = self.data

        return subset[indexes]

    def normalize_columns(self, columns: Iterable | int | str) -> list[str]:
        """Validates `columns` and returns them in type `list`

        Args:
            columns: Columns to extract, can be an iterable of column names, a single
                column name, or an integer index.
        Returns:
            A list of column names.

        Raises:
            FedbiomedUserInputError: If the input does not match the types expected
        """

        if isinstance(columns, str) or isinstance(columns, int):
            columns = [columns]

        # if columns is a list of int, convert it to a list of column names
        # (auto-generated by polars as column_0, column_1, etc. if there is no header)
        if all(isinstance(item, int) for item in columns):
            n_cols = len(self.columns)
            _inter = list(filter(lambda x: 0 > x or x > n_cols, columns))
            if any(_inter):
                raise FedbiomedUserInputError(
                    f"Column index(es) {_inter} is out of range (0 to {n_cols - 1})"
                )
            columns = [self.columns[i] for i in columns]
        _faulty_col = list(filter(lambda x: x not in self.columns, columns))
        if any(_faulty_col):
            msg = f"Cannot read columns {_faulty_col}: file does not contain those columns specified"
            raise FedbiomedUserInputError(msg)

        return columns

    def len(self) -> int:
        """Get number of samples"""
        return self._len

    def _pre_parse(self):
        """Pre-parse the CSV file to determine its delimiter and header.

        This method reads the first few bytes of the CSV file to detect the delimiter
        and whether the file has a header or not. It sets the `_delimiter` and `header`
        attributes accordingly.

        This method should be called before any data is read from the CSV file.

        Raises:
            FedbiomedError: if the delimiter or header cannot be detected automatically,
            or if the path is incorrect.
        """
        self.validate()
        # Get sample of the file to detect delimiter and header
        with open(self._path, "r") as f:
            sample = f.read(
                CsvReader._BEGINING_OF_FILE
            )  # read the first 32 kBit of the file
        if not sample:
            raise FedbiomedError(f"File {self._path} is empty or cannot be read.")

        # Create a CSV sniffer to detect the delimiter and header
        sniffer = csv.Sniffer()

        # Try to detect the delimiter
        if self._delimiter is None:
            try:
                self._delimiter = sniffer.sniff(sample).delimiter
            except csv.Error as err:
                raise FedbiomedError(
                    "Cannot detect delimiter automatically. Please specify it through `delimiter` argument"
                ) from err

        # Try to detect the header
        if self.header is None:
            try:
                self.header = sniffer.has_header(sample)
            except csv.Error as err:
                raise FedbiomedError(
                    "Cannot detect header automatically. Please specify it through `has_header` argument"
                ) from err

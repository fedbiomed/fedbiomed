# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader implementation for CSV file
"""

import csv
import io
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import torch

from fedbiomed.common.dataset_types import (
    ReaderShape,
)
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedUserInputError

from ._reader import Reader


class CsvReader(Reader):
    _NB_LINES_PARSED_HEADER: int = (
        5  # parse 5 lines to see whether there is a header or not
    )

    def __init__(
        self,
        path: Path,
        memory: bool = True,
        has_header: str | bool = "auto",
        delimiter: Optional[str] = None,
    ) -> None:
        """Class constructor"""

        self._path = path
        self._memory = memory

        self._delimiter = delimiter if delimiter is None else delimiter
        self.header = None if has_header == "auto" else has_header

        # Pre-parse the CSV file to determine its delimiter and header
        # Note: this will read the first line of the file
        self._reader = None
        self._is_preparsed = False  # whether the reader has been pre-parsed
        self._pre_parse()

        # Initialize the data and the column names
        self.data = self.read()
        self.columns = self.data.collect_schema().names()

        # Initialize shape and length
        self._shape = self._reader.shape
        self._len = self._shape[0]

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def read(self, **kwargs) -> Union[pd.DataFrame, torch.Tensor, np.ndarray]:
        """
        Reads all dataset and returns its content.
        If `memory` is set to True, it will read the whole file in memory,
        otherwise it will read it lazily.
        Args:
            **kwargs: additional arguments to pass to the pandas read_csv function if memory=True
                      or additional arguments to pass to the polars scan_csv function if memory=False.
        Returns:
            DataFrame or Tensor or ndarray: content of the CSV file.
        Raises:
            FedbiomedUserInputError: if the CSV file cannot be read due to inconsistent lines
            or if the delimiter cannot be detected automatically.
        """

        if not self._is_preparsed:
            self._pre_parse()

        # If memory is True, read the whole file in memory
        if self._memory:
            try:
                file_content = pl.read_csv(
                    self._path,
                    separator=self._delimiter,
                    has_header=bool(self.header),
                    **kwargs,
                )
            except pl.exceptions.ComputeError as err:
                msg = f"cannot read csv file {self._path} due to inconsistent lines: see details"
                raise FedbiomedError(msg) from err
        else:
            # If memory is False, read the file lazily
            # Note: Polars does not provide an index_col parameter like Pandas
            try:
                file_content = pl.scan_csv(
                    self._path,
                    separator=self._delimiter,
                    has_header=bool(self.header),
                    **kwargs,
                )
            except pl.exceptions.ComputeError as err:
                msg = f"cannot read csv file {self._path} due to inconsistent lines: see details"
                raise FedbiomedError(msg) from err

        return file_content

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def validate(self) -> None:
        """Validate coherence of data modality served by a reader

        Raises exception if coherence issue found
        """
        self._validate_path()

    def _validate_path(self):
        if not os.path.isfile(self._path):
            raise FedbiomedUserInputError(f"error: cannot find file {self._path}")

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def shape(self) -> ReaderShape:
        """
        Returns shape of the data modality served by a reader
        Computed before applying transforms or conversion to other format
        Returns:
            ReaderShape: shape of the data modality served by a reader"""

        if not self._is_preparsed:
            self._pre_parse()

        return ReaderShape(dict({"csv": self._shape}))

    def get(
        self,
        indexes: int | str | Iterable = None,
        columns: Optional[Iterable] = None,
        index_col: Optional[int | str] = None,
    ) -> Union[pd.DataFrame, torch.Tensor, np.ndarray]:
        """
        Reads a single row or a batch of rows within a csv file. Rows are indexed
        by an index.

        Args:
            row_indexes: index or indexes.
            columns: (Optional) list of columns to retrieve.
            index_col: (Optional) column to use as index for rows.

        Returns:
            DataFrame or Tensor or ndarray: content of the CSV file.
        """

        if not self._is_preparsed:
            self._pre_parse()

        if indexes is None:
            indexes = 0

        return self._read_single_entry(
            indexes=indexes, columns=columns, index_col=index_col
        )

    def _read_single_entry_from_index_col(
        self,
        indexes: Any | Iterable,
        index_col: int | str,
        columns: Optional[Iterable] = None,
    ) -> Union[Tuple[Any], Tuple[Any] | None]:
        """
        Reads a single row or a batch of rows within a csv file.
        Rows are indexed by index_column and
        indexes are used as a filter to select which indexes to retrieve.
        Args:
            indexes: index or indexes.
            index_col: column to use as index for rows.
            columns: (Optional) list of columns to retrieve.
        Returns:
            Tuple of one or several entries in a framework specified format for each lines
            of the file requested. If no framework specified format has been defined, convert
            into np.array.
        """
        if isinstance(index_col, int):
            try:
                col_name = (
                    self.header[index_col]
                    if self.header
                    else "column_" + str(index_col)
                )
            except IndexError as err:
                raise FedbiomedUserInputError(
                    f"Got {index_col} as `index_col`, but does not exist in the header. {self.header}"
                ) from err
        else:
            col_name = index_col

        try:
            self._index = pl.col(col_name)
            csv_entry = self._reader.filter(self._index.is_in(indexes))  #  .collect()
        except pl.exceptions.ColumnNotFoundError as err:
            raise FedbiomedUserInputError(
                f"column not found {index_col}"
                f" `index_col` should be selected from {self.header}."
            ) from err

        if csv_entry.is_empty():
            raise FedbiomedUserInputError(
                "cannot find any entry matching " + " ".join(indexes)
            )

        if columns is not None:
            # if columns is a single string or integer, convert it to a list
            if isinstance(columns, str) or isinstance(columns, int):
                columns = [columns]

            # if columns is a list of int, convert it to a list of column names
            # (auto-generated by polars as column_0, column_1, etc. if there is no header)
            if isinstance(columns, list) and all(
                isinstance(item, int) for item in columns
            ):
                n_cols = len(csv_entry.columns)
                for i in columns:
                    if not (0 <= i < n_cols):
                        raise FedbiomedError(
                            f"Column index {i} is out of range (0 to {n_cols - 1})"
                        )
                all_cols = [csv_entry.columns[i] for i in columns]
                columns = all_cols

            if not all(column in csv_entry.columns for column in columns):
                msg = f"Cannot read columns {columns}: file does not contain some columns specified"
                raise FedbiomedUserInputError(msg)

            csv_entry = csv_entry.select(columns)

        return csv_entry

    def _read_single_entry(
        self,
        indexes: int | str | Iterable,
        index_col: Optional[int | str] = None,
        columns: Optional[Iterable] = None,
    ) -> Tuple[Any]:
        """Reads a single row or a batch of rows within a csv file. Rows are indexed
        by an index.

        Args:
            indexes: index or indexes.

        Returns:
            Tuple of one or several entries in a framework specified format for each lines
              of the file requested. If no framework specified format has been defined, convert
              into np.array.
        """

        if not isinstance(indexes, Iterable) or isinstance(indexes, str):
            indexes = [indexes]

        if not self._is_preparsed:
            self._pre_parse()

        if index_col is not None:
            return self._read_single_entry_from_index_col(indexes, index_col, columns)
        else:
            row_nb = pl.int_range(0, pl.len())

            csv_entry = self._reader.filter(row_nb.is_in(indexes))

            if csv_entry.is_empty():
                msg = f"Cannot read lines {indexes}: file only contains {self._len} samples"
                raise FedbiomedUserInputError(msg)

            if columns is not None:
                # if columns is a single string or integer, convert it to a list
                if isinstance(columns, str) or isinstance(columns, int):
                    columns = [columns]

                # if columns is a list of int, convert it to a list of column names
                # (auto-generated by polars as column_0, column_1, etc. if there is no header)
                if isinstance(columns, list) and all(
                    isinstance(item, int) for item in columns
                ):
                    n_cols = len(csv_entry.columns)
                    for i in columns:
                        if not (0 <= i < n_cols):
                            raise FedbiomedUserInputError(
                                f"Column index {i} is out of range (0 to {n_cols - 1})"
                            )
                    all_cols = [csv_entry.columns[i] for i in columns]
                    columns = all_cols

                if not all(column in csv_entry.columns for column in columns):
                    msg = f"Cannot read columns {columns}: file does not contain some columns specified"
                    raise FedbiomedUserInputError(msg)

                csv_entry = csv_entry.select(columns)

            if csv_entry.is_empty():
                msg = f"Cannot read columns {columns}: file does not contain some columns specified"
                raise FedbiomedUserInputError(msg)

            return csv_entry

    def _random_sample(self, n_entries: int, seed: Optional[int] = None):
        entries = self._reader.sample(n_entries, seed)
        return entries

    def len(self) -> int:
        """Get number of samples"""
        if self._len is None:
            self.shape()
        return self._len

    def _cpu_reader(self, **reader_kwargs):
        try:
            self._reader = pl.read_csv(
                self._path,
                separator=self._delimiter,
                has_header=bool(self.header),
                **reader_kwargs,
            )
        except pl.exceptions.ComputeError as err:
            msg = f"cannot read csv file {self._path} due to inconsistent lines: see details"
            raise FedbiomedUserInputError(msg) from err
        self._reader.shrink_to_fit()  # .lazy()

    def _gpu_reader(self, **reader_kwrags):
        # polar provides a GPU support but still in beta
        # availabile in JAX
        # eg `a = df.to_jax(device="gpu")  `
        # or using https://docs.pola.rs/user-guide/gpu-support/

        import cudf

        #  Read CSV into cuDF DataFrame on GPU
        gdf = cudf.read_csv("data.csv")

        #  Write Parquet to an in-memory buffer (BytesIO)
        buf = io.BytesIO()
        gdf.to_parquet(buf)

        #  Reset buffer position to start
        buf.seek(0)

        gdf2 = cudf.read_parquet(buf)
        # read line
        idx = 0
        line = gdf2.iloc[idx]
        return line

    def _pre_parse(self, **reader_kwargs):
        """
        Pre-parse the CSV file to determine its delimiter and header.
        This method reads the first few lines of the CSV file to detect the delimiter
        and whether the file has a header or not. It sets the `_delimiter` and `header`
        attributes accordingly. If the delimiter or header cannot be detected automatically,
        it raises a `FedbiomedError`.
        If the `_reader` is not initialized, it calls `_cpu_reader()` to read the file.
        If `memory` is set to False, it sets the reader to lazy mode.
        This method should be called before any data is read from the CSV file.
        Args:
            **reader_kwargs: additional arguments to pass to the pandas read_csv function.
        Raises:
            FedbiomedError: if the delimiter or header cannot be detected automatically, or if the path is incorrect.
        """

        # Return if already pre-parsed
        if self._is_preparsed:
            return

        # Get sample of the file to detect delimiter and header
        with open(self._path, "r") as f:
            sample = f.read(8192)
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
        if not self.header:
            try:
                self.header = sniffer.has_header(sample)
            except csv.Error as err:
                raise FedbiomedError(
                    "Cannot detect header automatically. Please specify it through `has_header` argument"
                ) from err

        # Initialize the reader if it is not already initialized
        if self._reader is None:
            self._cpu_reader()

        # Set the reader to lazy mode if memory is False
        if self._memory is False:
            self._reader.lazy()

        self._is_preparsed = True

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
    DataReturnFormat,
    ReaderShape,
    Transform,
    drf_default,
)
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedUserInputError

from ._reader import Reader


class _CsvReader(Reader):
    def __init__(
        self,
        root: Path,
        memory: bool = True,
        # to_format: support DEFAULT TORCH SKLEARN
        to_format: DataReturnFormat = drf_default,
        reader_transform: Transform = None,
        reader_target_transform: Transform = None,
        has_header: bool = True,
        delimiter: Optional[str] = ",",
        # Optional parameters
        usecols=None,
        index_col=None,
        nrows=None,
        # Optional Polars parameters
        encoding: str = "utf-8",
        force_read: Optional[bool] = False,
        n_thread: Optional[int] = None,
        low_memory: bool = False,  # True read iteratively, False read all at once
        # Any other parameter ?
    ) -> None:
        # Mandatory parameters
        self._path = root
        self._memory = memory

        # Optional parameters
        self._usecols = usecols
        self._index_col = index_col
        self._nrows = nrows

        self._has_header = has_header
        self._header = 0 if has_header else None

        # Initialize lazy reader
        self._delimiter = delimiter
        self._cpu_reader()

        # # Detect delimiter
        # self._delimiter = self._detect_delimiter() if delimiter == "," else delimiter

        # # Update delimiter if it is not a comma
        # if self._delimiter != ",":
        #     self._cpu_reader()

        # Transform is going to be the data itself if there are no transform provided
        if reader_transform is None:
            self._reader_transform = lambda x: x
        else:
            self._reader_transform = reader_transform

        if reader_target_transform is None:
            self._reader_target_transform = lambda x: None
        else:
            self._reader_target_transform = reader_target_transform

        if not self._memory:
            # Use Polars for lazy reading
            # Note: Polars does not provide an index_col parameter like Pandas
            self._batched = pl.read_csv_batched(
                source=root,
                has_header=has_header,
                columns=self._usecols,
                separator=self._delimiter,
                n_rows=self._nrows,
            )  # lazy batches
            self._rows = iter(self._batched.next_batches(1))

    def get_single_item(
        self, row_index=None, columns=None, index_col=None
    ) -> Union[pd.DataFrame, torch.Tensor, np.ndarray]:
        """
        Function to get specific columns or rows from the CSV file.
        Args:
            columns: list of columns to retrieve.
            row_index: index of the row to retrieve.
            index_col: column to use as index for rows.
        Returns:
            DataFrame or Tensor or ndarray: content of the CSV file.
        """

        # Use row 0 if no row_index is specified
        if row_index is None:
            row_index = 0

        # Use all columns if no columns are specified
        selected_columns = pl.all() if columns is None else columns

        try:
            csv_entry = self._reader.select(selected_columns).row(row_index, named=True)
        except pl.exceptions.OutOfBoundsError as err:
            msg = (
                f"Cannot read row {row_index}: file only contains {self._len} samples"
                + f"See details in the error message: {str(err)}"
            )
            raise FedbiomedUserInputError(msg) from err

        return csv_entry

    def get_multiple_items(
        self, row_indexes=None, columns=None, index_col=None
    ) -> Union[pd.DataFrame, torch.Tensor, np.ndarray]:
        """
        Function to get specific columns or rows from the CSV file.
        Args:
            columns: list of columns to retrieve.
            row_index: index of the row to retrieve.
            index_col: column to use as index for rows.
        Returns:
            DataFrame or Tensor or ndarray: content of the CSV file.
        """

        # Use row 0 if no row_index is specified
        if row_indexes is None:
            row_indexes = [0]

        # Use all columns if no columns are specified
        selected_columns = pl.all() if columns is None else columns

        try:
            csv_entries = self._reader[row_indexes].select(selected_columns)
        except pl.exceptions.OutOfBoundsError as err:
            msg = (
                f"Cannot read rows {row_indexes}: file only contains {self._len} samples"
                + f"See details in the error message: {str(err)}"
            )
            raise FedbiomedUserInputError(msg) from err

        return csv_entries

    def _cpu_reader(self, **reader_kwargs):
        try:
            self._reader = pl.read_csv(
                self._path,
                separator=self._delimiter,
                has_header=self._has_header,
                # encoding= self._encoding,  # this causes perf issues for some reasons ...
                # n_threads=self._n_thread,
                # low_memory=self._low_memory,
                # truncate_ragged_lines=self._force_read,
                **reader_kwargs,
            )
        except pl.exceptions.ComputeError as err:
            msg = f"cannot read csv file {self._path} due to inconsistent lines: see details"
            raise FedbiomedUserInputError(msg) from err
        self._reader.shrink_to_fit()  # .lazy()

    def read(self, **kwargs) -> Union[pd.DataFrame, torch.Tensor, np.ndarray]:
        """
        Function to read a CSV file and return its content.
        It will read the whole file in memory if `memory` is set to True, otherwise it will read it iteratively.
        Args:
            usecols: list of columns to read from the CSV file.
            index_col: column to use as index for rows.
            nrows: number of rows to read from the CSV file.
            **kwargs: additional arguments to pass to the pandas read_csv function if memory=True
            or additional arguments to pass to the polars scan_csv function if memory=False.
        Returns:
            DataFrame or Tensor or ndarray: content of the CSV file.
        Raises:
            FedbiomedUserInputError: if the CSV file cannot be read due to inconsistent lines
            or if the delimiter cannot be detected automatically.
        """

        if self._memory:
            try:
                file_content = self.pd.read_csv(
                    self._path,
                    sep=self._delimiter,
                    header=self._header,
                    usecols=self._usecols,
                    index_col=self._index_col,
                    nrows=self._nrows,
                    **kwargs,
                )
            except pd.errors.ParserError as err:
                raise FedbiomedUserInputError(
                    f"Cannot read csv file {self._path} due to inconsistent lines:\n"
                    f"See details in the error message: {str(err)}"
                ) from err

        else:
            try:
                file_content = next(self._rows)
            except StopIteration as err:
                batches = self._batched.next_batches(1)  # [] at EOF
                if not batches:
                    raise FedbiomedUserInputError(
                        f"Cannot read csv file {self._path} no more batches available.\n"
                        f"See details in the error message: {str(err)}"
                    ) from err
                # iterate rows from the next DataFrame batch
                self._rows = iter(batches)
                file_content = next(self._rows)

        return self._reader_transform(file_content)

    def _detect_delimiter(self) -> str:
        """
        Detects the delimiter of a CSV file by reading the first line.

        Raises:
            FedbiomedUserInputError: if the delimiter cannot be detected automatically.
        """

        sniffer = csv.Sniffer()
        _first_line = self._reader.row(0)

        try:
            self._delimiter = sniffer.sniff(_first_line[0]).delimiter
        except Exception as err:
            raise FedbiomedUserInputError(
                "Cannot detect delimiter automatically. Please specify it through `delimiter` argument.\n"
                "Error message is: " + str(err)
            ) from err
        return self._delimiter

    def validate(self) -> None:
        """Validate coherence of data modality served by a reader

        Raises exception if coherence issue found
        """
        if not os.path.isfile(self._path):
            raise FedbiomedUserInputError(f"error: cannot find file {self._path}")

    def shape(self) -> ReaderShape:
        """Returns shape of the data modality served by a reader

        Computed before applying transforms or conversion to other format"""
        if self._memory:
            if self._usecols is not None:
                columns = self._usecols
            else:
                columns = None

            # Read the CSV file to determine its shape
            # Note: this will read the entire file into memory
            df = pd.read_csv(
                self._path,
                sep=self._delimiter,
                header=self._header,
                usecols=columns,
            )
            return ReaderShape(dict({"csv": df.shape}))
        else:
            # For lazy reading, we cannot determine the shape without reading the file
            # Lazy read of the CSV file (streaming, low memory usage)
            lf = pl.scan_csv(
                self._path,
                separator=self._delimiter,
                has_header=self._has_header,
            )
            # Get number of rows
            n_rows = lf.select(pl.count()).collect().to_pandas().iloc[0, 0]

            # Get number of columns
            n_cols = len(lf.columns)

            return ReaderShape(dict({"csv": (n_rows, n_cols)}))


class CsvReader(Reader):
    _NB_LINES_PARSED_HEADER: int = (
        5  # parse 5 lines to see whether there is a header or not
    )

    def __init__(
        self,
        root: Path,
        memory: bool = True,
        # to_format: support DEFAULT TORCH SKLEARN
        to_format: DataReturnFormat = drf_default,
        reader_transform: Transform = None,
        reader_target_transform: Transform = None,
        has_header: str | bool = "auto",
        delimiter: Optional[str] = None,
        # Optional polars parameters
        encoding: str = "utf-8",
        force_read: Optional[bool] = False,
        n_thread: Optional[int] = None,
        low_memory: bool = False,
        # Any other parameter ?
    ) -> None:
        """Class constructor"""

        self._path = root
        self._memory = memory

        self._delimiter = delimiter if delimiter is None else delimiter
        self.header = None if has_header == "auto" else has_header

        # Potential parameters to use in the future for lazy reading
        self._offsets = []
        self._header_offset = 0
        self._encoding = encoding  # triggers perf issues when reading file
        self._force_read = force_read
        self._low_memory = low_memory
        self._n_thread = n_thread
        self._index = None

        # Pre-parse the CSV file to determine its delimiter and header
        # Note: this will read the first line of the file
        self._reader = None
        self._is_preparsed = False  # whether the reader has been pre-parsed
        self._pre_parse()

        # Transform is going to be the data itself if there are no transform provided
        if reader_transform is None:
            self._reader_transform = lambda x: x
        else:
            self._reader_transform = reader_transform

        if reader_target_transform is None:
            self._reader_target_transform = lambda x: None
        else:
            self._reader_target_transform = reader_target_transform

        # default behaviour
        self.to_generic()

        # Initialize the data and the column names
        self.data = self.read()
        self.columns = self.data.columns

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

        return self._reader_transform(file_content)

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

        # res = res.drop(col_name)  # is it useful?

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
                        raise IndexError(
                            f"Column index {i} is out of range (0 to {n_cols - 1})"
                        )
                all_cols = [csv_entry.columns[i] for i in columns]
                columns = all_cols

            csv_entry = csv_entry.select(columns)

        data = self._transform_framework(self._reader_transform(csv_entry))
        target = self._transform_framework(self._reader_target_transform(csv_entry))

        return data, target

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
                            raise IndexError(
                                f"Column index {i} is out of range (0 to {n_cols - 1})"
                            )
                    all_cols = [csv_entry.columns[i] for i in columns]
                    columns = all_cols

                csv_entry = csv_entry.select(columns)

            if csv_entry.is_empty():
                msg = f"Cannot read columns {columns}: file does not contain some columns specified"
                raise FedbiomedUserInputError(msg)

            data, target = (
                self._reader_transform(csv_entry),
                self._reader_target_transform(csv_entry),
            )

            return self._transform_framework(data), self._transform_framework(target)

    def set_index(self, index_col: int | str):
        if not self._is_preparsed:
            self._pre_parse()
        if isinstance(index_col, int):
            col_name = (
                self.header[index_col] if self.header else "column_" + str(index_col)
            )
        else:
            col_name = index_col
        self._index = pl.col(col_name)

    def _random_sample(self, n_entries: int, seed: Optional[int] = None):
        entries = self._reader.sample(n_entries, seed)
        return entries

    def index(self):
        if not self._is_preparsed:
            self._pre_parse()
        if self._index is None:
            self.shape()
            self._index = pl.int_range(0, self._len)
            return pl.select(self._index.alias("index"))
        else:
            return self._reader.select(self._index.alias("index"))

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
                # encoding= self._encoding,  # this causes perf issues for some reasons ...
                n_threads=self._n_thread,
                low_memory=self._low_memory,
                truncate_ragged_lines=self._force_read,
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

    # Additional methods for exploring data, depending on Reader
    def _pre_parse(self, **reader_kwargs):
        # get delimiter and header
        # only retrieve info from the first line of csv

        if self._delimiter is None:
            with open(self._path, "r") as f:
                sample = f.read(8192)
            if not sample:
                raise FedbiomedUserInputError(
                    f"File {self._path} is empty or cannot be read."
                )

            sniffer = csv.Sniffer()
            try:
                self._delimiter = sniffer.sniff(sample).delimiter
            except csv.Error as err:
                raise FedbiomedError(
                    "Cannot detect delimiter automatically. Please specify it through `delimiter` argument"
                ) from err

        if not self.header:
            try:
                self.header = csv.Sniffer().has_header(sample)
            except csv.Error as err:
                raise FedbiomedError(
                    "Cannot detect header automatically. Please specify it through `has_header` argument"
                ) from err

        print(f"Detected delimiter: {self._delimiter}")
        print(f"Detected header: {self.header}")

        if self._reader is None:
            self._cpu_reader()

        # self._header_offset = len(_first_line) + 1 if self.header is not None else 0

        if self._memory is False:
            self._reader.lazy()

        self._is_preparsed = True

    def _post_parse(self, x):
        x.collect()

    def _get_first_line(self):
        _header = [x for x in self._reader.schema.items()]
        _first_line_types = [x[1] for x in _header]
        first_line = tuple(
            t.to_python()(x[0])
            for x, t in zip(self._reader.schema, _first_line_types, strict=True)
        )
        return first_line

    def _detect_header(self, sniffer: csv.Sniffer):
        _first_lines = []

        if self.header:
            _first_lines.append(self._get_first_line())

        i = 0
        _parse_next_row = True

        while i < CsvReader._NB_LINES_PARSED_HEADER and _parse_next_row:
            try:
                _first_lines.append(self._reader.row(i))
                i += 1
            except pl.exceptions.OutOfBoundsError:  # OutofBondsError
                # raised if file contains less value then CsvReader._NB_LINES_PARSED_HEADER
                _parse_next_row = False

        _first_lines = self._revert_to_text(_first_lines)

        if len(_first_lines) == 1 and len(_first_lines[0]) == 1:
            _first_lines = _first_lines[0]
        else:
            _first_lines = "".join(_first_lines)

        try:
            has_header = sniffer.has_header(_first_lines)
        except csv.Error as err:
            raise FedbiomedUserInputError(
                "Cannot detect header automatically. Please specify it through `has_header` argument"
            ) from err
        return has_header

    @staticmethod
    def _revert_to_text(t: Iterable) -> str:
        storage = []
        for item in t:
            storage.append(",".join((str(c) for c in item)))
        return "\n".join(storage)

    def set_tranform_framework(self, framework: DataReturnFormat):
        self.to_generic()

    def to_generic(self):
        def to_numpy(x):
            val = (
                np.array(x)
                if isinstance(x, (np.ndarray, tuple, type(None)))
                else x.to_numpy()
            )
            return val

        self._transform_framework = to_numpy
        # self._transform_framework = lambda x: np.array(x()) if isinstance(x(), (np.ndarray, tuple)) else x().to_numpy()
        self._transform_batch_framework = lambda x: np.stack(x)

    def to_pytorch(self):
        # TODO; implement lazy transform?
        self._transform_batch_framework = lambda x: torch.stack(x, dim=1).squeeze()
        pass

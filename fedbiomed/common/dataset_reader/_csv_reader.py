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
from fedbiomed.common.exceptions import FedbiomedUserInputError

from ._reader import Reader


class CsvReader(Reader):
    _NB_LINES_PARSED_HEADER: int = (
        5  # parse 5 lines to see whether there is a header or not
    )

    def __init__(
        self,
        root: Path,
        # to_format: support DEFAULT TORCH SKLEARN
        to_format: DataReturnFormat = drf_default,
        reader_transform: Transform = None,
        reader_target_transform: Transform = None,
        reader_transform: Transform = None,
        native_target_transform: Transform = None,
        encoding: str = "utf-8",
        has_header: str | bool = "auto",
        delimiter: Optional[str] = ",",
        force_read: Optional[bool] = False,
        n_thread: Optional[int] = None,
        low_memory: bool = False,
        # Any other parameter ?
    ) -> None:
        """Class constructor"""
        self._path = root
        self._offsets = []
        self._is_preparsed = False
        self._delimiter = "," if delimiter is None else delimiter
        self.header = None if has_header == "auto" else has_header
        self._header_offset = 0
        self._encoding = encoding  # triggers perf issues when reading file

        self._shape: Tuple[int] = None
        self._len = None
        self._force_read = force_read
        self._low_memory = low_memory

        # self._transform_framework = lambda x: x  # change that dpd on to_format arg
        self._format = to_format

        # self._native_transform = lambda x: x if native_transform is None else native_transform

        if native_transform is None:
            self._native_transform = lambda x: x
        else:
            self._native_transform = native_transform

        if native_target_transform is None:
            self._native_target_transform = lambda x: None
        else:
            self._native_target_transform = native_target_transform

        self._reader = None
        self._n_thread = n_thread
        self._index_col_name: int = 0

        # default behaviour
        self.to_generic()

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def read(
        self, index_col: Optional[int] = None, **kwargs
    ) -> Union[pd.DataFrame, torch.Tensor, np.ndarray]:
        """Retrieve data: read all dataset"""
        if not self._is_preparsed:
            self._pre_parse()
        file_content = pl.scan_csv(
            self._path,
            index_col,
            separator=self._delimiter,
            has_header=bool(self.header),
            n_threads=self._n_thread,
            **kwargs,
        )
        # TODO: add header + delimiter into the read_csv

        return self._native_transform(file_content)

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
        """Returns shape of the data modality served by a reader

        Computed before applying transforms or conversion to other format"""

        # Optional methods which can be implemented (or not) by some readers
        # Code is specific to each reader
        if self._shape is None:
            # if self._reader is None:
            #     self._reader = pl.read_csv(self._path, has_header=bool(self.header))
            # # with self._closer(open(self._path, "rb"), get_offset=False) as file:
            # #     file.seek(0)
            if not self._is_preparsed:
                self._pre_parse()

            self._len = self._reader.shape[0]
            self._shape = self._reader.shape

        return ReaderShape(dict({"csv": self._shape}))

    def read_single_entry_from_column(
        self, index_col: int | str, indexes: Any | Iterable
    ):
        self._index_col_name = index_col

        if isinstance(index_col, int):
            col_name = (
                self.header[index_col] if self.header else "column_" + str(index_col)
            )

        if not isinstance(indexes, Iterable):
            indexes = [indexes]

        if not self._is_preparsed:
            self._pre_parse()

        res = self._reader.filter(pl.col(col_name).is_in(indexes))  #  .collect()
        if res.is_empty():
            raise FedbiomedUserInputError(
                "cannot find any entry matching " + " ".join(indexes)
            )

        res = res.drop(col_name)
        # res = tuple(
        #     self._transform_framework(self._native_transform(res).row(e))
        #     for e in range(len(res))
        # )
        res = self._transform_framework(self._native_transform(res))
        target = self._transform_framework(self._native_target_transform(res))
        # return self._transform_batch_framework(res)
        return res, target

    def read_single_entry(self, indexes: int | Iterable) -> Tuple[Any]:
        """Reads a single row or a btach of rows within a csv file. Rows are indexed
        by an index.

        Args:
            indexes: index or indexes.

        Returns:
            Tuple of one or several entries in a framework specified format for each lines
              of the file requested. If no framework specified format has been defined, convert
              into np.array.
        """
        if not isinstance(indexes, Iterable):
            indexes = [indexes]

        if not self._is_preparsed:
            self._pre_parse()
        entries = []

        row_nb = pl.int_range(0, pl.len())

        csv_entry = self._reader.filter(row_nb.is_in(indexes))
        if csv_entry.is_empty():
            msg = f"Cannot read lines {indexes}: file only contains {self._len} samples"
            raise FedbiomedUserInputError(msg)
        # entries.append(self._transform_framework(csv_entry))
        # TODO: implement logic here

        data, target = (
            self._native_transform(csv_entry),
            self._native_target_transform(csv_entry),
        )
        return self._transform_framework(data), self._transform_framework(target)
        return self._transform_batch_framework(tuple(entries))
        return entries

    def _random_sample(self, n_entries: int, seed: Optional[int] = None):
        entries = self._reader.sample(n_entries, seed)
        return entries

    def get_index(self):
        pass

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
                # encoding= self._encoding,
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
        # only rertieve info from the first line of csv
        if self._reader is None:
            self._cpu_reader()

        sniffer = csv.Sniffer()
        _first_line = self._reader.row(0)

        if len(_first_line) == 1:
            # reload file with appropriate delimiter

            self._delimiter = sniffer.sniff(_first_line[0]).delimiter
            if self._delimiter != ",":
                # self._reader = pl.read_csv(
                #     self._path,
                #     separator=self._delimiter,
                #     has_header=bool(self.header),
                #     truncate_ragged_lines=self._force_read,
                #     **reader_kwargs,
                # )
                self._cpu_reader()
                _first_line = _first_line[0].split(self._delimiter)

        if not self.header and self._detect_header(sniffer):
            self.header = True  # temporary
            self._cpu_reader()  # needed to re-asses values of each columns
            self.header = self._reader.columns
            # self._reader = self._reader[1:]

            # assuming header contains only strings
        else:
            self.header = None
        # self._header_offset = len(_first_line) + 1 if self.header is not None else 0
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
                # raised if file comtains less value then CsvReader._NB_LINES_PARSED_HEADER
                _parse_next_row = False

        _first_lines = self._revert_to_text(_first_lines)

        if len(_first_lines) == 1 and len(_first_lines[0]) == 1:
            return sniffer.has_header(_first_lines[0])

        else:
            return sniffer.has_header("".join(_first_lines))

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

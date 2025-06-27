# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader implementation for CSV file
"""

import codecs
import csv
import io
import mmap
import os
import threading
from contextlib import contextmanager
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    ReaderShape,
    Transform,
    drf_default,
)
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedUserInputError
from fedbiomed.common.logger import logger

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
        delimiter: Optional[str] = None,
        # Any other parameter ?
    ) -> None:
        """Class constructor"""
        self._path = root
        self._offsets = []
        self._is_preparsed = False
        self._delimiter = ""
        self.header = None
        self._header_offset = 0
        self._encoding = encoding
        self._quoting = csv.QUOTE_NONE
        self._shape: Tuple[int] = None
        self._len = None
        self.stop_event = threading.Event()

        # self._transform_framework = lambda x: x  # change that dpd on to_format arg
        self._format = to_format

        self._native_transform = native_transform
        self._native_target_transform = native_target_transform

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def read(
        self, index_col: Optional[int] = None, index_max: Optional[int] = None, **kwargs
    ) -> Union[pd.DataFrame, torch.Tensor, np.ndarray]:
        """Retrieve data: read all dataset"""
        import dask.dataframe as dd

        file_content = dd.read_csv(self._path, index_col, **kwargs)
        # TODO: add header + delimiter into the read_csv
        # FIXME: use only dask if needed. or use context manager
        return file_content

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def validate(self) -> None:
        """Validate coherence of data modality served by a reader

        Raises exception if coherence issue found
        """

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
            with self._closer(open(self._path, "rb"), get_offset=False) as file:
                file.seek(0)
                if not self._is_preparsed:
                    self._pre_parse(file)

                if self.header is None:
                    file.seek(0)

                first_line = csv.reader(
                    codecs.iterdecode(file, self._encoding), delimiter=self._delimiter
                )  # parse first line for control
                nb_col = len(
                    next(first_line)
                )  # get column number from first line of the csv
                nb_row = max(sum(1 for row in file), 1)
                # nb_row = nb_row - 1 if self.header else nb_row
            self._shape = (
                nb_row,
                nb_col,
            )

        return ReaderShape(dict({"csv": self._shape}))

    def read_single_entry(self, indexes: int | Iterable) -> Tuple[np.ndarray]:
        """Reads a single row or a btach of rows within a csv file. Rows are indexed
        by an index.

        Args:
            indexes: index or indexes.

        Returns:
            Tuple of one or several np.array for each lines of the file requested
        """
        if not isinstance(indexes, Iterable):
            indexes = [indexes]
        entries = []

        # with (open(self._path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm):
        #     if not self._is_preparsed:
        #         self._pre_parse(mm)
        #     if not self._offsets:
        #         self._get_offset(mm)
        with (
            open(self._path, "rb") as f,
            self._closer(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as mm,
        ):
            for idx in indexes:
                try:
                    offset = self._offsets[idx]
                except IndexError as err:
                    msg = f"Cannot read line {idx}: file only contains {self._len} samples"
                    raise FedbiomedUserInputError(msg) from err
                mm.seek(offset)

                # Read until end of line
                row_bytes = mm.readline()

                # Decode bytes to string
                row = row_bytes.decode(self._encoding).strip()
                if not row:
                    msg = f"cannot read csv file {self._path} due to inconsistent lines: see line {idx} "
                    logger.error(msg)
                    raise FedbiomedError(msg)

                # csv_entry = csv.reader(io.StringIO(row), delimiter=self._delimiter, quotechar='|', quoting=csv.QUOTE_NONE)
                # entries.append(next(csv_entry))

                csv_entry = pd.read_csv(
                    io.StringIO(row), delimiter=self._delimiter, header=None
                )

                entries.append(csv_entry.values[0])

        return tuple(entries)

    def multithread_read_single_entry(
        self, indexes: int | Iterable
    ) -> Tuple[np.ndarray]:
        """Reads a single row or a btach of rows within a csv file. Rows are indexed
        by an index.

        Args:
            indexes: index or indexes.

        Returns:
            Tuple of one or several np.array for each lines of the file requested
        """
        if not isinstance(indexes, Iterable):
            indexes = [indexes]

        entries = [None] * len(indexes)
        index_map = {idx: i for i, idx in enumerate(indexes)}

        def rthread(idx: int):
            with mm_lock:
                try:
                    offset = self._offsets[idx]
                except IndexError as err:
                    msg = f"Cannot read line {idx}: file only contains {self._len} samples"
                    raise FedbiomedUserInputError(msg) from err
                mm.seek(offset)

                # Read until end of line
                row_bytes = mm.readline()

            # Decode bytes to string
            row = row_bytes.decode(self._encoding).strip()
            if not row:
                msg = f"cannot read csv file {self._path} due to inconsistent lines: see line {idx} "
                logger.error(msg)
                raise FedbiomedError(msg)

            # csv_entry = csv.reader(io.StringIO(row), delimiter=self._delimiter, quotechar='|', quoting=csv.QUOTE_NONE)
            # entries.append(next(csv_entry))

            csv_entry = pd.read_csv(
                io.StringIO(row), delimiter=self._delimiter, header=None
            )
            return index_map[idx], csv_entry.values[0]

        # with (open(self._path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm):
        #     if not self._is_preparsed:
        #         self._pre_parse(mm)
        #     if not self._offsets:
        #         self._get_offset(mm)

        try:
            with (
                open(self._path, "rb") as f,
                self._closer(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as mm,
            ):
                mm_lock = threading.Lock()
                thread_func = partial(rthread)
                # import pdb; pdb.set_trace()
                with ThreadPool(3) as pool:
                    for i, entry in pool.map(thread_func, indexes):
                        entries[i] = entry
                        if self.stop_event.is_set():
                            break
        except KeyboardInterrupt:
            self.stop_event.set()
            return
        return tuple(entries)

    def _read_single_entry(self, idx, mm):
        try:
            offset = self._offsets[idx]
        except IndexError as err:
            msg = f"Cannot read line {idx}: file only contains {self._len} samples"
            raise FedbiomedUserInputError(msg) from err
        mm.seek(offset)

        # Read until end of line
        row_bytes = mm.readline()

        # Decode bytes to string
        row = row_bytes.decode(self._encoding).strip()
        if not row:
            msg = f"cannot read csv file {self._path} due to inconsistent lines: see line {idx} "
            logger.error(msg)
            raise FedbiomedError(msg)

        # csv_entry = csv.reader(io.StringIO(row), delimiter=self._delimiter, quotechar='|', quoting=csv.QUOTE_NONE)
        # entries.append(next(csv_entry))

        csv_entry = pd.read_csv(
            io.StringIO(row), delimiter=self._delimiter, header=None
        )
        return csv_entry

    def get_index(self):
        if not self._offsets:
            self._get_index()

        return list(range(self._len))

    def _get_index(self):
        with (
            open(self._path, "rb") as f,
            self._closer(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)),
        ):
            pass

    def len(self) -> int:
        """Get number of samples"""
        if self._len is None:
            self.get_index()
        return self._len

    # Additional methods for exploring data, depending on Reader
    def _pre_parse(self, f):
        # get delimiter and header
        # only rertieve info from the first line of csv
        sniffer = csv.Sniffer()
        _first_line = f.readline()
        _first_line = _first_line.decode(self._encoding).strip()
        self._delimiter = sniffer.sniff(_first_line).delimiter
        f.seek(0)

        # _first_lines = [row.decode(self._encoding).strip() for row in f]
        # f.seek(0)
        if self._detect_header(f, sniffer):
            self.header = next(
                csv.reader(io.StringIO(_first_line), delimiter=self._delimiter)
            )
            # assuming header contains only strings
        else:
            self.header = None
        self._header_offset = len(_first_line) + 1 if self.header is not None else 0
        self._is_preparsed = True

    @contextmanager
    def _closer(self, mm, get_offset=True):
        if not self._is_preparsed:
            try:
                self._pre_parse(mm)
            except csv.Error as err:
                # TODO: add options for that in constructor + cli
                raise FedbiomedError(
                    "Cannot detect csv headers and delimiters automatically. It is possible the file is corrupted"
                ) from err
        if not self._offsets and get_offset:
            self._get_offset(mm)
        try:
            yield mm
        finally:
            mm.close()

    def _get_offset(self, mm):
        # get file offset
        # with open(file_path, 'r') as f:
        # mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offset = 0

        line = True
        while line:
            self._offsets.append(offset)
            line = mm.readline()

            offset += len(line)

        if self.header is not None:
            self._offsets.pop(0)
        self._len = len(self._offsets)

    def _detect_header(self, f, sniffer):
        _first_lines = []
        i = 0
        _parse_next_row = True
        iterable = iter(f)
        while i < CsvReader._NB_LINES_PARSED_HEADER and _parse_next_row:
            try:
                row = next(iterable)

                _first_lines.append(row.decode(self._encoding))
                i += 1
            except StopIteration:
                # raised if file comtains less value then CsvReader._NB_LINES_PARSED_HEADER
                _parse_next_row = False
        f.seek(0)

        if len(_first_lines) == 1 and len(_first_lines[0]) == 1:
            val = sniffer.has_header(_first_lines[0])

            return val
        else:
            return sniffer.has_header("".join(_first_lines))

    # #@staticmethod
    # def default_auto_cast(self, value) -> str | int | float:
    #     if not value:
    #         return value
    #     if value.isdigit() or (value[0] == '-' and value[1:].isdigit()):
    #         return int(value)

    # def numerical_cast(self, value):
    #     return value

    def set_tranform_framework(self, framework: DataReturnFormat):
        self.to_generic()

    def to_generic(self):
        self._transform_framework = lambda x: np.array(x)

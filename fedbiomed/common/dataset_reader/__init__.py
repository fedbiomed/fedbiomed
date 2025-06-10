# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Classes that simplify imports from fedbiomed.common.dataset_reader
"""

from ._reader import Reader
from ._csv_reader import CsvReader
from ._nifti_reader import NiftiReader
from ._mnist_reader import MnistReader


__all__ = [
    "Reader",
    "CsvReader",
    "NiftiReader",
    "MnistReader",
]

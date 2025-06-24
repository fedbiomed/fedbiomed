# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader implementation for CSV file
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    ReaderShape,
    Transform,
    drf_default,
)

from ._reader import Reader


class CsvReader(Reader):
    def __init__(
        self,
        root: Path,
        # to_format: support DEFAULT TORCH SKLEARN
        to_format: DataReturnFormat = drf_default,
        reader_transform: Transform = None,
        reader_target_transform: Transform = None,
        # Any other parameter ?
    ) -> None:
        """Class constructor"""

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def read(
        self, index: Optional[int] = None, index_max: Optional[int] = None
    ) -> Union[pd.DataFrame, torch.Tensor, np.ndarray]:
        """Retrieve data"""

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def validate(self) -> None:
        """Validate coherence of data modality served by a reader

        Raises exception if coherence issue found
        """

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def shape(self) -> ReaderShape:
        """Returns shape of the data modality served by a reader

        Computed before applying transforms or conversion to other format"""

    # Optional methods which can be implemented (or not) by some readers
    # Code is specific to each reader

    def len(self) -> int:
        """Get number of samples"""

    # Additional methods for exploring data, depending on Reader

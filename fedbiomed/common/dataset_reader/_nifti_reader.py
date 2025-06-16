# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader implementation for NIFTI image set
"""

from pathlib import Path

import torch

from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    ReaderShape,
    Transform,
    drf_default,
)

from ._reader import Reader


class NiftiReader(Reader):
    def __init__(
        self,
        root: Path,
        # to_format: support DEFAULT and TORCH (SKLEARN not supported)
        # both are torch.Tensor
        to_format: DataReturnFormat = drf_default,
        native_transform: Transform = None,
        native_target_transform: Transform = None,
        # Any other parameter ?
        modality: str = "T1",
    ) -> None:
        """Class constructor"""

    # Nota: does not include filtering of DLP, which is unknown to Reader
    # Nota: may want to support PIL as DEFAULT instead of torch.TENSOR
    def read(self) -> torch.Tensor:
        """Retrieve data"""

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def validate(self) -> None:
        """Validate coherence of data modality served by a reader

        Raises exception if coherence issue found
        """

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def shape(self) -> ReaderShape:
        """Returns shape of the data served by a reader

        Computed before applying transforms or conversion to other format"""

    # Optional methods which can be implemented (or not) by some readers
    # Code is specific to each reader

    # Additional methods for exploring data, depending on Reader

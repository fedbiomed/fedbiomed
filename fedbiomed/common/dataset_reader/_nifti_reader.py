# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader implementation for NIFTI image set
"""

from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
import torch

from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    drf_default,
)


class NiftiReader:
    return_format: DataReturnFormat = drf_default

    @classmethod
    def read(
        self, path: Union[str, Path]
    ) -> Union[torch.Tensor, np.ndarray, nib.Nifti1Image]:
        """Reads the NIfTI file and returns it as a tensor, optionally transformed.

        Args:
            path (Union[str, Path]): Path to the NIfTI file (.nii or .nii.gz)

        Returns:
            Union[torch.Tensor, np.ndarray, nib.Nifti1Image]: The image data in the specified format.
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid NIfTI file.
        """
        # IMPORTANT: Read function does not retunr header. It may bne useful in the future.
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError(f"Expected path to be a string or Path, got {type(path)}")

        path = Path(path)

        self.validate(path)

        img = nib.load(str(path))

        return img

    def validate(path: Path) -> None:
        """Validate the file path and extension.

        Args:
            path (Path): Path to the NIfTI file.
        """
        if not path.exists():
            raise FileNotFoundError(f"NIfTI file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Provided path is not a file: {path}")
        if path.suffix not in {".nii", ".gz"} and not path.name.endswith(".nii.gz"):
            raise ValueError(f"File must be .nii or .nii.gz: {path.name}")

    def shape(self) -> tuple:
        """Returns the shape of the NIfTI image."""
        # This method wont be used.
        pass

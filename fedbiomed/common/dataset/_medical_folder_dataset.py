# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Dataset implementation for BIDS-like MedicalFolderDataset
"""

from os import PathLike
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

from fedbiomed.common.dataset_controller import NewMedicalFolderController
from fedbiomed.common.dataset_types import DatasetDataItem, Transform

from ._dataset import StructuredDataset


class NewMedicalFolderDataset(StructuredDataset, NewMedicalFolderController):
    def __init__(
        self,
        framework_transform: Transform = None,
        framework_target_transform: Transform = None,
        generic_transform: Transform = None,
        generic_target_transform: Transform = None,
        # Keep actual names for backward compatibility
        #
        # reader_images_transform : Transform = None,
        # reader_images_target_transform : Transform = None,
        # reader_demographics_transform : Transform = None.
        transform: Transform = None,
        target_transform: Transform = None,
        demographics_transform: Transform = None,
        # Keep actual names for backward compatibility
        data_modalities: Optional[Union[str, Iterable[str]]] = "T1",
        target_modalities: Optional[Union[str, Iterable[str]]] = "label",
        tabular_file: Optional[Union[str, PathLike, Path, None]] = None,
        index_col: Optional[Union[int, str, None]] = None,
    ) -> None:
        """Class constructor"""

    # Implement abstract methods

    def __len__(self) -> int:
        """Get number of samples"""

    # Nota: use Controller._get_nontransformed_item
    def __getitem__(self, index: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Retrieve a data sample"""

    # Support returning samples in format for torch training plan
    #
    # Possible alternate implementation: class to be inherited by datasets that implement it
    # (multiple inheritance).
    def to_torch(self) -> bool:
        """Request dataset to return samples for a torch training plan

        Return True if accepted by the dataset (no generic transform to apply)
        Ignore + issue warning + return False if generic transform needs to be applied
        """

    # Still needed or replaced by implementation in DLP ? cf current MedicalFolderDataset
    #
    # def set_dataset_parameters(self, parameters: dict):

    # Notes for refactoring (from https://notes.inria.fr/aRJElvYmTFGNeU53HYJeCw?edit
    # "review MedicalFolderDataset implementation" )
    #
    # 1. no other public methods to implement in MedicalFolderdataset ?
    # 2. no other public methods to make private in MedicalFolderDataset ?
    # 3. properties make code very difficult to read (vs getter). Maybe ok after refactoring.

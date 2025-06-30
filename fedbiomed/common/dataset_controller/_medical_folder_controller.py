# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Controller implementation for BIDS-like MedicalFolderController
"""

from pathlib import Path
from typing import Tuple, Union

from fedbiomed.common.dataset_types import DatasetDataItem

from ._controller import Controller


class NewMedicalFolderController(Controller):
    def __init__(
        self,
        root: Union[str, Path],
        # Any other parameter ?
    ) -> None:
        """Class constructor"""

    # Implement abstract methods

    def validate(self) -> None:
        pass

    def _get_nontransformed_item(
        self, index: int
    ) -> Tuple[DatasetDataItem, DatasetDataItem]:
        pass

    # Notes for refactoring (from https://notes.inria.fr/aRJElvYmTFGNeU53HYJeCw?edit
    # "review MedicalFolderDataset implementation" )
    #
    # 1. the following methods from MedicalFolderDataset should probably move
    # to MedicalFolderController
    #
    # demographics
    # subjects_has_all_modalities
    # subjects_registered_in_demographics
    # load_images
    # subject_folders
    # shape
    #
    # 2. methods used as private should be made `private`
    # root (getter)
    # is_modalities_existing
    # complete_subjects
    # subjects_with_imaging_data_folders
    # demographics
    # subjects_registered_in_demographics
    # load_images
    # subject_folders
    #
    # 3. properties make code very difficult to read (vs getter). Maybe ok after refactoring.

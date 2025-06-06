# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract classes for controllers
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict
from pathlib import Path

from fedbiomed.common.dataset_reader import Reader
from fedbiomed.common.dataset_types import DatasetDataItem, DatasetShape
from fedbiomed.common.dataloadingplan import DataLoadingPlanMixin


class Controller(ABC, DataLoadingPlanMixin):

    # Possible implementation
    # key is a convenient name, internal to the class, for identifying a Reader
    # Enables common implementation, eg for `to_torch()`
    _readers : Dict[str, Reader]

    def __init__(
            self,
            root: Path,
            *args,
            **kwargs
    ) -> None:
        """Class constructor"""


    # Nota: no need to implement `static` method as in current MedicalFolderDataset
    #
    # Nota: if we want to do advanced coherence validation, coherence of a dataset
    # is not only validating coherence for each reader. Need to cross check data and
    # sample list of different modalities, which requires more code in this method
    # for this Dataset, plus support from the Reader
    # 
    # Nota: validate is not only called at initialization of object. If we change DLP
    # then we may want to validate again with different DLP context (cf MedicalFolderDataset)
    @abstractmethod
    def validate(self) -> None:
        """Validate coherence of dataset

        Raises exception if coherence issue found
        """

    # Nota: includes filtering of DLP, not of transforms
    # Nota: can use _get_nontransformed_item and/or Reader.shape
    # but can be more complicated than just querying `shape` of Readers
    # as some Controller may (eg) filter out incomplete samples
    @abstractmethod
    def shape(self) -> DatasetShape:
        """Returns shape of a dataset"""


    # Cf current implementation of MedicalFolderDataset, uses Reader
    # Nota: includes filtering of DLP, not of transforms
    @abstractmethod
    def _get_nontransformed_item(self, index: int) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Retrieve a data sample without applying transforms"""


    # Optional methods which can be implemented (or not) by every dataset
    # Possible implementation: class to be inherited by datasets that implement it
    # (multiple inheritance).


    # TODO: additional methods for exploring data (folders, modalities, subjects),
    # depending on Reader

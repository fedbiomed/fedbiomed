# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract classes for controllers
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Union

from fedbiomed.common.dataloadingplan import DataLoadingPlanMixin
from fedbiomed.common.dataset_reader import Reader
from fedbiomed.common.dataset_types import DatasetData, DatasetDataItem, DatasetShape


class Controller(ABC, DataLoadingPlanMixin):
    # Possible implementation
    # key is a convenient name, internal to the class, for identifying a Reader
    # Enables common implementation, eg for `validate()`
    _readers: Dict[str, Reader]

    # Store dataset structure and metadata
    #
    # Nota: can use _get_nontransformed_item and/or Reader.shape
    # but shape part be more complicated than just querying `shape` of Readers
    # as some Controller may (eg) filter out incomplete samples
    _dataset_data_meta: DatasetData

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    @property
    def root(self):
        return self._root
    
    @root.setter
    def root(self, path_input: Union[str, Path]):
        if not isinstance(path_input, (str, Path)):
            raise TypeError(f"Expected a string or Path, got {type(path_input).__name__}")
        path = Path(path_input).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        self._root = path


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
    # Probably uses content of `DatasetData`
    #
    # Nota: probably common to all datasets, as we scan `DatasetData`
    def shape(self) -> DatasetShape:
        """Returns shape of a dataset"""

    # Nota: includes filtering of DLP, not of transforms
    #
    # Nota: probably common to all datasets, as we scan `DatasetData`
    #
    # Nota: return a deep copy to avoid later modification ?
    def dataset_meta(self) -> DatasetData:
        """Returns full metadata of a dataset"""

    # Future extensions: methods to set some metadata

    # Cf current implementation of MedicalFolderDataset, uses Reader
    # Nota: includes filtering of DLP, not of transforms
    @abstractmethod
    def _get_nontransformed_item(
        self, index: int
    ) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Retrieve a data sample without applying transforms"""

    # Additional methods for exploring data (folders, modalities, subjects),
    # depending on Reader

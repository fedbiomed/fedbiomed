# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader implementation for NIFTI image set
"""

from pathlib import Path

import torch

from fedbiomed.common.dataset_types import Transform, ReaderItemShape
from ._reader import Reader


class NiftiReader(Reader):

    def __init__(
            self,
            root: Path,
            native_transform : Transform = None,
            native_target_transform : Transform = None,
            # Any other parameter ?
            modality: str = 'T1',
    ) -> None:
        """Class constructor"""


    def __len__(self) -> int:
        """Get number of samples"""

    # Nota: does not include filtering of DLP, which is unknown to Reader

    # CAVEAT: with current implementation sketch, a `Reader` is for *one* modality
    # eg if we have a `MedicalFolderDataset` with 2 modalitites `T1` and `T2`
    # we will instantiate 2 `Reader`s crawling through the same tree.
    # - cons: a bit more complexity ?
    # - pros: clearly separate `Dataset` responsibility for eg handling "incomplete" subjects
    def __getitem__(self, index: int) -> torch.Tensor:
        """Retrieve a data sample"""

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def validate(self) -> None:
        """Validate coherence of data modality served by a reader

        Raises exception if coherence issue found
        """

    # Nota: does not include filtering of DLP, which is unknown to Reader
    def shape(self) -> ReaderItemShape:
        """Returns shape of a data modality served by a reader"""


    # Optional methods which can be implemented (or not) by some readers
    # Code is specific to each reader

    # This is needed by MedicalFolderDataset (and probably most multimodal datasets)
    # to coordinate *which* subject they retrieve next. "Next" with `__getitem__` for
    # a `Reader` for one modality does not address same subject as "next" for another
    # `Reader` of another modality. Thus only dataset can ensure coherence
    # of "next" sample retrieval, using a tag.
    #
    def getitem_by_tag(self, tag: str) -> torch.Tensor:
        """Retrieve a data sample identified by an arbitrary string tag"""


        # Nothing to do to support it ?
    def to_torch(self) -> None:
        """Request reader to return samples for a torch training plan
    """ 

    # Additional methods for exploring data, depending on Reader

"""
Datasets submodule for Fed-BioMed
---------------------------------

This submodule provides the dataset classes for common cases of use in healthcare:
- NIFTI: For NIFTI medical images
"""
from abc import ABC
from os import PathLike
from pathlib import Path
from typing import Union, Tuple

import torch
from monai.data import ITKReader
from monai.transforms import Transform, LoadImage, ToTensor, Compose
from torch import Tensor
from torch.utils.data import Dataset


class NIFTIFolderDataset(Dataset, ABC):
    """A Generic class for loading NIFTI Images using the folder structure as the target labels.

    Supported formats:
        * NIFTI and compressed NIFTI files: `.nii`, `.nii.gz`

    This is a Dataset useful in classification tasks. Its usage is quite simple.
    Images must be contained by folders that describe the group/class they belong to.

    ```
    nifti_dataset_root_folder
    ├── control_group
    │   ├── subject_1.nii
    │   └── subject_2.nii
    │   └── ...
    └── disease_group
        ├── subject_1.nii
        └── subject_2.nii
        └── ...
    ```
    """

    ALLOWED_EXTENSIONS = ['.nii', '.nii.gz', '.jpeg']
    files = []
    class_names = []
    targets = []

    def __init__(self, root: Union[str, PathLike, Path],
                 transform: Transform = None,
                 target_transform: Transform = None
                 ):
        """
        Args:
            - root (str, PathLike): folder where the data is located.
            - transform (Transform): transforms to be applied on data.
            - target_transform (Transform): transforms to be applied on target.
        """
        self.root_dir = Path(root).expanduser()
        self.transform = transform
        self.target_transform = target_transform
        self.reader = Compose([
            LoadImage(ITKReader(), image_only=True),
            ToTensor()
        ])

        self._explore_root_folder()

    def _explore_root_folder(self) -> None:
        """Lists all files found in folder"""
        # Search files that correspond to the following criteria:
        # 1. Extension in ALLOWED extensions
        # 2. File folder's parent must be root (inspects folder only one level of depth)
        self.files = [p.resolve() for p in self.root_dir.glob("**/*") if ''.join(p.suffixes) in self.ALLOWED_EXTENSIONS and p.parent.parent == self.root_dir]

        # Create class names dictionary
        self.class_names = tuple(set([p.parent.name for p in self.files]))

        # Assign numerical value to target 0...n_classes
        self.targets = torch.tensor([self.class_names.index(p.parent.name) for p in self.files]).long()

    def __getitem__(self, item) -> Tuple[Tensor, Tensor]:
        img = self.reader(self.files[item])
        target = self.targets[item]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(img)
        return img, target

    def __len__(self):
        return len(self.files)

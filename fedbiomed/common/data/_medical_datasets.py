"""
Common healthcare data manager

Provides classes managing dataset for common cases of use in healthcare:
- NIFTI: For NIFTI medical images
"""
from os import PathLike
from pathlib import Path
from typing import Union, Tuple, List

import torch
from monai.data import ITKReader
from monai.transforms import Transform, LoadImage, ToTensor, Compose
from torch import Tensor
from torch.utils.data import Dataset

from fedbiomed.common.exceptions import FedbiomedDatasetError
from fedbiomed.common.constants import ErrorNumbers


class NIFTIFolderDataset(Dataset):
    """A Generic class for loading NIFTI Images using the folder structure as the target classes' labels.

    Supported formats:
    - NIFTI and compressed NIFTI files: `.nii`, `.nii.gz`

    This is a Dataset useful in classification tasks. Its usage is quite simple, quite near
    from `torchvision.datasets.ImageFolder`.
    Images must be contained in first level subfolders (level 2+ subfolders are ignored)
    that describe the target class they belong to (target class label is the name of the folder).

    ```
    nifti_dataset_root_folder
    ├── control_group
    │   ├── subject_1.nii
    │   └── subject_2.nii
    │   └── ...
    └── disease_group
        ├── subject_3.nii
        └── subject_4.nii
        └── ...
    ```

    In this example, there are 4 samples (one from each *.nii file),
    2 target class, with labels `control_group` and `disease_group`.
    `subject_1.nii` has class label `control_group`, `subject_3.nii` has class label `disease_group`,etc.
    """

    # constant, thus can be a class variable
    _ALLOWED_EXTENSIONS = ['.nii', '.nii.gz']

    def __init__(self, root: Union[str, PathLike, Path],
                 transform: Union[Transform, None] = None,
                 target_transform: Union[Transform, None] = None
                 ):
        """Constructor of the class

        Args:
            root: folder where the data is located.
            transform: transforms to be applied on data.
            target_transform: transforms to be applied on target indexes.
        """
        self._files = []
        self._class_labels = []
        self._targets = []

        self._root_dir = Path(root).expanduser()
        self._transform = transform
        self._target_transform = target_transform
        self._reader = Compose([
            LoadImage(ITKReader(), image_only=True),
            ToTensor()
        ])

        self._explore_root_folder()

    def _explore_root_folder(self) -> None:
        """Scans all files found in folder structure to populate dataset

        Raises:
            FedbiomedDatasetError: If compatible image files are not found
        """

        # Search files that correspond to the following criteria:
        # 1. Extension in ALLOWED extensions
        # 2. File folder's parent must be root (inspects folder only one level of depth)
        self._files = [p.resolve() for p in self._root_dir.glob("**/*")
                       if ''.join(p.suffixes) in self._ALLOWED_EXTENSIONS and
                       p.parent.parent == self._root_dir]

        # Create class labels dictionary
        self._class_labels = list(set([p.parent.name for p in self._files]))

        # Assign numerical value to target 0...n_classes
        self._targets = torch.tensor([self._class_labels.index(p.parent.name) for p in self._files]).long()

        # Raise error if empty dataset
        if len(self._files) == 0 or len(self._targets) == 0:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB612.value}: Cannot create dataset because no compatible files found"
                f" in the {self._root_dir}.")

    def labels(self) -> List[str]:
        """Retrieves the labels of the target classes.

        Target label index is the index of the corresponding label in this list.

        Returns:
            List of the labels of the target classes.
        """
        return self._class_labels

    def files(self) -> List[Path]:
        """Retrieves the paths to the sample images.

        Gives sames order as when retrieving the sample images (eg `self.files[0]`
        is the path to `self.__getitem__[0]`)

        Returns:
            List of the absolute paths to the sample images
        """
        return self._files

    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        """Gets item from dataset

        If `transform` is not `None`, applies it to the image.
        If `target_transform` is not `None`, applies it to the target class index

        Args:
            item: Index to select single sample from dataset

        Returns:
            A tuple composed of the input sample (an image) and a target sample index (label index).
        """
        img = self._reader(self._files[item])
        target = int(self._targets[item])

        if callable(self._transform):
            img = self._transform(img)
        if callable(self._target_transform):
            target = int(self._target_transform(target))

        return img, target

    def __len__(self) -> int:
        """Gets number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self._files)

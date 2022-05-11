"""
Datasets submodule for Fed-BioMed
---------------------------------

This submodule provides the dataset classes for common cases of use in healthcare:
- NIFTI: For NIFTI medical images
"""
from abc import ABC
from os import PathLike
from pathlib import Path
from typing import Union, Tuple, Dict, Iterable, Optional, List

import torch
from cachetools import cached
from monai.data import ITKReader
from monai.transforms import Transform, LoadImage, ToTensor, Compose
from torch import Tensor
from torch.utils.data import Dataset


class NIFTIFolderDataset(Dataset, ABC):
    """A Generic class for loading NIFTI Images using the folder structure as the target labels.

    Supported formats:
    - NIFTI and compressed NIFTI files: `.nii`, `.nii.gz`

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

    ALLOWED_EXTENSIONS = ['.nii', '.nii.gz']
    files = []
    class_names = []
    targets = []

    def __init__(self, root: Union[str, PathLike, Path],
                 transform: Transform = None,
                 target_transform: Transform = None
                 ):
        """
        Args:
            root: folder where the data is located.
            transform: transforms to be applied on data.
            target_transform: transforms to be applied on target.
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
        self.files = [p.resolve() for p in self.root_dir.glob("**/*") if
                      ''.join(p.suffixes) in self.ALLOWED_EXTENSIONS and p.parent.parent == self.root_dir]

        # Create class names dictionary
        self.class_names = tuple(set([p.parent.name for p in self.files]))

        # Assign numerical value to target 0...n_classes
        self.targets = torch.tensor([self.class_names.index(p.parent.name) for p in self.files]).long()

        # Raise error if empty dataset
        if len(self.files) == 0 or len(self.targets) == 0:
            raise FileNotFoundError(f"Not compatible files were found in the {self.root_dir}.")

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        """ Gets item from dataset

        Args:
            item: Key/index to select single sample from dataset

        Returns:
            inputs: Input sample
            target: Target sample
        """
        img = self.reader(self.files[item])
        target = self.targets[item]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(img)
        return img, target

    def __len__(self):
        return len(self.files)


class BIDSDataset(Dataset):
    """Torch dataset following the BIDS Structure.

    The BIDS structure has the following pattern:

    └─ my_processed_data/
       └─ sub-01/
           └─ ses-test/
              ├─ anat/
              │  └─ sub-01_ses-test_T1w.nii.gz
              └─ func/
                 ├─ sub-01_ses-test_task-overtverbgeneration_run-1_bold.nii.gz
                 ├─ sub-01_ses-test_task-overtverbgeneration_run-2_bold.nii.gz

    Certain modalities are allowed per subject in the dataset. Each of these is represented by a folder within each
    subject's directory.:

        * `T1` sequence magnetic resonance image
        * `T2` sequence magnetic resonance image
        * `label` which contains segmentation masks
    """
    ALLOWED_MODALITIES = ['T1', 'T2', 'LABEL']

    def __init__(self,
                 root: Union[PathLike, Path],
                 transform: Dict[str, Transform],
                 data_modalities: Optional[Union[str, Iterable[str]]] = 'T1',
                 target_modalities: Optional[Union[str, Iterable[str]]] = 'label'
                 ):
        """Constructor for class `BIDSDataset`.

        Args:
            root: Root folder containing all the subject directories.
            transform: A function or transform that preprocesses each data source (image).
            data_modalities (str, Iterable): Modality or modalities to be used as data sources.
            target_modalities (str, Iterable): Modality or modalities that will be used as target sources.
        """
        self.root_folder = Path(root).expanduser().resolve()
        self.transform = transform
        self.data_modalities = [data_modalities] if isinstance(data_modalities, str) else data_modalities
        self.target_modalities = [target_modalities] if isinstance(data_modalities, str) else target_modalities

    @cached
    @property
    def subject_list(self) -> List[Path]:
        """Loads a subject list by iterating over the root directory of the dataset."""
        # TODO: Iter over folders and get a list of subjects (verify for non-empty directories)
        subject_folders = self.root_folder.iterdir()

    def _load_modality(self):
        """Loads data from a particular modality."""
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

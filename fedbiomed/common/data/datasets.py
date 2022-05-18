"""
Datasets submodule for Fed-BioMed
---------------------------------

This submodule provides the dataset classes for common cases of use in healthcare:
- NIFTI: For NIFTI medical images
"""
from abc import ABC
from os import PathLike
from pathlib import Path
from typing import Union, Tuple, Dict, Iterable, Optional, List, Any, Callable

import pandas as pd
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


def _check_and_reformat_transforms(
        transform: Union[Callable, Dict[str, Callable]],
        modalities: Union[str, Iterable[str]]
) -> Dict[str, Callable]:
    """Checks and formats transforms into a dictionary of transforms.

    Args:
        transform: Function or dictionary of functions for preprocessing data.
        modalities: Modalities to be considered.

    Returns:
        A dict of transforms compatible with the provided modalities.
    """
    if isinstance(modalities, str):
        modalities = [modalities]

    if isinstance(transform, dict):
        for modality in transform.keys():
            assert modality in modalities, f'Modality `{modality}` is not present in {modalities}'
        return transform

    if len(modalities) == 1:
        return {modalities[0]: transform}


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
    ALLOWED_EXTENSIONS = ['.nii', '.nii.gz']

    def __init__(self,
                 root: Union[str, PathLike, Path],
                 transform: Union[Callable, Dict[str, Callable]] = None,
                 data_modalities: Optional[Union[str, Iterable[str]]] = 'T1',
                 target_modalities: Optional[Union[str, Iterable[str]]] = 'label',
                 target_transform: Union[Callable, Dict[str, Callable]] = None,
                 tabular_file: Union[str, PathLike, Path] = None,
                 index_col: Union[int, str] = 0,
                 ):
        """Constructor for class `BIDSDataset`.

        Args:
            root: Root folder containing all the subject directories.
            transform: A function or transform that preprocesses each data source (image).
            data_modalities (str, Iterable): Modality or modalities to be used as data sources.
            target_modalities (str, Iterable): Modality or modalities that will be used as target sources.
        """
        self.root_folder = Path(root).expanduser().resolve()
        self.data_modalities = [data_modalities] if isinstance(data_modalities, str) else data_modalities
        self.target_modalities = [target_modalities] if isinstance(data_modalities, str) else target_modalities

        self.transform = _check_and_reformat_transforms(transform, data_modalities)
        self.target_transform = _check_and_reformat_transforms(target_transform, target_modalities)

        self.tabular_file = Path(tabular_file).expanduser().resolve() if tabular_file is not None else None
        self.index_col = index_col

        # Image loader
        self.reader = Compose([
            LoadImage(ITKReader(), image_only=True),
            ToTensor()
        ])

        # Assert transforms format. They should be provided as dictionaries.
        # E.g. {'T1': Normalize(...), 'T2': ToTensor()}
        assert isinstance(self.transform, dict), f'As you have multiple data modalities, ' \
                                                 f'transforms have to a dictionary using ' \
                                                 f'the modality keys: {self.data_modalities}'
        assert isinstance(self.target_transform, dict), f'As you have multiple target modalities, ' \
                                                        f'transforms have to a dictionary using ' \
                                                        f'the modality keys: {self.target_modalities}'

    @property
    def demographics(self) -> pd.DataFrame:
        """Loads tabular data file (supports excel, csv, tsv and colon separated value files)."""
        try:
            return pd.read_csv(self.tabular_file, index_col=self.index_col, engine='python')
        except UnicodeDecodeError:
            return pd.read_excel(self.tabular_file, index_col=self.index_col)

    def load_images(self, subject_folder: Path, modalities: list):
        files = {}

        for modality in modalities:
            image_folder = subject_folder.joinpath(modality)
            nii_files = [p.resolve() for p in image_folder.glob("**/*")
                         if ''.join(p.suffixes) in self.ALLOWED_EXTENSIONS]

            # Load the first, we assume there is going to be a single image per modality for now.
            img_path = nii_files[0]
            img = self.reader(img_path)
            files[modality] = img
        return files

    def __getitem__(self, item):
        subject_folder = self.complete_subject_folders[item]
        data = self.load_images(subject_folder, modalities=self.data_modalities)
        targets = self.load_images(subject_folder, modalities=self.target_modalities)

        # Apply transforms to data elements
        for modality, transform in self.transform.items():
            if transform:
                data[modality] = transform(data[modality])

        # Apply transform to target elements
        for modality, target_transform in self.target_transform.items():
            if target_transform:
                targets[modality] = target_transform(targets[modality])

        return dict(data=data, target=targets)

    def __len__(self):
        return len(self.complete_subject_folders)

    @property
    def subject_folders(self) -> List[Path]:
        """Loads a subject list by iterating over the root directory of the dataset."""
        subject_folders = [subject_folder for subject_folder in self.root_folder.iterdir() if subject_folder.is_dir()]
        return subject_folders

    @property
    def complete_subject_folders(self) -> List[Path]:
        """Returns subject folders of only those who have their complete modalities"""
        all_modalities = self.data_modalities + self.target_modalities
        complete_subject_folders = []

        # Iterate over all folders and append only the subjects that have all modalities present.
        for subject_folder in self.subject_folders:
            if all([subject_folder.joinpath(modality).is_dir() for modality in all_modalities]):
                complete_subject_folders.append(subject_folder)

        return complete_subject_folders


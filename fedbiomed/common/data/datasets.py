"""
Datasets submodule for Fed-BioMed
---------------------------------

This submodule provides the dataset classes for common cases of use in healthcare:
- NIFTI: For NIFTI medical images
"""
import os

from abc import ABC
from os import PathLike
from pathlib import Path
from typing import Union, Tuple, Dict, Iterable, Optional, List, Any, Callable

import pandas as pd
import torch
from functools import cache
from monai.data import ITKReader
from monai.transforms import Transform, LoadImage, ToTensor, Compose
from torch import Tensor
from torch.utils.data import Dataset

from fedbiomed.common.exceptions import FedbiomedBIDSDatasetError, FedbiomedError
from fedbiomed.common.logger import logger


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
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.files)


class BIDSController:
    """Controller class for BIDS dataset.

    Contains methods to validate BIDS folder hierarchy  and extract folder-base meta data
    in formation such as modalities number of subject etc.
    """

    def __init__(self, root: Union[str, Path, None] = None):
        """Constructs BIDSControler"""
        if root is not None:
            root = self.validate_bids_root_folder(root)

        self._root = root

    @property
    def root(self):
        """Root property of BIDSController"""
        return self._root

    @root.setter
    def root(self, path: Union[str, Path]):
        """ Setter for root directory of BIDS dataset

        Args:
            path: Path to set as root directory of BIDS dataset
        """
        path = self.validate_bids_root_folder(path)
        self._root = path

    def modalities(self):
        """ Gets all available modalities under root directory

        Returns:
             List of unique available modalities
        """

        # Accept only folders that don't start with "." and "_"
        modalities = [f.name for f in self._root.glob("*/*") if f.is_dir() and not f.name.startswith((".", "_"))]
        return list(set(modalities)), modalities

    def is_modalities_existing(self, subject: str, modalities: List[str]) -> List[bool]:
        """Checks whether given modalities exists in the subject directory

        Args:
            subject: Subject ID or subject folder name
            modalities: List of modalities to check

        Returns:
            List of `bool` that represents whether modality is existing respectively for each of modality.
        """
        return [self._root.joinpath(subject, modality).is_dir() for modality in modalities]

    def complete_subjects(self, subjects: List[str], modalities: List[str]) -> List[str]:
        """Retries subjects that have given all the modalities.

        Args:
            subjects: List of subject folder names
            modalities: List of required modalities

        Returns:
            List of subject folder names that have required modalities
        """
        return [subject for subject in subjects if all(self.is_modalities_existing(subject, modalities))]

    def subjects(self) -> List[str]:
        """Retries subject folder names under BIDS roots directory.

        Returns:
            subject folder names under BIDS roots directory.
        """
        return [f.name for f in self._root.iterdir() if f.is_dir() and not f.name.startswith(".")]

    def available_subjects(self,
                           subjects_from_index: [list, pd.Series, None],
                           subjects_from_folder: list = None) -> tuple[list[str], list[str], list[str]]:
        """Checks missing subject folders and missing entries in demographics

        Args:
            subjects_from_index: Given subject folder names in demographics
            subjects_from_folder: List of subject folder names to get intersection of given subject_from_index

        Returns:
            complete_subjects:
            missing_subject_folders:
            missing_entries:
        """

        # Select oll subject folders if it is not given
        if subjects_from_folder is None:
            subjects_from_folder = self.subjects()

        # Missing subject that will cause warnings
        missing_subject_folders = list(set(subjects_from_index) - set(subjects_from_folder))

        # Missing entries that will cause errors
        missing_entries = list(set(subjects_from_folder) - set(subjects_from_index))

        # Intersection
        complete_subjects = list(set(subjects_from_index).intersection(set(subjects_from_folder)))

        return complete_subjects, missing_subject_folders, missing_entries

    @staticmethod
    def read_demographics(path: Union[str, Path], index_col: int):
        """ Read demographics tabular file for BIDS dataset

        """
        path = Path(path)
        if not path.is_file():
            raise FedbiomedBIDSDatasetError(f"Demographics should be a file not a directory")

        if 'xls' in path.suffix.lower():
            return pd.read_excel(path, index_col=index_col)
        else:
            return pd.read_csv(path, index_col=index_col, engine='python')

    @staticmethod
    def validate_bids_root_folder(path: Union[str, Path]) -> Path:
        """ Validates BIDS root directory by checking folder structure

        The BIDS structure has the following pattern:

        ```
        └─ BIDS_root/
            └─ sub-01/
                ├─ T1/
                │  └─ sub-01_xxx.nii.gz
                └─ T2/
                    ├─ sub-01_xxx.nii.gz
        ```

        Args:
            path:

        Returns:
            Path to root folder of BIDS dataset

        Raises:
            FedbiomedError: - If path is not an instance of `str` or `pathlib.Path`
                            - If path is not a directory
        """
        if not isinstance(path, (Path, str)):
            raise FedbiomedError(f"The argument root should an instance of `Path` or `str`, but got {type(path)}")

        if not isinstance(path, Path):
            path = Path(path)

        if not path.is_dir():
            raise FedbiomedError(f"Root for BIDS dataset should be a directory.")

        directories = [f for f in path.iterdir() if f.is_dir()]
        if len(directories) == 0:
            raise FedbiomedBIDSDatasetError("Root folder of BIDS should contain subject folders, but no "
                                            "sub folder has been found. ")

        modalities = [f for f in path.glob("*/*") if f.is_dir()]
        if len(modalities) == 0:
            raise FedbiomedBIDSDatasetError("Subject folders for BIDS should contain modalities as folders. Folder "
                                            "structure should be root/<subjects>/<modalities>")

        return path


class BIDSDeploymentController(BIDSController):

    def __init__(self, root: str = None, demographics_file: str = None):
        super().__init__(root=root)
        self._demographics = demographics_file

    def is_modalities(self):
        """"""
        unique_modalities, modalities = self.modalities()
        if len(unique_modalities) == len(modalities):
            raise FedbiomedBIDSDatasetError("")

        return True


class BIDSDataset(Dataset):
    """Torch dataset following the BIDS Structure.

    Certain modalities are allowed per subject in the dataset. Each of these is represented by a folder within each
    subject's directory.:

    * `T1` sequence magnetic resonance image
    * `T2` sequence magnetic resonance image
    * `label` which contains segmentation masks

    """
    # ALLOWED_MODALITIES = ['T1', 'T2', 'LABEL']

    ALLOWED_EXTENSIONS = ['.nii', '.nii.gz']

    def __init__(self,
                 root: Union[str, PathLike, Path],
                 data_modalities: Optional[Union[str, Iterable[str]]] = 'T1',
                 transform: Union[Callable, Dict[str, Callable]] = None,
                 target_modalities: Optional[Union[str, Iterable[str]]] = 'label',
                 target_transform: Union[Callable, Dict[str, Callable]] = None,
                 tabular_file: Union[str, PathLike, Path, None] = None,
                 index_col: Union[int, str, None] = None,
                 ):
        """Constructor for class `BIDSDataset`.

        Args:
            root: Root folder containing all the subject directories.
            data_modalities (str, Iterable): Modality or modalities to be used as data sources.
            transform: A function or dict of function transform(s) that preprocess each data source.
            target_modalities: (str, Iterable): Modality or modalities to be used as target sources.
            target_transform: A function or dict of function transform(s) that preprocess each target source.
            tabular_file: Path to a CSV or Excel file containing the demographic information from the patients.
            index_col: Column name in the tabular file containing the subject ids which mush match the folder names.
        """

        try:
            # Validate root directory
            self._bids_controller = BIDSController(root=root)
        except FedbiomedError as e:
            raise FedbiomedBIDSDatasetError(f"Can not create BIDS dataset due to root path. {e}")

        self._root = Path(root).expanduser().resolve()
        self._tabular_file = tabular_file
        self._index_col = index_col

        self.data_modalities = [data_modalities] if isinstance(data_modalities, str) else data_modalities
        self.target_modalities = [target_modalities] if isinstance(data_modalities, str) else target_modalities

        self.transform = self._check_and_reformat_transforms(transform, data_modalities)
        self.target_transform = self._check_and_reformat_transforms(target_transform, target_modalities)

        # Image loader
        self._reader = Compose([
            LoadImage(ITKReader(), image_only=True),
            ToTensor()
        ])

        self._complete_subject_folders = None

        # Raise if transform objects are not provided as dictionaries.
        # E.g. {'T1': Normalize(...), 'T2': ToTensor()}
        if not isinstance(self.transform, dict):
            raise FedbiomedBIDSDatasetError(f'As you have multiple data modalities, transforms have to a dictionary '
                                            f'using the modality keys: {self.data_modalities}')
        if not isinstance(self.target_transform, dict):
            raise FedbiomedBIDSDatasetError(f'As you have multiple target modalities, transforms have to a dictionary '
                                            f'using the modality keys: {self.target_modalities}')

    def __getitem__(self, item):

        # For the first item retrieve complete subject folders
        if self._complete_subject_folders is None:
            self._complete_subject_folders = self.subject_folders()

        # Get subject folder
        subject_folder = self._complete_subject_folders[item]

        # Load data modalities
        data = self.load_images(subject_folder, modalities=self.data_modalities)

        # Load target modalities
        targets = self.load_images(subject_folder, modalities=self.target_modalities)

        # Demographics
        demographics = self._get_from_demographics(subject_id=subject_folder.name)

        # Apply transforms to data elements
        for modality, transform in self.transform.items():
            if transform:
                data[modality] = transform(data[modality])

        # Apply transform to target elements
        for modality, target_transform in self.target_transform.items():
            if target_transform:
                targets[modality] = target_transform(targets[modality])

        return dict(data=data, target=targets, demographics=demographics)

    def __len__(self):
        length = len(self.complete_subject_folders)
        assert length > 0, 'Dataset cannot be empty. ' \
                           'Check again that the folder and ' \
                           'the tabular data (if provided) exist and match properly.'
        return length

    @property
    def tabular_file(self):
        return self._tabular_file

    @property
    def index_col(self):
        return self._index_col

    @tabular_file.setter
    def tabular_file(self, value: Union[str, Path]):
        """Sets `tabular_file` property

        Args:
            value:

        Returns:

        """
        if not isinstance(value, (str, Path)):
            raise FedbiomedBIDSDatasetError(f"Path for tabular file should be of `str` or `Path` type, "
                                            f"but got {type(value)} ")

        path = Path(value)
        if not path.is_file():
            raise FedbiomedBIDSDatasetError(f"Path should be a data file")

        self._tabular_file = Path(path).expanduser().resolve()

    @index_col.setter
    def index_col(self, value: int):
        """Sets `tabular_file` property.

        Args:
            value: Column index

        Raises:
            FedbiomedBIDSDatasetError: If value to set is not of `int` type
        """
        if not isinstance(value, int):
            raise FedbiomedBIDSDatasetError(f"`index_col` should be of `int` type, but got {type(value)}")

        self._index_col = value

    @property
    @cache
    def demographics(self) -> pd.DataFrame:
        """Loads tabular data file (supports excel, csv, tsv and colon separated value files)."""

        if self._tabular_file is None or self._index_col is None:
            raise FedbiomedBIDSDatasetError("Please set ")

        return self._bids_controller.read_demographics(self._tabular_file, self._index_col)

    def set_dataset_parameters(self, parameters: dict):
        """Sets dataset parameters.

        Args:
            parameters: Parameters to initialize

        Raises:
            FedbiomedBIDSDatasetError: If given parameters are not of `dict` type
        """
        if not isinstance(parameters, dict):
            raise FedbiomedBIDSDatasetError(f"Expected type for `parameters` is `dict, but got {type(parameters)}`")

        for key, value in parameters.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Trying to set undefined attribute {key} ti BIDSDataset")

    def load_images(self, subject_folder: Path, modalities: list):
        """Loads modality images in given subject folder

        Args:
            subject_folder: Subject folder where modalities are stored
            modalities: List of available modalities

        Returns:
            Subject image data as victories where keys represent each modality.
        """
        subject_data = {}

        for modality in modalities:
            image_folder = subject_folder.joinpath(modality)
            nii_files = [p.resolve() for p in image_folder.glob("**/*")
                         if ''.join(p.suffixes) in self.ALLOWED_EXTENSIONS]

            # Load the first, we assume there is going to be a single image per modality for now.
            img_path = nii_files[0]
            img = self._reader(img_path)
            subject_data[modality] = img

        return subject_data

    def _get_from_demographics(self, subject_id):
        """Extracts subject information from a particular subject in the form of a dictionary."""

        if self._tabular_file:
            demographics = self.demographics.loc[subject_id].to_dict()

            # Extract only compatible types for torch
            # TODO Decide what to do with missing variables
            return {key: val for key, val in demographics.items() if isinstance(val, (int, float, str, bool))}
        else:
            return {}

    def subject_folders(self) -> List[Path]:
        """Retries subject folder names of only those who have their complete modalities

        !!! info "Important"
            It is a cached property.

        Returns:
            List of subject directories that has all requested modalities
        """

        all_modalities = self.data_modalities + self.target_modalities
        subject_folder_names = self._bids_controller.available_subjects()

        # Get subject that has all requested modalities
        complete_subject_folders = self._bids_controller.complete_subjects(subject_folder_names, all_modalities)

        # If demographics are present
        if self._tabular_file is not None:
            # Consider only those who are present in the demographics file
            # Subject ids are the folder name which is contained as the basename of the path `.name`.
            # Subjects that contained all the modalities and the demographics: subs_with_all

            complete_subject_folders, *_ = self._bids_controller.available_subjects(
                subjects_from_folder=complete_subject_folders,
                subjects_from_index=self.demographics.index)

        return [self._root.joinpath(folder) for folder in complete_subject_folders]

    @staticmethod
    def _check_and_reformat_transforms(transform: Union[Callable, Dict[str, Callable]],
                                       modalities: Union[str, Iterable[str]]) -> Dict[str, Callable]:
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

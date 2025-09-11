# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Common healthcare data manager

Provides classes managing dataset for common cases of use in healthcare:
- NIFTI: For NIFTI medical images
"""

from os import PathLike
import os
from pathlib import Path
from typing import Union, Tuple, Dict, Iterable, Optional, List, Callable
from enum import Enum

import torch
import pandas as pd

from functools import cache
from monai.data import ITKReader
from monai.transforms import LoadImage, ToTensor, Compose
from torch import Tensor
from torch.utils.data import Dataset

from fedbiomed.common.exceptions import FedbiomedDatasetError, FedbiomedError
from fedbiomed.common.constants import ErrorNumbers, DataLoadingBlockTypes, DatasetTypes
from fedbiomed.common.data._data_loading_plan import DataLoadingPlanMixin


class MedicalFolderLoadingBlockTypes(DataLoadingBlockTypes, Enum):
    MODALITIES_TO_FOLDERS: str = "modalities_to_folders"


class NIFTIFolderDataset(Dataset):
    """A Generic class for loading NIFTI Images using the folder structure as the target classes' labels.

    Supported formats:
    - NIFTI and compressed NIFTI files: `.nii`, `.nii.gz`

    This is a Dataset useful in classification tasks. Its usage is quite simple, quite similar
    to `torchvision.datasets.ImageFolder`.
    Images must be contained in first level sub-folders (level 2+ sub-folders are ignored)
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
    _ALLOWED_EXTENSIONS = [".nii", ".nii.gz"]

    def __init__(
        self,
        root: Union[str, PathLike, Path],
        transform: Union[Callable, None] = None,
        target_transform: Union[Callable, None] = None,
    ):
        """Constructor of the class

        Args:
            root: folder where the data is located.
            transform: transforms to be applied on data.
            target_transform: transforms to be applied on target indexes.

        Raises:
            FedbiomedDatasetError: bad argument type
            FedbiomedDatasetError: bad root path
        """
        # check parameters type
        for tr, trname in (
            (transform, "transform"),
            (target_transform, "target_transform"),
        ):
            if not callable(tr) and tr is not None:
                raise FedbiomedDatasetError(
                    f"{ErrorNumbers.FB612.value}: Parameter {trname} has incorrect "
                    f"type {type(tr)}, cannot create dataset."
                )

        if (
            not isinstance(root, str)
            and not isinstance(root, PathLike)
            and not isinstance(root, Path)
        ):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB612.value}: Parameter `root` has incorrect type "
                f"{type(root)}, cannot create dataset."
            )

        # initialize object variables
        self._files = []
        self._class_labels = []
        self._targets = []

        try:
            self._root_dir = Path(root).expanduser()
        except RuntimeError as e:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB612.value}: Cannot expand path {root}, error message is: {e}"
            )

        self._transform = transform
        self._target_transform = target_transform
        self._reader = Compose([LoadImage(ITKReader(), image_only=True), ToTensor()])

        self._explore_root_folder()

    def _explore_root_folder(self) -> None:
        """Scans all files found in folder structure to populate dataset

        Raises:
            FedbiomedDatasetError: If compatible image files/folders for input or target are not found.
        """

        # Search files that correspond to the following criteria:
        # 1. Extension in ALLOWED extensions
        # 2. File folder's parent must be root (inspects folder only one level of depth)
        self._files = [
            p.resolve()
            for p in self._root_dir.glob("*/*")
            if "".join(p.suffixes) in self._ALLOWED_EXTENSIONS
        ]

        # note: no PermissionError raised. If directory cannot be listed it is ignored
        # except PermissionError as e:
        #    # can other exceptions occur ?
        #    raise FedbiomedDatasetError(
        #        f"{ErrorNumbers.FB612.value}: Cannot create dataset because scan of "
        #        f"directory {self._root_dir} failed with error message: {e}.")

        # Create class labels dictionary
        self._class_labels = list(set([p.parent.name for p in self._files]))

        # Assign numerical value to target 0...n_classes
        self._targets = torch.tensor(
            [self._class_labels.index(p.parent.name) for p in self._files]
        ).long()

        # Raise error if empty dataset
        if len(self._files) == 0 or len(self._targets) == 0:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB612.value}: Cannot create dataset because no compatible files found"
                f" in first level subdirectories of {self._root_dir}."
            )

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

        Raises:
            FedbiomedDatasetError: bad argument type
            FedbiomedDatasetError: cannot get sample
            FedbiomedDatasetError: cannot apply transform to sample
        """
        # check type and value for arguments
        if not isinstance(item, int):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB612.value}: Parameter `item` has incorrect type {type(item)}, "
                f"cannot get item from dataset."
            )
        if item < 0 or item >= len(self._files):
            # need an IndexError, cannot use a FedbiomedDatasetError
            raise IndexError(f"Bad index {item} in dataset samples")

        try:
            img = self._reader(self._files[item])
        except Exception as e:
            # many possible errors, too hard to list
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB612.value}: Cannot get sample number {item} from dataset, "
                f"error message is {e}."
            )

        target = int(self._targets[item])

        if self._transform is not None:
            try:
                img = self._transform(img)
            except Exception as e:
                # cannot list all exceptions
                raise FedbiomedDatasetError(
                    f"{ErrorNumbers.FB612.value}: Cannot apply transformation to source sample number {item} "
                    f"from dataset, error message is {e}."
                )

        if self._target_transform is not None:
            try:
                target = int(self._target_transform(target))
            except Exception as e:
                # cannot list all exceptions
                raise FedbiomedDatasetError(
                    f"{ErrorNumbers.FB612.value}: Cannot apply transformation to target sample number {item} "
                    f"from dataset, error message is {e}."
                )

        return img, target

    def __len__(self) -> int:
        """Gets number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self._files)


class MedicalFolderBase(DataLoadingPlanMixin):
    """Controller class for Medical Folder dataset.

    Contains methods to validate the MedicalFolder folder hierarchy and extract folder-base metadata
    information such as modalities, number of subject etc.
    """

    default_modality_names = ["T1", "T2", "label"]

    def __init__(self, root: Union[str, Path, None] = None):
        """Constructs MedicalFolderBase

        Args:
            root: path to Medical Folder root folder.
        """
        super(MedicalFolderBase, self).__init__()

        if root is not None:
            root = self.validate_MedicalFolder_root_folder(root)

        self._root = root

    @property
    def root(self):
        """Root property of MedicalFolderController"""
        return self._root

    @root.setter
    def root(self, path: Union[str, Path]):
        """Setter for root directory of Medical Folder dataset

        Args:
            path: Path to set as root directory of Medical Folder dataset
        """
        path = self.validate_MedicalFolder_root_folder(path)
        self._root = path

    def modalities_candidates_from_subfolders(self) -> Tuple[list, list]:
        """Gets all possible modality folders under root directory

        Returns:
             List of unique available modality folders appearing at least once
             List of all encountered modality folders in each subject folder, appearing once per folder
        """

        # Accept only folders that don't start with "." and "_"
        modalities = [
            f.name
            for f in self._root.glob("*/*")
            if f.is_dir() and not f.name.startswith((".", "_"))
        ]
        return sorted(list(set(modalities))), modalities

    # TODO: is `modality_folders_list` useful or should it be removed ?
    # should it return encountered modalities instead of encountered modality folders ?
    # (see `check_modalities`)
    def modalities(self) -> Tuple[list, list]:
        """Gets all modalities based either on all possible candidates or those provided by the DataLoadingPlan.

        Returns:
             List of unique available modalities
             List of all encountered modality folders in each subject folder, appearing once per folder
        """
        modality_candidates, modality_folders_list = (
            self.modalities_candidates_from_subfolders()
        )
        if (
            self._dlp is not None
            and MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS in self._dlp
        ):
            modalities = list(
                self._dlp[
                    MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS
                ].map.keys()
            )
            return modalities, modality_folders_list
        else:
            return modality_candidates, modality_folders_list

    def is_modalities_existing(self, subject: str, modalities: List[str]) -> List[bool]:
        """Checks whether given modalities exists in the subject directory

        Args:
            subject: Subject ID or subject folder name
            modalities: List of modalities to check

        Returns:
            List of `bool` that represents whether modality is existing respectively for each of modality.

        Raises:
            FedbiomedDatasetError: bad argument type
        """
        if not isinstance(subject, str):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Expected string for subject folder/ID, "
                f"but got {type(subject)}"
            )
        if not isinstance(modalities, list):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Expected a list for modalities, "
                f"but got {type(modalities)}"
            )
        if not all([type(m) is str for m in modalities]):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Expected a list of string for modalities, "
                f"but some modalities are "
                f"{' '.join([str(type(m) for m in modalities if isinstance(m, str))])}"
            )
        are_modalities_existing = list()
        for modality in modalities:
            modality_folder = self._subject_modality_folder(subject, modality)
            are_modalities_existing.append(
                bool(modality_folder)
                and self._root.joinpath(subject, modality_folder).is_dir()
            )
        return are_modalities_existing

    def _subject_modality_folder(
        self, subject_or_folder: Union[str, Path], modality: str
    ) -> Optional[Path]:
        """Get the folder containing the modality image for a subject.

        When we interrogate the DataLoadingPlan for the folder names corresponding to a given modality, we obtain
        a list of possibilities. But which one is right depends on which one actually exists among the subject's
        subfolders. This function is responsible for finding this, by inspecting all the subfolders and comparing
        their names to the list of possibilities provided by the DataLoadingPlan.

        It returns the intersection of two sets:
        - the modality folder names as returned by the DataLoadingPlan (or as inferred by the modality itself)
        - the first-level subfolders of the subject folder

        !!! warning "This function will not work properly if the modality images are in nested subfolders!"

        Args:
            subject_or_folder: the Path to the subject folder, or the name of the subject as a str
            modality: (str) the name of the modality

        Returns:
            a Path to the (unique) folder with the modality image, or None. None is returned if no folders
            were found, or if more than one matching folder was found.

        Raises:
            FedbiomedDatasetError: bad argument type
            FedbiomedDatasetError: cannot access folder
        """
        if not isinstance(modality, str):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Bad type for modality. "
                f"Expected str got {type(modality)}"
            )
        if isinstance(subject_or_folder, str):
            subject_or_folder = self._root.joinpath(subject_or_folder)
        elif not isinstance(subject_or_folder, Path):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Bad type for subject folder argument. "
                f"Expected str or Path got type({type(subject_or_folder)})"
            )

        modality_folders = set(
            self.apply_dlb(
                [modality],
                MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS,
                modality,
            )
        )
        try:
            subject_subfolders = set(
                [
                    x.name
                    for x in subject_or_folder.iterdir()
                    if x.is_dir() and not x.name.startswith(".")
                ]
            )
        except (FileNotFoundError, PermissionError, NotADirectoryError) as e:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Cannot access folders for subject "
                f"{subject_or_folder}. Error message is: {e}"
            )
        folder = modality_folders.intersection(subject_subfolders)

        if len(folder) == 0 or len(folder) > 1:
            return None
        return Path(folder.pop())

    def complete_subjects(
        self, subjects: List[str], modalities: List[str]
    ) -> List[str]:
        """Retrieves subjects that have given all the modalities.

        Args:
            subjects: List of subject folder names
            modalities: List of required modalities

        Returns:
            List of subject folder names that have required modalities
        """
        return [
            subject
            for subject in subjects
            if all(self.is_modalities_existing(subject, modalities))
        ]

    def subjects_with_imaging_data_folders(self) -> List[str]:
        """Retrieves subject folder names under Medical Folder root directory.

        Returns:
            subject folder names under Medical Folder root directory.
        """
        return [
            f.name
            for f in self._root.iterdir()
            if f.is_dir() and not f.name.startswith(".")
        ]

    def available_subjects(
        self,
        subjects_from_index: Union[list, pd.Series],
        subjects_from_folder: list = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Checks missing subject folders and missing entries in demographics

        Args:
            subjects_from_index: Given subject folder names in demographics
            subjects_from_folder: List of subject folder names to get intersection of given subject_from_index

        Returns:
            available_subjects: subjects that have an imaging data folder and are also present in the demographics file
            missing_subject_folders: subjects that are in the demographics file but do not have an imaging data folder
            missing_entries: subjects that have an imaging data folder but are not present in the demographics file
        """

        # Select all subject folders if it is not given
        if subjects_from_folder is None:
            subjects_from_folder = self.subjects_with_imaging_data_folders()

        # Missing subject that will cause warnings
        missing_subject_folders = list(
            set(subjects_from_index) - set(subjects_from_folder)
        )

        # Missing entries that will cause errors
        missing_entries = list(set(subjects_from_folder) - set(subjects_from_index))

        # Intersection
        available_subjects = list(
            set(subjects_from_index).intersection(set(subjects_from_folder))
        )

        return available_subjects, missing_subject_folders, missing_entries

    @staticmethod
    def read_demographics(path: Union[str, Path], index_col: Optional[int] = None):
        """Read demographics tabular file for Medical Folder dataset

        Raises:
            FedbiomedDatasetError: bad file format or a drectory
        """
        path = Path(path)
        if not path.is_file():
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Demographics should be , but got a folder {path.name}"
            )
        if path.suffix.lower() not in (
            ".csv",
            ".tsv",
        ):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Demographics should be CSV or TSV files"
            )

        # thank you [HalvarLilienthal](https://github.com/HalvarLilienthal) for this solution
        return pd.read_csv(
            path, index_col=index_col, dtype={index_col: str}, engine="python"
        )

    @staticmethod
    def demographics_column_names(path: Union[str, Path]):
        return MedicalFolderBase.read_demographics(path).columns.values

    @staticmethod
    def validate_MedicalFolder_root_folder(path: Union[str, Path]) -> Path:
        """Validates Medical Folder root directory by checking folder structure

        Args:
            path: path to root directory

        Returns:
            Path to root folder of Medical Folder dataset

        Raises:
            FedbiomedDatasetError: - If path is not an instance of `str` or `pathlib.Path`
                                   - If path is not a directory
        """
        if not isinstance(path, (Path, str)):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: The argument root should an instance of "
                f"`Path` or `str`, but got {type(path)}"
            )

        if not isinstance(path, Path):
            path = Path(path)

        path = Path(path).expanduser().resolve()

        if not path.exists():
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Folder or file {path} not found on system"
            )
        if not path.is_dir():
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Root for Medical Folder dataset "
                f"should be a directory."
            )

        directories = [f for f in path.iterdir() if f.is_dir()]
        if len(directories) == 0:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Root folder of Medical Folder should "
                f"contain subject folders, but no sub folder has been found. "
            )

        modalities = [f for f in path.glob("*/*") if f.is_dir()]
        if len(modalities) == 0:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value} Subject folders for Medical Folder should "
                f"contain modalities as folders. Folder structure should be "
                f"root/<subjects>/<modalities>"
            )

        return path

    @staticmethod
    def get_dataset_type() -> DatasetTypes:
        return DatasetTypes.MEDICAL_FOLDER


class MedicalFolderDataset(Dataset, MedicalFolderBase):
    """Torch dataset following the Medical Folder Structure.

    The Medical Folder structure is loosely inspired by the [BIDS standard](https://bids.neuroimaging.io/) [1].
    It should respect the following pattern:
    ```
    └─ MedicalFolder_root/
        └─ demographics.csv
        └─ sub-01/
            ├─ T1/
            │  └─ sub-01_xxx.nii.gz
            └─ T2/
                ├─ sub-01_xxx.nii.gz
    ```
    where the first-level subfolders or the root correspond to the subjects, and each subject's folder contains
    subfolders for each imaging modality. Images should be in Nifti format, with either the .nii or .nii.gz extensions.
    Finally, within the root folder there should also be a demographics file containing at least one index column
    with the names of the subject folders. This column will be used to explore the data and load the images. The
    demographics file may contain additional information about each subject and will be loaded alongside the images
    by our framework.

    [1] https://bids.neuroimaging.io/
    """

    ALLOWED_EXTENSIONS = [".nii", ".nii.gz"]

    def __init__(
        self,
        root: Union[str, PathLike, Path],
        data_modalities: Optional[Union[str, Iterable[str]]] = "T1",
        transform: Union[Callable, Dict[str, Callable]] = None,
        target_modalities: Optional[Union[str, Iterable[str]]] = "label",
        target_transform: Union[Callable, Dict[str, Callable]] = None,
        demographics_transform: Optional[Callable] = None,
        tabular_file: Union[str, PathLike, Path, None] = None,
        index_col: Union[int, str, None] = None,
    ):
        """Constructor for class `MedicalFolderDataset`.

        Args:
            root: Root folder containing all the subject directories.
            data_modalities (str, Iterable): Modality or modalities to be used as data sources.
            transform: A function or dict of function transform(s) that preprocess each data source.
            target_modalities: (str, Iterable): Modality or modalities to be used as target sources.
            target_transform: A function or dict of function transform(s) that preprocess each target source.
            demographics_transform: TODO
            tabular_file: Path to a CSV or Excel file containing the demographic information from the patients.
            index_col: Column name in the tabular file containing the subject ids which mush match the folder names.
        """
        super(MedicalFolderDataset, self).__init__(root=root)

        self._tabular_file = tabular_file
        self._index_col = index_col

        self._data_modalities = (
            [data_modalities] if isinstance(data_modalities, str) else data_modalities
        )
        self._target_modalities = (
            [target_modalities]
            if isinstance(target_modalities, str)
            else target_modalities
        )

        self._transform = self._check_and_reformat_transforms(
            transform, data_modalities
        )
        self._target_transform = self._check_and_reformat_transforms(
            target_transform, target_modalities
        )
        self._demographics_transform = (
            demographics_transform
            if demographics_transform is not None
            else lambda x: {}
        )

        # Image loader
        self._reader = Compose([LoadImage(ITKReader(), image_only=True), ToTensor()])

    def get_nontransformed_item(self, item):
        # For the first item retrieve complete subject folders
        subjects = self.subject_folders()

        if not subjects:
            # case where subjects is an empty list (subject folders have not been found)
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Cannot find complete subject folders with all the modalities"
            )
        # Get subject folder
        subject_folder = subjects[item]

        # Load data modalities
        data = self.load_images(subject_folder, modalities=self._data_modalities)

        # Load target modalities
        targets = self.load_images(subject_folder, modalities=self._target_modalities)

        # Demographics
        demographics = self._get_from_demographics(subject_id=subject_folder.name)
        return (data, demographics), targets

    def __getitem__(self, item):
        (data, demographics), targets = self.get_nontransformed_item(item)

        # Apply transforms to data elements
        if self._transform is not None:
            for modality, transform in self._transform.items():
                try:
                    data[modality] = transform(data[modality])
                except Exception as e:
                    raise FedbiomedDatasetError(
                        f"{ErrorNumbers.FB613.value}: Cannot apply transformation to modality `{modality}` in "
                        f"sample number {item} from dataset, error message is {e}."
                    )

        # Apply transforms to demographics elements
        if self._demographics_transform is not None:
            try:
                demographics = self._demographics_transform(demographics)
            except Exception as e:
                raise FedbiomedDatasetError(
                    f"{ErrorNumbers.FB613.value}: Cannot apply demographics transformation to "
                    f"sample number {item} from dataset. Error message: {repr(e)}. "
                    f"If the dataset was loaded without a demographics file, please ensure that the provided "
                    f"demographics transform immediately returns an empty dict when an empty dict is given as input."
                )

        # Try to convert demographics to tensor one last time
        if isinstance(demographics, dict) and len(demographics) == 0:
            demographics = torch.empty(
                0
            )  # handle case where demographics is an empty dict
        else:
            try:
                demographics = torch.as_tensor(demographics)
            except Exception as e:
                raise FedbiomedDatasetError(
                    f"{ErrorNumbers.FB310.value}: Could not convert demographics to torch Tensor. "
                    f"Please use demographics_transformation argument of BIDSDataset to convert "
                    f"the results manually or provide a data type that can be easily converted.\n"
                    f"Reason for failed conversion: {e}"
                )

        # Apply transform to target elements
        if self._target_transform is not None:
            for modality, target_transform in self._target_transform.items():
                try:
                    targets[modality] = target_transform(targets[modality])
                except Exception as e:
                    raise FedbiomedDatasetError(
                        f"{ErrorNumbers.FB613.value}: Cannot apply target transformation to modality `{modality}`"
                        f"in sample number {item} from dataset, error message is {e}."
                    )

        return (data, demographics), targets

    def __len__(self):
        """Length method to get number of samples

        Raises:
            FedbiomedDatasetError: If the dataset is empty.
        """

        subject_folders = self.subject_folders()
        length = len(subject_folders)

        if length <= 0:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Dataset cannot be empty. Check again that the "
                f"folder and the tabular data (if provided) exist and match properly."
            )
        return length

    @property
    def tabular_file(self):
        return self._tabular_file

    @property
    def index_col(self):
        """Getter/setter of the column containing folder's name (in the tabular file)"""
        return self._index_col

    @tabular_file.setter
    def tabular_file(self, value: Union[str, Path]) -> Union[str, Path]:
        """Sets `tabular_file` property

        Args:
            value: path to the tabular file

        Returns:
            path to the tabular file

        Raises:
            FedbiomedDatasetError:
        """
        if not isinstance(value, (str, Path)):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value} Path for tabular file should be of `str` or "
                f"`Path` type, but got {type(value)} "
            )

        path = Path(value)
        if not path.is_file():
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Path should be a data file"
            )

        self._tabular_file = Path(path).expanduser().resolve()
        return path

    @index_col.setter
    def index_col(self, value: int):
        """Sets `tabular_file` property.

        Args:
            value: Column index

        Raises:
            FedbiomedDatasetError: If value to set is not of `int` type
        """
        if not isinstance(value, (int, str)):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: `index_col` should be of `int` type, but "
                f"got {type(value)}"
            )

        self._index_col = value

    @property
    @cache
    def demographics(self) -> pd.DataFrame:
        """Loads tabular data file (supports excel, csv, tsv and colon separated value files)."""

        if self._tabular_file is None or self._index_col is None:
            # If there is no tabular file return empty data frame
            return None

        # Read demographics CSV
        try:
            demographics = self.read_demographics(self._tabular_file, self._index_col)
        except Exception as e:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Can not load demographics tabular file. "
                f"Error message is: {e}"
            )

        # Keep the first one in duplicated subjects
        return demographics.loc[~demographics.index.duplicated(keep="first")]

    @property
    def subjects_has_all_modalities(self):
        """Gets only the subjects that have all required modalities"""

        all_modalities = list(set(self._data_modalities + self._target_modalities))
        subject_folder_names = self.subjects_with_imaging_data_folders()

        # Get subject that has all requested modalities
        complete_subjects = self.complete_subjects(subject_folder_names, all_modalities)

        return complete_subjects

    @property
    @cache
    def subjects_registered_in_demographics(self):
        """Gets the subject only those who are present in the demographics file."""
        if self.demographics is None:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}:  demographics file and `index_col` arguments"
                " hasnot been sepcified"
            )
        complete_subject_folders, *_ = self.available_subjects(
            subjects_from_folder=self.subjects_has_all_modalities,
            subjects_from_index=self.demographics.index,
        )

        return complete_subject_folders

    def set_dataset_parameters(self, parameters: dict):
        """Sets dataset parameters.

        Args:
            parameters: Parameters to initialize

        Raises:
            FedbiomedDatasetError: If given parameters are not of `dict` type
        """
        if not isinstance(parameters, dict):
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Expected type for `parameters` is `dict, "
                f"but got {type(parameters)}`"
            )

        for key, value in parameters.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise FedbiomedDatasetError(
                    f"{ErrorNumbers.FB613.value}: Trying to set non existing attribute '{key}'"
                )

    def load_images(
        self, subject_folder: Path, modalities: list
    ) -> Dict[str, torch.Tensor]:
        """Loads modality images in given subject folder

        Args:
            subject_folder: Subject folder where modalities are stored
            modalities: List of available modalities

        Returns:
            Subject image data as victories where keys represent each modality.
        """
        # FIXME: improvment suggestion of this function at #1279

        subject_data = {}

        for modality in modalities:
            modality_folder = self._subject_modality_folder(subject_folder, modality)
            image_folder = subject_folder.joinpath(modality_folder)
            nii_files = [p.resolve() for p in image_folder.glob("**/*")]

            # Load the first, we assume there is going to be a single image per modality for now.

            nii_files = tuple(
                img
                for img in nii_files
                if any(str(img).endswith(fmt) for fmt in self.ALLOWED_EXTENSIONS)
            )
            if len(nii_files) < 1:
                raise FedbiomedDatasetError(
                    f"{ErrorNumbers.FB613.value}: folder {os.path.join(image_folder, modality)}"
                    " is empty, but should contain an niftii image. Aborting"
                )

            elif len(nii_files) > 1:
                raise FedbiomedDatasetError(
                    f"{ErrorNumbers.FB613.value}: more than one niftii file has been detected"
                    " {', '.join(tuple(str(f) for f in nii_files))}. "
                    "\nThere should be only one niftii image per modality. Aborting"
                )

            img_path = nii_files[0]
            img = self._reader(img_path)
            subject_data[modality] = img

        return subject_data

    def subject_folders(self) -> List[Path]:
        """Retrieves subject folder names of only those who have their complete modalities

        Returns:
            List of subject directories that has all requested modalities
        """

        # If demographics are present
        if self._tabular_file and self._index_col is not None:
            complete_subject_folders = self.subjects_registered_in_demographics
        else:
            complete_subject_folders = self.subjects_has_all_modalities

        return [self._root.joinpath(folder) for folder in complete_subject_folders]

    def shape(self) -> dict:
        """Retrieves shape information for modalities and demographics csv"""

        # Get all modalities
        data_modalities = list(set(self._data_modalities))
        target_modalities = list(set(self._target_modalities))
        modalities = list(set(self._data_modalities + self._target_modalities))
        (image, _), targets = self.get_nontransformed_item(0)

        result = {modality: list(image[modality].shape) for modality in data_modalities}

        result.update(
            {modality: list(targets[modality].shape) for modality in target_modalities}
        )
        num_modalities = len(modalities)
        demographics_shape = (
            self.demographics.shape if self.demographics is not None else None
        )
        result.update(
            {"demographics": demographics_shape, "num_modalities": num_modalities}
        )

        return result

    def _get_from_demographics(self, subject_id):
        """Extracts subject information from a particular subject in the form of a dictionary."""

        if self._tabular_file and self._index_col is not None:
            demographics = self.demographics.loc[subject_id].to_dict()

            # Extract only compatible types for torch
            # TODO Decide what to do with missing variables
            return {
                key: val
                for key, val in demographics.items()
                if isinstance(val, (int, float, str, bool))
            }
        else:
            return {}

    @staticmethod
    def _check_and_reformat_transforms(
        transform: Union[Callable, Dict[str, Callable]],
        modalities: Union[str, Iterable[str]],
    ) -> Dict[str, Callable]:
        """Checks and formats transforms into a dictionary of transforms.

        Args:
            transform: Function or dictionary of functions for preprocessing data.
            modalities: Modalities to be considered.

        Returns:
            A dict of transforms compatible with the provided modalities.

        Raises:
            FedbiomedDatasetError: sample uses unknown modality
            FedbiomedDatasetError: transform method must be callable
            FedbiomedDatasetError: bad type for parameters
        """

        # Return None if any transform is not provided
        if transform is None:
            return None

        # Convert str type modality to list
        if isinstance(modalities, str):
            modalities = [modalities]

        # If transform is dict, map modalities to transforms
        if isinstance(transform, dict):
            # Raise if transform objects are not provided as dictionaries.
            # E.g. {'T1': Normalize(...), 'T2': ToTensor()}
            for modality, method in transform.items():
                if modality not in modalities:
                    raise FedbiomedDatasetError(
                        f"{ErrorNumbers.FB613.value}: Modality `{modality}` is not present "
                        f"in {modalities}"
                    )

                if not callable(method):
                    raise FedbiomedDatasetError(
                        f"{ErrorNumbers.FB613.value} Transform method/function for "
                        f"`{modality}` should be callable"
                    )

            return transform

        # If transform is not dict and there is only one modality
        elif len(modalities) == 1:
            if not callable(transform):
                raise FedbiomedDatasetError(
                    f"{ErrorNumbers.FB613.value}: Transform method/function for "
                    f"`{modalities[0]}` should be callable"
                )

            return {modalities[0]: transform}

        # Raise ------
        else:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: As you have multiple data modalities, transforms "
                f"have to be a dictionary using the modality keys: {modalities}"
            )


class MedicalFolderController(MedicalFolderBase):
    """Utility class to construct and verify Medical Folder datasets without knowledge of the experiment.

    The purpose of this class is to enable key functionalities related to the MedicalFolderDataset at the time of
    dataset deployment, i.e. when the data is being added to the node's database.

    Specifically, the MedicalFolderController class can be used to:
    - construct a MedicalFolderDataset with all available data modalities, without knowing which ones will be used as
        targets or features during an experiment
    - validate that the proper folder structure has been respected by the data managers preparing the data
    - identify which subjects have which modalities
    """

    def __init__(self, root: str = None):
        """Constructs MedicalFolderController

        Args:
            root: Folder path to dataset. Defaults to None.
        """
        super(MedicalFolderController, self).__init__(root=root)

    # TODO: suppress `check_modalities` ? (currently unused)
    # TODO: `check_modalities` looks bugged
    #   - `len(unique_modalities) == len(modalities)` doesn't test whether "subject
    #     folders contains at least one common modality" ???
    #   - `self.modalities()[1]` needs to be different for this purpose: return a list of list
    #     [['label', 'T1'], ['label', 'T1', 'T2']] + check a modality exists in all sub-lists ?

    # def check_modalities(self, _raise: bool = True) -> Tuple[bool, str]:
    #    """Checks whether subject folders contains at least one common modality
    #
    #    Args:
    #        _raise: Flag to indicate whether function should raise in case of error. If `False` returns
    #            tuple contains respectively `False` and error message
    #
    #    Returns:
    #        status: True, if folders contain at least one common modality
    #        message: Error message if folder do not contain at least one common modality. If they do, error message
    #            will be empty string
    #
    #    Raises:
    #        FedbiomedDatasetError:
    #    """
    #    unique_modalities, modalities = self.modalities()
    #    if len(unique_modalities) == len(modalities):
    #        message = f"{ErrorNumbers.FB613.value}: Subject folders in Medical Folder root folder does not contain" \
    #                  f"any common modalities. At least one common modality is expected."
    #        if _raise:
    #            raise FedbiomedDatasetError(message)
    #        else:
    #            return False, message
    #
    #    return True, ""

    def subject_modality_status(self, index: Union[List, pd.Series] = None) -> Dict:
        """Scans subjects and checks which modalities are existing for each subject

        Args:
            index: Array-like index that comes from reference csv file of Medical Folder dataset. It represents subject
                folder names. Defaults to None.
        Returns:
            Modality status for each subject that indicates which modalities are available
        """

        modalities, _ = self.modalities()
        subjects = self.subjects_with_imaging_data_folders()
        modality_status = {"columns": [*modalities], "data": [], "index": []}

        if index is not None:
            _, missing_subjects, missing_entries = self.available_subjects(
                subjects_from_index=index
            )
            modality_status["columns"].extend(["in_folder", "in_index"])

        for subject in subjects:
            modality_report = self.is_modalities_existing(subject, modalities)
            status_list = [status for status in modality_report]
            if index is not None:
                status_list.append(False if subject in missing_subjects else True)
                status_list.append(False if subject in missing_entries else True)

            modality_status["data"].append(status_list)
            modality_status["index"].append(subject)

        return modality_status

    def load_MedicalFolder(
        self, tabular_file: Union[str, Path] = None, index_col: Union[str, int] = None
    ) -> MedicalFolderDataset:
        """Load Medical Folder dataset with given tabular_file and index_col

        Args:
            tabular_file: File path to demographics data set
            index_col: Column index that represents subject folder names

        Returns:
            MedicalFolderDataset object

        Raises:
            FedbiomedDatasetError: If Medical Folder dataset is not successfully loaded
        """
        if self._root is None:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Can not load Medical Folder dataset without "
                f"declaring root directory. Please set root or build MedicalFolderController "
                f"with by providing `root` argument use"
            )

        modalities, _ = self.modalities()

        try:
            dataset = MedicalFolderDataset(
                root=self._root,
                tabular_file=tabular_file,
                index_col=index_col,
                data_modalities=modalities,
                target_modalities=modalities,
            )
        except FedbiomedError as e:
            raise FedbiomedDatasetError(
                f"{ErrorNumbers.FB613.value}: Can not create Medical Folder dataset. {e}"
            )

        if self._dlp is not None:
            dataset.set_dlp(self._dlp)
        return dataset

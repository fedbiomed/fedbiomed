from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd

from fedbiomed.common.constants import DataLoadingBlockTypes, DatasetTypes, ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan, DataLoadingPlanMixin
from fedbiomed.common.dataset_reader import CsvReader, NiftiReader
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._controller import Controller


class MedicalFolderLoadingBlockTypes(DataLoadingBlockTypes, Enum):
    MODALITIES_TO_FOLDERS: str = "modalities_to_folders"


class MedicalFolderController(Controller):
    _extensions: tuple[str, ...] = (".nii", ".nii.gz")

    """MedicalFolder where data is arranged like this:
    root
    ├── sub-01
    │   ├── modality-a
    │   │   └── sub-01_xxx.nii.gz
    │   ├── modality-b
    │   │   └── sub-01_xxx.nii.gz
    │   └── ...
    ├── sub-02
    │   ├── modality-a
    │   │   └── sub-02_xxx.nii.gz
    │   └── ...
    ├── ...
    └── demographics.csv
    """

    def __init__(
        self,
        root: Union[str, Path],
        tabular_file: Optional[Union[str, PathLike, Path]] = None,
        index_col: Optional[Union[int, str]] = None,
    ):
        """Constructor for class `MedicalFolder`

        Args:
            root: Root directory path
            tabular_file: Path to CSV file containing the demographic information
            index_col: Column name in tabular file containing the subjects names

        Raises:
            FedbiomedError:
            - if one in `tabular_file` and `index_col` is given and the other is not
        """
        DataLoadingPlanMixin.__init__(self)
        self.root = root
        self.tabular_file = tabular_file
        self.index_col = index_col

        # Folder structure <subject>/<modality>/<file> in DataFrame format
        self._df_dir = self._make_df_dir(root=self.root, extensions=self._extensions)

        # Demographics
        if (tabular_file is None) != (index_col is None):
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: "
                "Arguments `tabular_file` and `index_col`, both or none are expected"
            )
        self._demographics = (
            None
            if tabular_file is None
            else self.read_demographics(tabular_file, index_col)
        )

        # Filter subjects to contain all modalities
        self._modalities, filtered_df_dir = (
            self.filter_df_dir_subjects_with_all_modalities(
                df_dir=self._df_dir,
            )
        )

        # Generate list of samples: dict, with demographics and path to modalities
        self._subjects, self._samples = self._make_dataset(
            root=self.root,
            df_dir=filtered_df_dir,
            demographics=self.demographics,
        )

        # Check if is possible to use `reader` to recover a valid item
        _ = self._get_nontransformed_item(0)

        self._controller_kwargs = {
            "root": str(self.root),
            "tabular_file": str(self.tabular_file),
            "index_col": self.index_col,
        }

    # === Properties ===
    @property
    def tabular_file(self):
        return self._tabular_file

    @tabular_file.setter
    def tabular_file(self, filepath: Optional[Union[str, Path]]):
        """Sets `tabular_file` property"""
        if filepath is not None:
            filepath = self._validate_tabular_file(filepath)
        self._tabular_file = filepath

    @staticmethod
    def _validate_tabular_file(filepath: Union[str, Path]) -> Path:
        """Validates `tabular_file` property

        Raises:
            FedbiomedError:
            - if filepath is not of type `str` or `Path`
            - if filepath does not match a file or is not csv or tsv
        """
        if not isinstance(filepath, (str, Path)):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected a string or Path, got "
                f"{type(filepath).__name__}"
            )
        filepath = Path(filepath).expanduser().resolve()
        if not filepath.is_file() and not filepath.suffix.lower().endswith(
            (".csv", ".tsv")
        ):
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: "
                "Path does not correspond to a CSV or TSV file"
            )
        return filepath

    @property
    def index_col(self):
        return self._index_col

    @index_col.setter
    def index_col(self, value: Optional[Union[int, str]]):
        """Sets `index_col` property

        Raises:
            FedbiomedError: if value is not of type `int` or `str`
        """
        if value is not None:
            if not isinstance(value, (int, str)):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB613.value}: `index_col` should be of type "
                    f"`int` or `str`, but got {type(value).__name__}"
                )
        self._index_col = value

    @property
    def demographics(self):
        return self._demographics

    @property
    def modalities(self):
        """When a dlp is given `self._modalities` is a dict (modalities_to_folders)"""
        if isinstance(self._modalities, dict):
            return list(self._modalities.keys())
        return self._modalities

    @staticmethod
    def _normalize_modalities(modalities: Union[str, Iterable[str]]) -> set[str]:
        """Validates `modalities` and returns it as `set`

        Returns:
            `modalities` in type `set`

        Raises:
            FedbiomedError: If the input does not math the types expected
        """
        if isinstance(modalities, str):
            return {modalities}
        if (
            not isinstance(modalities, dict)
            and isinstance(modalities, Iterable)
            and all(isinstance(item, str) for item in modalities)
        ):
            return set(modalities)
        raise FedbiomedError(
            f"{ErrorNumbers.FB613.value}: "
            "Bad type for modalities. Expected str or Iterable[str], got "
            f"{type(modalities).__name__}"
        )

    @property
    def subjects(self):
        return self._subjects

    # === Functions ===
    @staticmethod
    def read_demographics(
        tabular_file: Union[str, Path],
        index_col: Optional[Union[int, str]] = None,
    ) -> pd.DataFrame:
        """Read demographics tabular file

        Args:
            tabular_file: path to demographics file
            index_col: Index column that matches <subject>. Defaults to None.

        Raises:
            FedbiomedError: if the file can not be loaded

        Returns:
            Demographics in DataFrame format
        """
        tabular_file = MedicalFolderController._validate_tabular_file(tabular_file)

        try:
            demographics = CsvReader(tabular_file).data.to_pandas()
            if index_col is not None:
                if isinstance(index_col, int):
                    if index_col < 0 or index_col >= len(demographics.columns):
                        raise FedbiomedError(
                            f"{ErrorNumbers.FB613.value}: "
                            f"Index column {index_col} is out of bounds"
                        )
                    index_col = demographics.columns[index_col]
                demographics = demographics.set_index(index_col)

        except FedbiomedError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: :"
                f"Can not load demographics tabular file. Error message is: {e}"
            ) from e

        length = len(demographics)
        logger.info(f"Number of rows in demographics file: {length}")

        # Keep the first one in duplicated subjects
        demographics = demographics.loc[~demographics.index.duplicated(keep="first")]
        if length != len(demographics):
            logger.info(f"Length of demographics for unique index {len(demographics)}")

        return demographics

    def demographics_column_names(self, path: Union[str, Path]):
        return self.read_demographics(path).columns.values

    @staticmethod
    def _make_df_dir(root: Path, extensions: tuple[str, ...]) -> pd.DataFrame:
        """Match files for expected structure.
        Filter files by extension and avoid hidden folders and files.

        Args:
            root: path to medical folder
            extensions: to identify valid files (lowercase)

        Raises:
            FedbiomedError:
            - if folder structure does not match root/<subjects>/<modalities>/<files>
            - if no file is identified as valid in the folder structure
            - if more than one file is identified as valid per <subject>/<modality>

        Returns:
            DataFrame with columns ('subject', 'modality', 'file') for matches in root
        """
        rows = []
        for path in root.rglob("*/*/*"):
            if path.is_file():
                subject, modality, file = path.relative_to(root).parts
                # set of conditions to validate that a file is valid
                if file.lower().endswith(extensions) and not any(
                    part.startswith((".", "_")) for part in (subject, modality, file)
                ):
                    rows.append(
                        {"subject": subject, "modality": modality, "file": file}
                    )

        if not rows:
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: Root folder does not match "
                "Medical Folder structure root/<subjects>/<modalities>/<files> "
                f"for files with extensions in {extensions}. "
                "Hidden files or folders are not considered"
            )

        df_dir = pd.DataFrame(rows)

        # === Ensure one valid file per modality folder
        _files_count = (
            df_dir.groupby(["subject", "modality"])["file"]
            .count()
            .reset_index(name="count")
        )

        _multiple_files = _files_count[_files_count["count"] != 1]
        if not _multiple_files.empty:
            _folders = ", ".join(
                f"{row.subject}/{row.modality}"
                for _, row in _multiple_files.head().iterrows()
            )
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: more than one valid file per modality "
                f"has been found for next <subject>/<modality>: {_folders}, ..."
            )

        return df_dir

    @staticmethod
    def filter_df_dir_subjects_with_all_modalities(
        df_dir: pd.DataFrame, modalities: Optional[Union[str, Iterable[str]]] = None
    ) -> tuple[list[str], pd.DataFrame]:
        """Filter DataFrame that represents directory: every 'subject' has to
        have all `modalities`. When `modalities` is not given, all 'modality' values
        in DataFrame are required for each 'subject'.

        Args:
            df_dir: Contains matches in folder structure for <subject><modality><file>
            modalities: Optional input to specify modalities

        Raises:
            FedbiomedError:
            - if not all `modalities` were found in the folder structure
            - if no 'subject' has all `modalities`

        Returns:
            tuple composed of modalities and filtered df_dir
        """
        # === Identify modalities (-available-, and -missing- when `modalities` is not None)
        if modalities is None:
            modalities = set(df_dir["modality"].unique())
        else:
            modalities = MedicalFolderController._normalize_modalities(modalities)
            missing_modalities = modalities.difference(df_dir["modality"].unique())
            if missing_modalities:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB613.value}: Some modality names are not found in "
                    f"the root folder structure: {', '.join(missing_modalities)}"
                )

        # === Identify subjects that have all modalities
        _group_modality_sets = df_dir.groupby("subject")["modality"].apply(set)
        _matching_subject = _group_modality_sets[
            _group_modality_sets.apply(lambda x: modalities.issubset(x))
        ].index
        # === Filter df by 'subjects with all modalities' and 'modalities'
        filtered_df_dir = df_dir[
            df_dir["subject"].isin(_matching_subject)
            & df_dir["modality"].isin(modalities)
        ]

        if df_dir.empty:
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: "
                f"No 'subject' matches all `modalities`: {', '.join(modalities)}"
            )
        logger.info(
            f"{len(df_dir['subject'].unique())} subjects in folder structure with all "
            f"modalities: {', '.join(modalities)}"
        )

        return list(modalities), filtered_df_dir

    @staticmethod
    def _make_dataset(
        root: Path,
        demographics: Optional[pd.DataFrame],
        df_dir: pd.DataFrame,
    ) -> tuple[list[str], list[str], list[Dict[str, Any]]]:
        """Builds samples as `dict` with `modalities` and `demographics`

        Args:
            df_dir: filtered DataFrame where all subjects contain all modalities
            modalities: list of modalities already validated alongside with df_dir
            tabular_file: path to demographics file
            index_col: Index column that matches <subject>

        Raises:
            FedbiomedError:
            - if one of `tabular_file` and `index_col` is given and the other is not
            - if empty intersection between subjects from folders and demographics

        Returns:
            `subjects` and `samples`
        """
        samples = []
        subject_groups = dict(tuple(df_dir.groupby("subject")))
        subjects = set(subject_groups.keys())

        if demographics is not None:
            subjects = subjects.intersection(demographics.index.values)
            if not subjects:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Selected column from demographics as "
                    "subject reference does not match any subject in folder structure"
                )

        for subject in subjects:
            sample = (
                {}
                if demographics is None
                else {"demographics": demographics.loc[subject].to_dict()}
            )
            for _, row in subject_groups[subject].iterrows():
                sample[row["modality"]] = str(
                    root / row["subject"] / row["modality"] / row["file"]
                )
            samples.append(sample)

        logger.info(f"{len(samples)} complete samples successfully identified")
        return list(subjects), samples

    def subject_modality_status(self, index: Union[list, pd.Series] = None) -> Dict:
        """Scans subjects and checks which modalities exist for each subject

        Args:
            index: Array-like index that comes from reference csv file.
                It represents subject folder names. Defaults to None.
        Returns:
            Modality status that indicates which modalities are available per subject
        """
        # Pivot into wide format with boolean indicators
        df_ = (
            self._df_dir.assign(val=True)
            .pivot_table(
                index="subject",
                columns="modality",
                values="val",
                fill_value=False,
            )
            .astype(bool)
        )

        if index is not None:
            df_["in_folder"] = True
            # Merge with pivot (outer join)
            df_ = pd.merge(
                df_,
                pd.DataFrame(True, index=index, columns=["in_index"]),
                left_index=True,
                right_index=True,
                how="outer",
            )
            # Fill missing values with False
            fill_cols = ["in_folder", "in_index", *self.modalities]
            df_[fill_cols] = df_[fill_cols].fillna(False)

        return {
            "columns": df_.columns.tolist(),
            "data": df_.values.tolist(),
            "index": df_.index.tolist(),
        }

    def available_subjects(
        self,
        subjects_from_index: Union[list, pd.Series],
        subjects_from_folder: list = None,
    ) -> Dict[str, str]:
        """Checks missing subject folders and missing entries in demographics

        Args:
            subjects_from_index: Given subject folder names in demographics
            subjects_from_folder: list of subject folder names

        Returns:
            Dict with next keys:
            - missing_folders: subjects in demographics absent in folder structure
            - missing_entries: subjects in folder structure absent in demographics
            - intersection: subjects present in folder structure and demographics
        """
        # Select all subject folders if it is not given
        if subjects_from_folder is None:
            subjects_from_folder = self.subjects

        return {
            # Missing subject that will cause warnings
            "missing_folders": list(
                set(subjects_from_index).difference(subjects_from_folder)
            ),
            # Missing entries that will cause errors
            "missing_entries": list(
                set(subjects_from_folder).difference(subjects_from_index)
            ),
            # Intersection
            "intersection": list(
                set(subjects_from_index).intersection(set(subjects_from_folder))
            ),
        }

    def set_dlp(self, dlp: DataLoadingPlan) -> None:
        DataLoadingPlanMixin.set_dlp(self, dlp)
        if MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS in self._dlp:
            self._modalities = self._dlp[
                MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS
            ].map

            # Filter subjects to contain all modalities
            _, filtered_df_dir = self.filter_df_dir_subjects_with_all_modalities(
                df_dir=self._df_dir,
                modalities=list(self._modalities.values()),
            )

            # Generate list of samples: dict, with demographics and path to modalities
            self._subjects, self._samples = self._make_dataset(
                root=self.root,
                df_dir=filtered_df_dir,
                demographics=self.demographics,
            )

    def _get_nontransformed_item(
        self, index: int
    ) -> Dict[str, NiftiReader.data_type | Dict[str, Any]]:
        """Retrieve a data sample without applying transforms"""
        sample = self._samples[index]
        try:
            if isinstance(self._modalities, dict):
                data = {
                    modality: NiftiReader.read(sample[folder])
                    for modality, folder in self._modalities.items()
                }
            else:
                data = {
                    modality: NiftiReader.read(sample[modality])
                    for modality in self.modalities
                }
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item at index {index}"
            ) from e
        if "demographics" in sample:
            data["demographics"] = sample["demographics"]
        return data

    def __len__(self):
        return len(self._samples)

    @staticmethod
    def get_dataset_type() -> DatasetTypes:
        return DatasetTypes.MEDICAL_FOLDER

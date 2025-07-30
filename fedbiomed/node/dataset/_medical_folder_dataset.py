from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

import pandas as pd
from monai.data import ITKReader
from monai.transforms import Compose, LoadImage, ToTensor

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._dataset import Dataset


class MedicalFolderDataset(Dataset):
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

    # CONSTANTS ===============================================================
    EXTENSIONS = (".nii", ".nii.gz")
    # DEFAULT PROPERTIES VALUES ===============================================
    _reader: Callable[[str], Any] = Compose(
        [LoadImage(ITKReader(), image_only=True), ToTensor()]
    )

    # =========================================================================
    def __init__(
        self,
        root: Union[str, Path],
        tabular_file: Optional[Union[str, PathLike, Path]] = None,
        index_col: Optional[Union[int, str]] = None,
        data_modalities: Optional[Union[str, Iterable[str]]] = None,
        target_modalities: Optional[Union[str, Iterable[str]]] = None,
        transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
        target_transform: Optional[Union[Callable, Dict[str, Callable]]] = None,
        demographics_transform: Optional[Callable] = None,
    ):
        """Constructor for class `MedicalFolderDataset`.

        Args:
            transform: A function or dict of function transform(s) that preprocess each data source.
            target_transform: A function or dict of function transform(s) that preprocess each target source.
            demographics_transform: TODO
            tabular_file: Path to a CSV or Excel file containing the demographic information from the patients.
            index_col: Column name in the tabular file containing the subject ids which mush match the folder names.
        """
        self.root = root
        self.data_modalities = data_modalities
        self.target_modalities = target_modalities
        self.tabular_file = tabular_file
        self.index_col = index_col
        self.transform = transform

        self._samples = self._make_dataset(
            root=self.root,
            modalities=self.modalities,
            tabular_file=self.tabular_file,
            index_col=self.index_col,
        )

    """
    def __init__(
        self,
        framework_transform: Transform = None,
        framework_target_transform: Transform = None,
        generic_transform: Transform = None,
        generic_target_transform: Transform = None,
        # Keep actual names for backward compatibility
        #
        # reader_images_transform : Transform = None,
        # reader_images_target_transform : Transform = None,
        # reader_demographics_transform : Transform = None.
        transform: Transform = None,
        target_transform: Transform = None,
        demographics_transform: Transform = None,
        # Keep actual names for backward compatibility
    ) -> None:
    """

    # =========================================================================
    @property
    def tabular_file(self):
        return self._tabular_file

    @tabular_file.setter
    def tabular_file(self, filepath: Optional[Union[str, Path]]):
        """Sets `tabular_file` property"""
        if filepath is not None:
            filepath = self._validate_tabular_file(filepath)
        self._tabular_file = filepath

    def _validate_tabular_file(self, filepath: Union[str, Path]) -> Path:
        """Validates `tabular_file` property

        Raises:
            FedbiomedError:
            - if filepath is not of typr str or Path
            - if filepath does not match a file or is not csv or tsv
        """
        if not isinstance(filepath, (str, Path)):
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: Expected a string or Path, got "
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
        """Sets `index_col` property.

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
        if self.tabular_file is None or self.index_col is None:
            return None
        return self.read_demographics(self.tabular_file, self.index_col)

    @property
    def data_modalities(self):
        return self._data_modalities

    @data_modalities.setter
    def data_modalities(self, modalities: Optional[Union[str, Iterable[str]]]) -> set:
        self._data_modalities = (
            set() if modalities is None else self._validate_modalities(modalities)
        )

    @property
    def target_modalities(self):
        return self._target_modalities

    @target_modalities.setter
    def target_modalities(self, modalities: Optional[Union[str, Iterable[str]]]) -> set:
        self._target_modalities = (
            set() if modalities is None else self._validate_modalities(modalities)
        )

    def _validate_modalities(self, modalities: Union[str, Iterable[str]]) -> set:
        """Validates `modalities`

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
            "Bad type for modalities. Expected str or Iterable[str]"
        )

    @property
    def modalities(self):
        return self._data_modalities.union(self._target_modalities)

    @property
    def reader(self):
        return (
            self._reader
            if isinstance(self._reader, dict)
            else {modality: self._reader for modality in self.modalities}
        )

    # =========================================================================
    def set_dataset_parameters(self, parameters: dict):
        """Sets dataset parameters.

        Args:
            parameters: Parameters to initialize

        Raises:
            FedbiomedError:
            - If given parameters are not of `dict` type
            - If keys of `dict` are not all of type `str`
            - If the attribute does not exist ()
        """
        if not isinstance(parameters, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: Expected type for `parameters` is `dict, "
                f"but got {type(parameters)}`"
            )

        if not all(isinstance(_k, str) for _k in parameters.keys()):
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: "
                "Expected type for all keys in `parameters` is `str`"
            )

        for key, value in parameters.items():
            if not key.startswith("_") and hasattr(self, key):
                setattr(self, key, value)
            else:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB613.value}: "
                    "Trying to set non existing attribute '{key}'"
                )

        """Careful Rebuild
        self._modalities, self._df_dir = self._make_df_dir(
            root=self.root,
            modalities=self._data_modalities.union(self._target_modalities),
        )

        if self._tabular_file is not None and self._index_col is not None:
            self._demographics = self.read_demographics(
                tabular_file=self._tabular_file, index_col=self._index_col
            )
        """

    def read_demographics(
        self,
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
        tabular_file = self._validate_tabular_file(tabular_file)
        try:
            demographics = pd.read_csv(
                tabular_file, index_col=index_col, engine="python"
            )
        except Exception as e:
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

    def _make_df_dir(
        self,
        root: Path,
        extensions: Union[str, tuple[str, ...]] = EXTENSIONS,
        modalities: Optional[Union[str, Iterable[str]]] = None,
    ) -> pd.DataFrame:
        """Matches files for expected structure.
        Filters files by extension and avoid hidden folders and files.

        Args:
            root: path to medical folder
            extensions: extensions to consider (lowercase)

        Raises:
            FedbiomedError:
            - if folder structure does not match root/<subjects>/<modalities>/<files>
            - if no file is identified as valid in the folder structure
            - if more than one file is identified as valid per <subject>/<modality>
            - if not all `modalities` are found in the folder structure
            - if no <subject> has all `modalities`

        Returns:
            DataFrame with columns ('subject', 'modality', 'file') for matches in root
        """
        extensions = extensions if isinstance(extensions, str) else tuple(extensions)
        rows = []
        _no_structure_match_flag = True
        for path in root.rglob("*/*/*"):
            if path.is_file():
                _no_structure_match_flag = False
                parts = path.relative_to(root).parts
                # set of conditions to validate that a file is valid
                if parts[-1].lower().endswith(extensions) and not any(
                    part.startswith((".", "_")) for part in parts
                ):
                    rows.append(
                        dict(zip(("subject", "modality", "file"), parts, strict=True))
                    )
        # =====================================================================
        if _no_structure_match_flag:
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: Root folder does not match Medical "
                "Folder structure: root/<subjects>/<modalities>/<files>"
            )
        if not rows:
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: "
                "No match found in root/<subjects>/<modalities>/<files> for files with "
                f"extensions in {extensions}. Hidden files or folders are not considered"
            )
        # =====================================================================
        df_dir = pd.DataFrame(rows)
        # =====================================================================
        _files_count = df_dir.groupby(["subject", "modality"])["file"].count()
        _files_count = _files_count.reset_index(name="count")
        _multiple_files = _files_count[_files_count["count"] != 1]
        if len(_multiple_files) != 0:
            _folders = ", ".join(
                [
                    "/".join([_row["subject"], _row["modality"]])
                    for _, _row in _multiple_files.iterrows()
                ][:3]  # max n to display
            )
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: more than one valid file per modality has "
                f"been found for next <subject>/<modality>: {_folders}, ..."
            )
        # =====================================================================
        candidate_modalities = set(df_dir["modality"].unique())
        modalities = (
            candidate_modalities
            if not modalities
            else self._validate_modalities(modalities)
        )
        missing_modalities = modalities.difference(candidate_modalities)
        if missing_modalities:
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: Some modality names are not "
                f"found in the root folder structure: {missing_modalities}"
            )
        # subjects that have all given modalities
        _group_modality_sets = df_dir.groupby("subject")["modality"].apply(set)
        _matching_subject = _group_modality_sets[
            _group_modality_sets.apply(lambda x: modalities.issubset(x))
        ].index
        # filter df by 'subjects with all modalities' and 'modalities'
        df_dir = df_dir[
            df_dir["subject"].isin(_matching_subject)
            & df_dir["modality"].isin(modalities)
        ]
        # =====================================================================
        if len(df_dir) == 0:
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: "
                f"No 'subject' matches all `modalities`: {modalities}"
            )
        # =====================================================================
        logger.info(
            f"{len(df_dir['subject'].unique())} subjects in folder structure with all "
            f"modalities: {', '.join(modalities)}"
        )
        return df_dir

    def _make_dataset(
        self,
        root: Path,
        modalities: set[str],
        tabular_file: Optional[Path],
        index_col: Optional[Union[int, str]],
    ):
        if (tabular_file is None) != (index_col is None):
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: "
                "Arguments `tabular_file` and `index_col`, both or none are expected"
            )

        samples = []
        df_dir = self._make_df_dir(root=root, modalities=modalities)
        subject_groups = dict(tuple(df_dir.groupby("subject")))
        subjects = set(subject_groups.keys())

        if tabular_file is not None:
            demographics = self.read_demographics(tabular_file, index_col)
            subjects = subjects.intersection(demographics.index.values)

        for subject in subjects:
            # TODO: Order of colums should be ensured ???
            sample = (
                {}
                if tabular_file is None
                else {"demographics": demographics.loc[subject].to_dict()}
            )
            for _, row in subject_groups[subject].iterrows():
                sample[row["modality"]] = str(
                    root / row["subject"] / row["modality"] / row["file"]
                )
            samples.append(sample)

        logger.info(f"{len(samples)} complete samples succesfully identified")
        return samples

    @staticmethod
    def _build_item(
        sample: Dict[str, Any],
        modalities: set[str],
        transform: Dict[str, Callable],
    ) -> Any:
        return {
            modality: transform[modality](sample[modality]) for modality in modalities
        }

    def _get_nontransformed_item(self, idx: int) -> tuple[Any, Any]:
        sample = self._samples[idx]
        try:
            demographics = sample["demographics"] if "demographics" in sample else None
            data = self._build_item(
                sample=sample,
                modalities=self.data_modalities,
                transform=self.reader,
            )
            target = self._build_item(
                sample=sample,
                modalities=self.target_modalities,
                transform=self.reader,
            )
        except Exception as e:
            raise FedbiomedError("msg") from e
        return (data, demographics), target

    def __getitem__(self, idx):
        pass

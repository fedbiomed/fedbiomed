from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from monai.data import ITKReader
from monai.transforms import LoadImage

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_reader import CsvReader
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._controller import Controller


class MedicalFolderController(Controller):
    _extensions: Tuple[str, ...] = (".nii", ".nii.gz")
    _reader: Callable[[str], Any] = LoadImage(ITKReader(), image_only=True)

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
        modalities: Optional[Union[str, Iterable[str]]] = None,
    ):
        """Constructor for class `MedicalFolder`

        Args:
            root: Root directory path
            tabular_file: Path to CSV file containing the demographic information
            index_col: Column name in tabular file containing the subjects names
        """
        self.root = root
        self.tabular_file = tabular_file
        self.index_col = index_col

        self._modalities, self._samples = self._make_dataset(
            root=self.root,
            tabular_file=self.tabular_file,
            index_col=self.index_col,
            modalities=modalities,
        )
        # Check if is possible to use `reader` to recover a valid item
        _ = self._get_nontransformed_item(0)

        self._controller_kwargs = {
            "root": str(self.root),
            "tabular_file": str(self.tabular_file),
            "index_col": self.index_col,
            "modalities": list(self.modalities),
        }

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
        if self.tabular_file is None or self.index_col is None:
            return None
        return self.read_demographics(self.tabular_file, self.index_col)

    @property
    def modalities(self):
        return self._modalities

    def _validate_modalities(self, modalities: Union[str, Iterable[str]]) -> set[str]:
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
            "Bad type for modalities. Expected str or Iterable[str], got"
            f"{type(modalities).__name__}"
        )

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
            demographics = CsvReader(tabular_file).data.to_pandas()
            # demographics = pd.read_csv(tabular_file, index_col=index_col)
            if index_col is not None:
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

    def _make_df_dir(
        self,
        root: Path,
        extensions: Union[str, tuple[str, ...]] = _extensions,
        modalities: Optional[Union[str, Iterable[str]]] = None,
    ) -> Tuple[set[str], pd.DataFrame]:
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

        df_dir = pd.DataFrame(rows)

        # === Ensure one valid file per modality folder
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

        # === Identify modalities (-available-, and -missing- when `modalities` is not None)
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

        # === Identify subjects that have all modalities
        _group_modality_sets = df_dir.groupby("subject")["modality"].apply(set)
        _matching_subject = _group_modality_sets[
            _group_modality_sets.apply(lambda x: modalities.issubset(x))
        ].index
        # === Filter df by 'subjects with all modalities' and 'modalities'
        df_dir = df_dir[
            df_dir["subject"].isin(_matching_subject)
            & df_dir["modality"].isin(modalities)
        ]

        if len(df_dir) == 0:
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: "
                f"No 'subject' matches all `modalities`: {modalities}"
            )
        logger.info(
            f"{len(df_dir['subject'].unique())} subjects in folder structure with all "
            f"modalities: {', '.join(modalities)}"
        )

        return modalities, df_dir

    def _make_dataset(
        self,
        root: Path,
        tabular_file: Optional[Path] = None,
        index_col: Optional[Union[int, str]] = None,
        modalities: Optional[Union[str, Iterable[str]]] = None,
    ) -> Tuple[set[str], List[Dict[str, Any]]]:
        """Builds samples in `dict` with `modalities` and `demographics`

        Args:
            root: path to medical folder
            tabular_file: path to demographics file
            index_col: Index column that matches <subject>
            modalities: Iterable of modalities to recover. Defaults to None.

        Raises:
            FedbiomedError:
            - if one of `tabular_file` and `index_col` is given and the other is not

        Returns:
            `modalities` and `samples`
        """
        if (tabular_file is None) != (index_col is None):
            raise FedbiomedError(
                f"{ErrorNumbers.FB613.value}: "
                "Arguments `tabular_file` and `index_col`, both or none are expected"
            )

        samples = []
        modalities, df_dir = self._make_df_dir(root=root, modalities=modalities)
        subject_groups = dict(tuple(df_dir.groupby("subject")))
        subjects = set(subject_groups.keys())

        if tabular_file is not None:
            demographics = self.read_demographics(tabular_file, index_col)
            subjects = subjects.intersection(demographics.index.values)

        for subject in subjects:
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

        logger.info(f"{len(samples)} complete samples successfully identified")
        return modalities, samples

    def _get_nontransformed_item(self, index: int) -> Dict[str, Any]:
        """Retrieve a data sample without applying transforms"""
        sample = self._samples[index]
        try:
            data = {
                modality: self._reader(sample[modality]) for modality in self.modalities
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


# TODO: dlp

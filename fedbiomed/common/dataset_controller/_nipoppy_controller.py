# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset_controller._controller import Controller
from fedbiomed.common.dataset_reader._nipoppy_reader import NipoppyReader
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError

SessionFilters = Union[str, List[str], List[Tuple[str, str]]]
Derivatives = List[Tuple[str, str, str]]
InstalledPipelines = Dict[str, Dict[str, List[str]]]  # {type: {name: [versions]}}


class NipoppyController(Controller):
    """Controller for datasets organized as a Nipoppy study.

    The constructor reads the requested phenotypes/derivatives and applies
    session filters, `dropna` and the optional whole-dataframe transform. With
    no selection (as at dataset registration), all phenotypes are read.

    !!! Note
        `_controller_kwargs` (stored in the node database as
        `dataset_parameters` at registration) includes the study's
        `available_phenotypes` and `installed_pipelines`, read from disk. The
        node passes them back to `NipoppyDataset.load`, which exposes them
        through `get_metadata`. They may become out of sync if the study
        changes after registration: re-register the dataset to refresh them.
    """

    _reader: NipoppyReader

    COL_PARTICIPANT_ID = "participant_id"
    COL_SESSION_ID = "session_id"

    def __init__(
        self,
        root: Union[str, Path],
        phenotypes: Optional[List[str]] = None,
        derivatives: Optional[Derivatives] = None,
        session_filters: Optional[SessionFilters] = None,
        drop_na: bool = False,
        drop_na_kwargs: Optional[Dict[str, Any]] = None,
        whole_df_transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        dlp: Optional[DataLoadingPlan] = None,
    ) -> None:
        """Constructor of the class

        Args:
            root: Root directory of the Nipoppy study
            phenotypes: Phenotypes to read. If both `phenotypes` and
                `derivatives` are `None`, all phenotypes are read.
            derivatives: Derivatives to read, as `(pipeline, version, path)` tuples
            session_filters: Sessions to keep. Either a session id, a list of
                session ids, or a list of `(participant_id, session_id)` tuples.
            drop_na: Whether to drop rows with missing values after filtering.
                Defaults to `False` so that, at dataset registration (where the
                node builds the controller without these options), the stored
                shape counts all participant-session rows. `NipoppyDataset`
                always passes its own `drop_na` value explicitly.
            drop_na_kwargs: Keyword arguments forwarded to `pd.DataFrame.dropna`
            whole_df_transform: Callable applied to the whole dataframe after
                filtering and `dropna`
            dlp: Data loading plan. Not supported yet for Nipoppy datasets;
                must be `None`.

        Raises:
            FedbiomedValueError: if the reader rejects the requested columns
            FedbiomedError: if `root` does not exist, a `dlp` is given, the data
                cannot be read, or no data is left after filtering
        """
        super().__init__()  # initialises DataLoadingPlanMixin (self._dlp = None)

        if dlp is not None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Data loading plans are not supported "
                "for Nipoppy datasets."
            )

        self.root = root
        self._reader = NipoppyReader(self.root)

        # Read from disk: stored in the node database as `dataset_parameters`
        # (minus `root`) at registration
        self._controller_kwargs = {
            "root": str(self.root),
            "available_phenotypes": list(self._reader.get_schema().keys()),
            "installed_pipelines": self._reader.get_installed_pipelines(),
        }

        if isinstance(session_filters, str):
            session_filters = [session_filters]
        self._session_filters = session_filters
        self._drop_na = drop_na
        self._drop_na_kwargs = drop_na_kwargs if drop_na_kwargs is not None else {}
        self._whole_df_transform = whole_df_transform

        self._data = self._load_data(phenotypes=phenotypes, derivatives=derivatives)

    def _load_data(
        self,
        phenotypes: Optional[List[str]],
        derivatives: Optional[Derivatives],
    ) -> pd.DataFrame:
        """Reads the requested columns, then applies filters and transforms.

        Raises:
            FedbiomedValueError: if the reader rejects the requested columns
            FedbiomedError: if the data cannot be read, or no data is left
        """
        try:
            data = self._reader.read(phenotypes=phenotypes, derivatives=derivatives)
        except ValueError as e:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Invalid data selection. {e}"
            ) from e
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to read data. {e}"
            ) from e

        data = self._filter_transform_data(data)
        if data.empty:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: No data left after applying session "
                "filters and `dropna`. Please check your session filters and the dataset."
            )
        return data

    def _filter_transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies session filters, `dropna` and the whole-dataframe transform."""
        if self._session_filters:
            if all(isinstance(f, str) for f in self._session_filters):
                df = df[
                    df.index.get_level_values(self.COL_SESSION_ID).isin(
                        self._session_filters
                    )
                ]
            elif all(
                isinstance(f, (tuple, list)) and len(f) == 2
                for f in self._session_filters
            ):
                idx = pd.MultiIndex.from_tuples(
                    [tuple(f) for f in self._session_filters],
                    names=[self.COL_PARTICIPANT_ID, self.COL_SESSION_ID],
                )
                df = df.loc[df.index.intersection(idx)]
            else:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: `session_filters` must be a session "
                    "id, a list of session ids, or a list of "
                    "(participant_id, session_id) pairs."
                )
        if self._drop_na:
            df = df.dropna(**self._drop_na_kwargs)
        if self._whole_df_transform is not None:
            if df.empty:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: No data left after applying session "
                    "filters and `dropna`. Please check your session filters and the dataset."
                )
            try:
                df = self._whole_df_transform(df)
            except Exception as e:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Failed to apply "
                    f"`whole_df_level_transform`. {e}"
                ) from e
        return df

    @property
    def index(self) -> pd.MultiIndex:
        """`(participant_id, session_id)` index of the loaded data."""
        return self._data.index

    def get_sample(self, index: int) -> pd.Series:  # type: ignore[override]
        """Retrieves a data sample without applying transforms

        Raises:
            FedbiomedError: if `index` is out of range
        """
        if not 0 <= index < len(self._data):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item at index {index}"
            )
        return self._data.iloc[index]

    def __len__(self) -> int:
        return len(self._data)

    def shape(self) -> Dict[str, Tuple[int, ...]]:
        """Shape of the loaded data, as `{"nipoppy": (n_rows, n_columns)}`."""
        return {"nipoppy": self._data.shape}

    def get_types(self) -> Dict[str, str]:
        """Dtypes of all phenotypes in the study on disk, regardless of any selection.

        Stored in the node database at registration.
        """
        return self._reader.get_schema()

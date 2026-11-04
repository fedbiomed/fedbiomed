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

    - Data is read lazily
    - The controller is cheap to build because it only inspects the
    study layout
    - The tabular data is materialized by `select`, which reads
    the requested phenotypes/derivatives and applies session filters, `dropna`
    and the optional whole-dataframe transform.

    !! Note
        Metadata (`available_phenotypes`, `installed_pipelines`) is exposed
        through `metadata`, and contains exactly the values passed to the
        constructor: the node passes the ones stored in its database at dataset
        registration (training and analytics), so the dataset sees the same
        metadata the researcher saw when searching datasets. Values that are
        not passed are `None`; there is no fallback to the study on disk.
        `_controller_kwargs` (what gets stored in the node database at
        registration) is always read from disk. Stored values may become out
        of sync if the study changes after registration: re-register the
        dataset to refresh them.
    """

    _reader: NipoppyReader

    COL_PARTICIPANT_ID = "participant_id"
    COL_SESSION_ID = "session_id"

    def __init__(
        self,
        root: Union[str, Path],
        session_filters: Optional[SessionFilters] = None,
        drop_na: bool = True,
        drop_na_kwargs: Optional[Dict[str, Any]] = None,
        whole_df_transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        available_phenotypes: Optional[List[str]] = None,
        installed_pipelines: Optional[InstalledPipelines] = None,
        dlp: Optional[DataLoadingPlan] = None,
    ) -> None:
        """Constructor of the class

        Args:
            root: Root directory of the Nipoppy study
            session_filters: Sessions to keep. Either a session id, a list of
                session ids, or a list of `(participant_id, session_id)` tuples.
            drop_na: Whether to drop rows with missing values after filtering
            drop_na_kwargs: Keyword arguments forwarded to `pd.DataFrame.dropna`
            whole_df_transform: Callable applied to the whole dataframe after
                filtering and `dropna`
            available_phenotypes: Phenotypes stored in the node database at
                dataset registration. Exposed as-is through `metadata`
                (`None` if not given).
            installed_pipelines: Pipelines stored in the node database at
                dataset registration. Same behavior as `available_phenotypes`.
            dlp: Data loading plan. Not supported yet for Nipoppy datasets; 
                must be `None`.

        Raises:
            FedbiomedError: if `root` does not exist, or if a `dlp` is given
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

        # Values stored in the node database, as passed by the node (no fallback)
        self.metadata: Dict[str, Optional[Any]] = {
            "available_phenotypes": available_phenotypes,
            "installed_pipelines": installed_pipelines,
        }

        if isinstance(session_filters, str):
            session_filters = [session_filters]
        self._session_filters = session_filters
        self._drop_na = drop_na
        self._drop_na_kwargs = drop_na_kwargs if drop_na_kwargs is not None else {}
        self._whole_df_transform = whole_df_transform

        self._data: Optional[pd.DataFrame] = None  # set by `select`

    def select(
        self,
        phenotypes: Optional[List[str]] = None,
        derivatives: Optional[Derivatives] = None,
    ) -> pd.DataFrame:
        """Reads the requested columns and applies filtering and transforms.

        Must be called before `get_sample`. After this call, `__len__` and
        `shape` reflect the selected data.
        The purpose of `select` is to set the self._data attribute according
        to the requested phenotypes and derivatives.

        Args:
            phenotypes: Phenotypes to read. If both `phenotypes` and
                `derivatives` are `None`, all phenotypes are read.
            derivatives: Derivatives to read, as `(pipeline, version, path)` tuples

        Returns:
            The selected dataframe

        Raises:
            FedbiomedValueError: if the reader rejects the requested columns
            FedbiomedError: if the data cannot be read, or if the resulting data
                is empty or invalid
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
        self._validate_data(data)
        self._data = data
        return self._data

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

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validates the data after filtering.

        Raises:
            FedbiomedError: if the data is empty or a participant has several sessions
        """
        if df.empty:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: No data left after applying session "
                "filters and `dropna`. Please check your session filters and the dataset."
            )
        participant_ids = df.index.get_level_values(self.COL_PARTICIPANT_ID)
        if participant_ids.has_duplicates:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Some participants have more than one "
                "session. Use `session_filters` to keep a single session per participant."
            )

    def get_sample(self, index: int) -> pd.Series:  # type: ignore[override]
        """Retrieves a data sample without applying transforms

        Raises:
            FedbiomedError: if `select` was not called, or `index` is out of range
        """
        if self._data is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: No data selected. Call `select` before "
                "retrieving samples."
            )
        if not 0 <= index < len(self._data):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item at index {index}"
            )
        return self._data.iloc[index]

    def __len__(self) -> int:
        return self.shape()["nipoppy"][0]

    def shape(self) -> Dict[str, Tuple[int, ...]]:
        """Gets the shape of the dataset.

        If `select` was called, returns the shape of the selected data.
        Otherwise (e.g. at dataset registration) the shape is estimated:
        - with no filters, `dropna` or transform: number of participant-session
          rows in the study;
        - otherwise: all phenotypes are read and filters/transforms are applied.
          This may differ from the shape after `select`, since derivatives and
          the column subset affect `dropna`.
        """
        if self._data is not None:
            return {"nipoppy": self._data.shape}
        if (
            self._session_filters is None
            and self._whole_df_transform is None
            and not self._drop_na
        ):
            return {"nipoppy": (len(self._reader), 1)}
        return {"nipoppy": self._filter_transform_data(self._reader.read()).shape}

    def get_types(self) -> Dict[str, str]:
        """Dtypes of all phenotypes in the study on disk, regardless of any selection.

        Stored in the node database at registration. For the columns of the loaded
        samples, see `NipoppyDataset.analytics_schema`.
        """
        return self._reader.get_schema()

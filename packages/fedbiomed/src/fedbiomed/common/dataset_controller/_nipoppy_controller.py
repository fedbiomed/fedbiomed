# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from fedbiomed.common.dataset_controller._controller import Controller
from fedbiomed.common.dataset_reader._nipoppy_reader import NipoppyReader


class NipoppyController(Controller):
    _reader: NipoppyReader

    COL_PARTICIPANT_ID = "participant_id"
    COL_SESSION_ID = "session_id"

    def __init__(
        self,
        root: Union[str, Path],
        session_filters: Optional[str | List[str] | List[Tuple[str, str]]] = None,
        drop_na: bool = True,
        whole_df_transform: Optional[Callable] = None,
    ) -> None:
        """Constructor of the class

        Args:
            root: Root directory path

        Raises:
            FedbiomedError: if `root` does not exist
        """
        self.root = root
        self._reader = NipoppyReader(self.root)
        self._controller_kwargs = {
            "root": str(self.root),
        }
        self._data: Optional[pd.DataFrame] = None  # lazy loaded
        session_filters = [session_filters] if isinstance(session_filters, str) else session_filters  # wrap single string in list
        self._session_filters = session_filters
        self._drop_na = drop_na
        self._whole_df_transform = whole_df_transform

    def _read_and_filter_data(self,
                          phenotypes: Optional[List[str]] = None,
                          derivatives: Optional[List[Tuple[str, str, str]]] = None) -> pd.DataFrame:
        """Reads the data and applies filtering based on phenotypes and derivatives

        Args:
            phenotypes: List of phenotypes to filter the data
            derivatives: List of derivatives to filter the data

        Returns:
            Filtered DataFrame
        """
        self._data = self._reader._read(phenotypes=phenotypes, derivatives=derivatives)
        if self._session_filters:
            # self._session_filters at this point is guaranteed to be a non-empty list of either strings or tuples due to the wrapping logic in __init__
            if all(isinstance(filter, str) for filter in self._session_filters):
                self._data = self._data.query(
                    f"{NipoppyController.COL_SESSION_ID} in @self._session_filters"
                )
            elif all(isinstance(filter, tuple) and len(filter) == 2 for filter in self._session_filters):
                idx = pd.MultiIndex.from_tuples(self._session_filters, 
                                                names=[NipoppyController.COL_PARTICIPANT_ID, NipoppyController.COL_SESSION_ID])
                self._data = self._data.loc[idx]
        if self._drop_na:
            self._data = self._data.dropna()
        self._apply_whole_df_transform()
        self._validate_data_after_filtering()
        return self._data
    
    def get_sample(self, 
                   index: int, 
                   phenotypes: Optional[List[str]] = None,
                   derivatives: Optional[List[Tuple[str, str, str]]] = None) -> pd.DataFrame:
        """Retrieve a data sample without applying transforms"""
        if self._data is None:
            self._read_and_filter_data(phenotypes=phenotypes, derivatives=derivatives)
        return self._data.iloc[index]

    def __len__(self) -> int:
        return self.shape()['nipoppy'][0]

    def shape(self) -> Dict:
        if self._data is None:
            return {"nipoppy": (1, 1)}
        return {"nipoppy": self._data.shape}

    def _apply_whole_df_transform(self) -> None:
        """Apply any necessary transformations to the data after reading and filtering

        This method can be used to apply any transformations that are necessary after reading
        and filtering the data, such as encoding categorical variables, normalizing numerical
        variables, etc. This is a placeholder for now and can be implemented as needed.
        """
        if self._whole_df_transform:
            self._data = self._whole_df_transform(self._data)

    def _validate_data_after_filtering(self) -> None:
        """Validate the data after applying session filters

        Raises:
            ValueError: if the resulting data is empty after filtering
        """
        if self._data is None:
            raise RuntimeError("Data should have been loaded and filtered at this point, but it is None.")
        if self._data is not None and self._data.empty:
            raise ValueError("No data left after applying session filters. Please check your session filters and the dataset.")
        participant_ids = self._data.index.get_level_values(
            NipoppyController.COL_PARTICIPANT_ID
        ).tolist()
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("Some participants have more than one session")

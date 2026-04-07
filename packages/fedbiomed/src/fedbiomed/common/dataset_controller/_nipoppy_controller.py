# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
        session_filters: Optional[str | List[str] | List[Tuple[str,str]]] = None,
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
        self._data = self._data.reset_index()
        if self._session_filters:
            # self._session_filters at this point is guaranteed to be a non-empty list of either strings or tuples due to the wrapping logic in __init__
            if all(isinstance(filter, str) for filter in self._session_filters):
                self._data = self._data[self._data[NipoppyController.COL_SESSION_ID].isin(self._session_filters)]
            elif all(isinstance(filter, tuple) and len(filter) == 2 for filter in self._session_filters):
                idx = pd.MultiIndex.from_tuples(self._session_filters, 
                                                names=[NipoppyController.COL_PARTICIPANT_ID, NipoppyController.COL_SESSION_ID])
                self._data = self._data.set_index([NipoppyController.COL_PARTICIPANT_ID, NipoppyController.COL_SESSION_ID])
                self._data = self._data.loc[idx]   
                self._data = self._data.reset_index()
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
        return self.shape()[0]

    def shape(self) -> Dict:
        if self._data is None:
            return {"nipoppy": (-1, -1)}
        return {"nipoppy": self._data.shape}

    def _validate_data_after_filtering(self) -> None:
        """Validate the data after applying session filters

        Raises:
            ValueError: if the resulting data is empty after filtering
        """
        if self._data is not None and self._data.empty:
            raise ValueError("No data left after applying session filters. Please check your session filters and the dataset.")
        participant_ids = self._data[NipoppyController.COL_PARTICIPANT_ID].tolist()
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("Some participants have more than one session")

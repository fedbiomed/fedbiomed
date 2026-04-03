# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller._controller import Controller
from fedbiomed.common.dataset_reader._nipoppy_reader import NipoppyReader
from fedbiomed.common.exceptions import FedbiomedError


class NipoppyController(Controller):
    _reader: NipoppyReader

    def __init__(
        self,
        root: Union[str, Path],
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

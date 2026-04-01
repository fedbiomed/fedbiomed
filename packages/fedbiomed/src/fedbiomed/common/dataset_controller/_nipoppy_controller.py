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

    def get_sample(self, 
                   index: int, 
                   phenotypes: Optional[List[str]] = None,
                   derivatives: Optional[List[Tuple[str, str, str]]] = None) -> pd.DataFrame:
        """Retrieve a data sample without applying transforms"""
        return self._reader.get(index, phenotypes=phenotypes, derivatives=derivatives)

    def __len__(self) -> int:
        return self._reader.len()

    def shape(self) -> Dict:
        return self._reader.shape()

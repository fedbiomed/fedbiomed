# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader implementation for CSV file
"""
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from nipoppy import NipoppyDataRetriever

from fedbiomed.common.exceptions import FedbiomedError


class NipoppyReader:
    def __init__(
        self,
        path: Path,
    ) -> None:
        """Constructs the csv reader.

        Args:
            path: The path of the csv file that contains the dataset.
        """
        self._path = path
        self._retriever = NipoppyDataRetriever(self._path)

    def _read(self,
              phenotypes: Optional[List[str]] = None,
              derivatives: Optional[List[Tuple[str, str, str]]] = None) -> pd.DataFrame:
        """Reads all dataset and returns the dataframe.

        Returns:
            Pandas DataFrame: The content of the CSV file.
        Raises:
            FedbiomedError: if the CSV file cannot be read due to inconsistent lines
        """
        return self._retriever.get_tabular_data(phenotypes=phenotypes, derivatives=derivatives)

    
# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader implementation for CSV file
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from nipoppy import NipoppyDataRetriever
from nipoppy.layout import DatasetLayout as NippopyDatasetLayout
from nipoppy.study import Study as NippopyStudy


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

    def read(self,
              phenotypes: Optional[List[str]] = None,
              derivatives: Optional[List[Tuple[str, str, str]]] = None) -> pd.DataFrame:
        """Reads all dataset and returns the dataframe.

        Returns:
            Pandas DataFrame: The content of the CSV file.
        Raises:
            FedbiomedError: if the CSV file cannot be read due to inconsistent lines
        """
        if phenotypes is None and derivatives is None:
            return self._retriever.get_all_phenotypes()
        return self._retriever.get_tabular_data(phenotypes=phenotypes, derivatives=derivatives)

    def get_schema(self) -> Dict[str, str]:
        """Gets the schema of the dataset.

        Returns:
            Dict[str, str]: The schema of the dataset.
        """
        return {str(col): str(dtype) for col, dtype in self._retriever.get_all_phenotypes().dtypes.items()}

    def __len__(self):
        return len(NippopyStudy(NippopyDatasetLayout(self._path)))

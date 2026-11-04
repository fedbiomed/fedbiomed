# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Reader implementation for Nipoppy studies
"""
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError

if TYPE_CHECKING:
    from nipoppy.study import Study as NipoppyStudy


class NipoppyReader:
    def __init__(
        self,
        path: Path,
    ) -> None:
        """Constructs the Nipoppy reader.

        Nipoppy is imported here rather than at module level, because it is only
        needed to read Nipoppy studies.

        Args:
            path: Root directory of the Nipoppy study.

        Raises:
            FedbiomedError: if Nipoppy is not installed
        """
        try:
            from nipoppy import NipoppyDataRetriever
        except ImportError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Nipoppy datasets require the `nipoppy` "
                'package. Please install it with `pip install "nipoppy>=0.4.7"`.'
            ) from e

        self._path = path
        self._retriever = NipoppyDataRetriever(self._path)

    @cached_property
    def _study(self) -> "NipoppyStudy":
        """Nipoppy study, built on first use."""
        from nipoppy.layout import DatasetLayout
        from nipoppy.study import Study

        return Study(DatasetLayout(self._path))

    def read(self,
              phenotypes: Optional[List[str]] = None,
              derivatives: Optional[List[Tuple[str, str, str]]] = None) -> pd.DataFrame:
        """Reads phenotypes and derivatives from the study.

        Args:
            phenotypes: Phenotypes to read.
            derivatives: Derivatives to read, as `(pipeline, version, path)` tuples.

        Returns:
            Pandas DataFrame indexed by `(participant_id, session_id)`. If both
            `phenotypes` and `derivatives` are `None`, all phenotypes are returned.
        """
        if phenotypes is None and derivatives is None:
            return self._retriever.get_all_phenotypes()
        return self._retriever.get_tabular_data(phenotypes=phenotypes, derivatives=derivatives)

    def get_schema(self) -> Dict[str, str]:
        """Gets the schema of the phenotypes.

        Returns:
            Dict[str, str]: Mapping of phenotype name to dtype.
        """
        return {str(col): str(dtype) for col, dtype in self._retriever.get_all_phenotypes().dtypes.items()}

    def get_installed_pipelines(self) -> Dict[str, Dict[str, List[str]]]:
        """Gets the installed pipelines.

        Returns:
            Nested dictionary `{pipeline_type: {pipeline_name: [version, ...]}}`,
            with pipeline types "bidsification", "processing" and "extraction".
        """
        return self._study.get_installed_pipelines()

    def __len__(self) -> int:
        """Number of unique participant-session combinations in the study."""
        return len(self._study)

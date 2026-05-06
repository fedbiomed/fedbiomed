# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Union

from fedbiomed.common.dataset_controller._controller import Controller
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger


class CustomController(Controller):
    """Controller for user-defined custom datasets.

    Stores only the root path. methods are delegated to :class:`CustomDataset`
    """

    def __init__(self, root: Union[str, Path]) -> None:
        """Args:
            root: Dataset root directory (must exist on disk).

        Raises:
            FedbiomedError: if *root* is not a valid, existing path.
        """
        super().__init__()  # initialises self._dlp = None via Controller → DataLoadingPlanMixin
        self.root = root
        self._controller_kwargs = {"root": str(self.root)}

    def shape(self):
        """Not meaningful for custom datasets — shape is owned by CustomDataset."""
        logger.warning(
            "CustomController.shape() should not be called; shape is defined by the CustomDataset."
        )
        return None

    def get_sample(self, index: int):
        """Not meaningful for custom datasets — data access is owned by CustomDataset."""
        logger.warning(
            "CustomController.get_sample() should not be called; data access is handled by CustomDataset."
        )
        return None

    def get_types(self):
        """Not meaningful for custom datasets — type info is owned by CustomDataset."""
        logger.warning(
            "CustomController.get_types() should not be called; type information is defined by the CustomDataset."
        )
        return None

    def __len__(self) -> int:
        """Not supported — length is owned by CustomDataset.

        Raises:
            FedbiomedError: always.
        """
        raise FedbiomedError(
            "'__len__' is not supported on CustomController. "
            "Length must be implemented in the CustomDataset."
        )

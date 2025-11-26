# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Controller implementation for MedNIST
"""

import os
import tarfile
from pathlib import Path
from typing import Any, Dict, Union
from urllib.request import urlretrieve

from torchvision.datasets import ImageFolder

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._controller import Controller


def download_mednist(root: Path) -> None:
    """Download MedNIST dataset in root

    Raises:
        FedbiomedError:
        - If there is a problem downloading or extracting MedNIST
    """
    if not isinstance(root, Path):
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: Expected `root` to be of type `Path`"
        )

    URL = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    filepath = root / "MedNIST.tar.gz"

    logger.info("Now downloading MedNIST...")
    try:
        urlretrieve(URL, filepath)
    except Exception as e:
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: The following error raised while "
            f"downloading MedNIST dataset from the MONAI repo: {e}"
        ) from e

    logger.info("Now extracting MEDNIST...")
    try:
        with tarfile.open(filepath) as tar_file:
            tar_file.extractall(root)
        os.remove(filepath)
    except Exception as e:
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: The following error raised while "
            f"extracting MedNIST.tar.gz: {e}"
        ) from e


class MedNistController(Controller):
    """Generic data controller where the data is arranged in this way:
    root
    └── MedNIST
        ├── AbdomenCT
        │   ├── 000000.jpeg
        │   └── ...
        ├── BreastMRI/
        ├── ChestCT/
        ├── CXR/
        ├── Hand/
        └── HeadCT/
    """

    def __init__(self, root: Union[str, Path]) -> Dict[str, Any]:
        """Constructor of the class

        Args:
            root: Root directory path

        Raises:
            FedbiomedError: if `ImageFolder` can not be initialized
        """
        self.root = root
        if not (self.root / "MedNIST").exists():
            download_mednist(self.root)

        try:
            self._dataset = ImageFolder(self.root / "MedNIST")
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                f"The following error raised while loading the data folder: {e}"
            ) from e

        self._controller_kwargs = {
            "root": str(self.root),
        }

    def get_sample(self, index: int) -> Dict[str, Any]:
        """Retrieve a data sample without applying transforms"""
        try:
            data, target = self._dataset[index]
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item at index {index}"
            ) from e
        return {"data": data, "target": target}

    def __len__(self):
        return len(self._dataset)

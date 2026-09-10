# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Controller implementation for MNIST
"""

from pathlib import Path
from typing import Any, Dict, Union

from torchvision.datasets import MNIST

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError

from ._controller import Controller


def download_mnist(root: Union[str, Path]) -> None:
    """Downloads the MNIST files into `root`, unless they are already there.

    Deploying a dataset is the one moment its files may legitimately be missing,
    so it is the only step that fetches them by default.

    Raises:
        FedbiomedError: if the files could not be downloaded or extracted
    """
    try:
        MNIST(root=root, download=True)
    except Exception as e:
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: The following error raised while "
            f"downloading MNIST into {root}. {e}"
        ) from e


class MnistController(Controller):
    """Generic Mnist controller where the data is arranged in this way:
    root
    └──MNIST
        └── raw
            ├── train-images-idx3-ubyte
            ├── train-labels-idx1-ubyte
            ├── t10k-images-idx3-ubyte
            └── t10k-labels-idx1-ubyte
    """

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        download: bool = False,
    ) -> None:
        """Constructor of the class

        Args:
            root: Root directory path
            train: If true then train files are used
            download: If true then downloads and extracts the files if they do not
                exist. Off by default: a deployed dataset is already under `root`,
                so reading it back reports missing files instead of fetching them.

        Raises:
            FedbiomedError: if `torchvision.datasets.MNIST` can not be initialized
        """
        self.root = root

        try:
            self._dataset = MNIST(root=self.root, train=train, download=download)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                f"Failed to instantiate MnistDataset object. {e}"
            ) from e

        # The entry locates the dataset; `train`/`download` are not recorded
        self._controller_kwargs = {"root": str(self.root)}

    def get_sample(self, index: int) -> Dict[str, Any]:
        """Retrieve a data sample without applying transforms"""
        try:
            data, target = self._dataset[index]
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item at index {index}"
            ) from e
        return {"data": data, "target": target}

    def __len__(self) -> int:
        return len(self._dataset)

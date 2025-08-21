# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Controller implementation for MNIST
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from PIL.Image import Image
from torchvision.datasets import folder

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError

from ._controller import Controller


class ImageFolderController(Controller):
    """Generic ImageFolder where data is arranged like this:
    root
    ├── class_x
    │   ├── xxx.ext
    │   ├── xxy.ext
    │   └── ...
    ├── class_y
    │   ├── 123.ext
    │   └── ...
    └── ...
    """

    _extensions: Tuple[str, ...] = folder.IMG_EXTENSIONS
    _loader: Callable[[str], Image] = staticmethod(folder.default_loader)
    _is_valid_file: Optional[Callable[[str], bool]] = None
    _allow_empty: bool = False

    def __init__(self, root: Union[str, Path]) -> Dict[str, Any]:
        """Constructor of the class

        Args:
            root: Root directory path

        Raises:
            FedbiomedError: if `ImageFolder` can not be initialized
        """
        self.root = root
        try:
            _, self._class_to_idx = folder.find_classes(directory=self.root)
            self._samples = folder.make_dataset(
                directory=self.root,
                class_to_idx=self._class_to_idx,
                extensions=self._extensions,
                is_valid_file=self._is_valid_file,
                allow_empty=self._allow_empty,
            )
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                "Failed to instantiate ImageFolderDataset object"
            ) from e

        self._controller_kwargs = {
            "root": str(self.root),
        }

    def _get_nontransformed_item(self, index: int) -> Dict[str, Any]:
        """Retrieve a data sample without applying transforms"""
        try:
            path, target = self._samples[index]
            data = self._loader(path)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item at index {index}"
            ) from e
        return {"data": data, "target": target}

    def __len__(self):
        return len(self._samples)

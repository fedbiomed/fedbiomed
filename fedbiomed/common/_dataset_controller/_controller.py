# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract classes for controllers
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

from PIL import Image

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


class Controller(ABC):
    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, path_input: Union[str, Path]):
        self._root = self._normalize_path(path_input)

    def _normalize_path(self, path_input: Union[str, Path]):
        """Normalizes `path_input` into type `Path`

        Raises:
            FedbiomedError:
            - if root type is not str or pathlib.Path
            - if root does not exist
        """
        if not isinstance(path_input, (str, Path)):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected a string or Path, got "
                f"{type(path_input).__name__}"
            )
        path = Path(path_input).expanduser().resolve()
        if not path.exists():
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Path does not exist, {str(path)}"
            )
        return path

    @abstractmethod
    def _get_nontransformed_item(self, index: int) -> Dict[str, Any]:
        """Retrieve a data sample without applying transforms"""
        pass

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._get_nontransformed_item(index=index)

    def shape(self):
        sample = self._get_nontransformed_item(0)

        if not isinstance(sample, dict):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Expected `sample` to be a `dict`, got "
                f"{type(sample).__name__}"
            )

        output = {}
        for key, val in sample.items():
            if hasattr(val, "shape"):
                output[key] = val.shape
            elif isinstance(val, (int, float)):
                output[key] = 1
            elif isinstance(val, dict):
                output[key] = len(val)
            elif isinstance(val, Image.Image):
                output[key] = {"size": val.size, "mode": val.mode}
                # (len(val.getbands()),) + val.size  # C, H, W
            elif val is None:
                output[key] = (None,)
            else:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Not possible to get shape for value "
                    f"of key: {key} that is type {type(val).__name__}"
                )
        return output

    def get_types(self):
        return {
            _k: type(_v).__name__ for _k, _v in self._get_nontransformed_item(0).items()
        }

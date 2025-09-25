# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Base abstract classes for controllers
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

from PIL.Image import Image

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlanMixin
from fedbiomed.common.exceptions import FedbiomedError


class Controller(ABC, DataLoadingPlanMixin):
    _controller_kwargs: Dict[str, Any]

    # === Properties ===
    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, path_input: Union[str, Path]):
        """Root setter

        Raises:
            FedbiomedError:
            - if root type is not str or pathlib.Path
            - if root does not exist
        """
        if not isinstance(path_input, (str, Path)):
            raise FedbiomedError(
                ErrorNumbers.FB632.value
                + ": Expected a string or Path, got "
                + type(path_input).__name__
            )
        path = Path(path_input).expanduser().resolve()
        if not path.exists():
            raise FedbiomedError(
                ErrorNumbers.FB632.value + ": Path does not exist, " + str(path)
            )
        self._root = path

    # === Abstract functions ===
    @abstractmethod
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Retrieve a data sample without applying transforms"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    # === Functions ===
    def get_types(self):
        """Get `type` directly from values in `dict` returned by `get_sample`"""
        return {_k: type(_v).__name__ for _k, _v in self.get_sample(0).items()}

    def shape(self) -> Dict[str, Any]:
        """Get `shape` directly from values in `dict` returned by `get_sample`"""
        # Supported: int, float, dict, PIL.Image, None and obj.shape (if available)
        # This function can be overwritten for specific cases in child class
        sample = self.get_sample(0)

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
            elif isinstance(val, Image):
                output[key] = {"size": val.size, "mode": val.mode}
            elif val is None:
                output[key] = None
            else:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Not possible to get shape for value "
                    f"of key: {key} that is type {type(val).__name__}"
                )
        return output

    def validate(self) -> None:
        """Validates coherence of controller

        Raises:
            FedbiomedError: if coherence issue is found
        """
        return None

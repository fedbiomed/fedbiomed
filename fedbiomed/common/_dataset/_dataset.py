# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from fedbiomed.common.constants import ErrorNumbers, _BaseEnum
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError


class DataReturnFormat(_BaseEnum):
    NUMPY = np.ndarray
    SKLEARN = (np.ndarray, pd.DataFrame, pd.Series)
    TORCH = torch.Tensor


Transform = Optional[Union[Callable, Dict[str, Callable]]]


class Dataset(ABC):
    _controller = None
    _to_format: DataReturnFormat = DataReturnFormat.TORCH

    @property
    def to_format(self) -> DataReturnFormat:
        return self._to_format

    @to_format.setter
    def to_format(self, to_format_input: DataReturnFormat):
        if not isinstance(to_format_input, DataReturnFormat):
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: `to_format` is not `DataReturnFormat` type"
            )
        self._to_format = to_format_input

    def constructor_controller(self):
        # TODO
        print("constructor?")
        pass

    def add_arguments(self):
        # TODO
        print("args?")
        pass

    @abstractmethod
    def _validate(self) -> None:
        pass

    def _get_nontransformed_item(self, index: int) -> Dict[str, Any]:
        try:
            item = self._controller._get_nontransformed_item(index=index)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item from controller"
            ) from e
        return item

    @abstractmethod
    def _apply_transforms(self, sample: Dict[str, Any]) -> Tuple[Any, Any]:
        pass

    def __getitem__(self, idx: int):
        return self._apply_transforms(self._get_nontransformed_item(idx))

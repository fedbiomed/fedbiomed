from abc import ABC
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch import Tensor
from torchvision.datasets import MNIST, ImageFolder, folder

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import (
    FedbiomedError,
    FedbiomedTypeError,
    FedbiomedValueError,
)

from ._dataset import (
    DataReturnFormat,
    Dataset,
    ItemModality,
    ModalityType,
    SampleModality,
)


class TorchDataset(ABC, Dataset):
    _to_torch: DataReturnFormat = DataReturnFormat.TORCH

    # ALIASES =================================================================
    @property
    def transform(self):
        return self.framework_transform

    @property
    def target_transform(self):
        return self.framework_target_transform

    # FUNCTIONS ===============================================================
    def _get_nontransformed_item(self, idx: int) -> tuple[Any, Any]:
        try:
            data, target = self._dataset[idx]
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                f"Failed to retrieve item at index {idx}. {e}"
            ) from e
        return data, target

    def _get_generic_format_item(self, idx: int) -> SampleModality:
        data, target = self._get_nontransformed_item(idx)
        data_item = ItemModality(
            name="data",
            type=ModalityType.IMAGE,
            data=np.array(data),
        )
        target_item = ItemModality(
            name="target",
            type=ModalityType.TABULAR,
            data=pd.DataFrame([target]),
        )
        return tuple({item.name: item} for item in [data_item, target_item])

    def _get_torch_format_item(self, idx) -> tuple[Tensor, Tensor]:
        data, target = self._get_nontransformed_item(idx)
        try:
            data = self.framework_transform(data)
            target = self.framework_target_transform(target)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply transforms. {e}"
            ) from e
        if not isinstance(data, Tensor):
            raise FedbiomedTypeError(
                f"Expected output of type 'Tensor' after applying 'transform', "
                f"but got '{type(data).__name__}' instead. "
            )
        if not isinstance(target, Tensor):
            raise FedbiomedTypeError(
                f"Expected output of type 'Tensor' after applying 'target_transform', "
                f"but got '{type(target).__name__}' instead. "
            )
        return data, target

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        if self._to_format == DataReturnFormat.TORCH:
            return self._get_torch_format_item(idx)
        elif self._to_format == DataReturnFormat.DEFAULT:
            return self._get_generic_format_item(idx)
        else:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: DataReturnFormat not supported."
            )

    def __len__(self) -> int:
        return len(self._dataset)

    def to_torch(self) -> bool:
        self._to_format = DataReturnFormat.TORCH
        return True

    def validate(self):
        pass

    def set_transforms(
        self,
        transform: Callable = None,
        target_transform: Callable = None,
        framework_transform: Callable = None,
        framework_target_transform: Callable = None,
    ):
        if transform is not None and framework_transform is not None:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Not necessary to provide "
                "'transform' and 'framework_transform'. Choose one."
            )
        self.framework_transform = (
            transform
            if transform is not None
            else framework_transform
            if framework_transform is not None
            else T.ToTensor()  # default
        )

        if target_transform is not None and framework_target_transform is not None:
            raise FedbiomedValueError(
                f"{ErrorNumbers.FB632.value}: Not necessary to provide "
                "'target_transform' and 'framework_target_transform'. Choose one."
            )
        self.framework_target_transform = (
            target_transform
            if target_transform is not None
            else framework_target_transform
            if framework_target_transform is not None
            else torch.tensor  # default
        )


class MnistDataset(TorchDataset):
    """Generic MNIST where data is arranged like this:
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
        train: str = True,
        transform: Callable = None,
        framework_transform: Callable = None,
        target_transform: Callable = None,
        framework_target_transform: Callable = None,
        download: bool = False,
    ):
        self.root = root
        self.set_transforms(
            transform=transform,
            target_transform=target_transform,
            framework_transform=framework_transform,
            framework_target_transform=framework_target_transform,
        )
        try:
            self._dataset = MNIST(
                root=self.root,
                train=train,
                download=download,
            )
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                "Failed to instantiate MnistDataset object. {e}"
            ) from e


class ImageFolderDataset(TorchDataset):
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

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        framework_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        framework_target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        self.root = root
        self.set_transforms(
            transform=transform,
            target_transform=target_transform,
            framework_transform=framework_transform,
            framework_target_transform=framework_target_transform,
        )
        try:
            self._dataset = ImageFolder(
                root=self.root,
                loader=loader,
                is_valid_file=is_valid_file,
                allow_empty=allow_empty,
            )
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                "Failed to instantiate ImageFolderDataset object. {e}"
            ) from e

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Controller implementation for Image Folder
"""

import os
import tarfile
from pathlib import Path
from tarfile import TarError
from typing import Tuple, Union
from urllib.error import ContentTooShortError, HTTPError, URLError
from urllib.request import urlretrieve

import numpy as np
from torchvision import datasets

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_types import (
    DatasetData,
    DatasetDataItem,
    DatasetDataModality,
    DatasetDataStructured,
    DataType,
)
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._controller import Controller


class ImageFolderController(Controller):
    """Generic data controller where the data are images arranged in this way:

        root
        ├── class_x
        │   ├── xxx.ext
        │   ├── xxy.ext
        │   └── ...
        ├── class_y
        │   ├── 123.ext
        │   └── ...
        └── ...

    Uses `torchvision.datasets.ImageFolder`.
    """

    def __init__(self, root: Union[str, Path], is_mednist=False, **kwargs) -> None:
        """Constructor of the class.

        Args:
            root: Root directory path.
            is_mednist: If True, download MedNIST into indicated path and add Mednist
                to `self.root` if last folder in path is not called MedNIST.

        Raises:
            FedbiomedError:
            - if root is not valid or do not exist (from `Controller`)
            - if MedNIST download fails
            - if `datasets.ImageFolder` can not be initialized
                (classes, samples and loader)
        """
        self.root = root

        if is_mednist and self.root.name != "MedNIST":
            self._download_mednist()
            self.root = self.root / "MedNIST"

        try:
            dataset = datasets.ImageFolder(self.root)
        except Exception as e:
            msg = (
                ErrorNumbers.FB632.value
                + "\n The following error raised while loading the data folder: "
                + str(e)
            )
            logger.error(msg)
            raise FedbiomedError(msg) from e

        self._class_to_idx = dataset.class_to_idx
        self._samples = dataset.samples
        self._loader = dataset.loader
        self._dataset_data_meta = self._get_dataset_data_meta()
        super().__init__(**kwargs)

    def _get_nontransformed_item(
        self, index: int
    ) -> Tuple[DatasetDataItem, DatasetDataItem]:
        """Returns image and target associated to index.

        Args:
            index (int): Index

        Returns:
            ({"data": ImagePIL}, {"target": int})

        Raises:
            FedbiomedError: if is not possible to load image from path in samples[index]
        """
        try:
            data_item = {"data": self._loader(self._samples[index][0])}
        except Exception as e:
            msg = (
                ErrorNumbers.FB632.value
                + "\n The following error raised while loading the image file: "
                + str(e)
            )
            logger.error(msg)
            raise FedbiomedError(msg) from e

        target_item = {"target": self._samples[index][1]}
        return data_item, target_item

    def _get_dataset_data_meta(self) -> DatasetData:
        """Returns meta data of samples recovered with `_get_nontransformed_item`"""
        data_item, target_item = self._get_nontransformed_item(index=0)

        # shape = (H, W, C) if C!=1 else (H, W)
        _channels = len(data_item["data"].getbands())
        _channels = () if _channels == 1 else (_channels,)
        data_meta = {
            "data": DatasetDataModality(
                modality_name="data",
                type=DataType.IMAGE,
                shape=(data_item["data"].size + _channels),
            )
        }

        target_meta = {
            "target": DatasetDataModality(
                modality_name="target",
                type=DataType.TABULAR,
                shape=np.shape(target_item),
            )
        }

        return DatasetDataStructured(
            data=data_meta,
            target=target_meta,
            len=len(self._samples),
        )

    def validate(self):
        pass

    def _download_mednist(self) -> None:
        """Download MedNIST dataset in directory self.root

        Raises:
            FedbiomedError: If there is a problem downloading or extracting MedNIST
        """
        URL = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
        filepath = self.root / "MedNIST.tar.gz"
        logger.info("Now downloading MEDNIST...")

        try:
            urlretrieve(URL, filepath)
            with tarfile.open(filepath) as tar_file:
                logger.info("Now extracting MEDNIST...")
                tar_file.extractall(self.root)
            os.remove(filepath)
        except (
            URLError,
            HTTPError,
            ContentTooShortError,
            OSError,
            TarError,
            MemoryError,
        ) as e:
            msg = (
                ErrorNumbers.FB632.value
                + "\n The following error raised while "
                + "downloading MedNIST dataset from the MONAI repo: "
                + str(e)
            )
            logger.error(msg)
            raise FedbiomedError(msg) from e

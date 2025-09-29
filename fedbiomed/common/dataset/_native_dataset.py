from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller import (
    ImageFolderController,
    MedNistController,
    MnistController,
)
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError


class NativeDataset:
    dataset_to_controller = {
        "ImageFolderDataset": ImageFolderController,
        "MedNistDataset": MedNistController,
        "MnistDataset": MnistController,
    }

    _native_to_framework = {
        DataReturnFormat.SKLEARN: np.array,
        DataReturnFormat.TORCH: lambda x: (
            T.ToTensor()(x)  # In case the target is a PIL Image
            if isinstance(x, (Image.Image, np.ndarray))
            else torch.tensor(x)
        ),
    }

    def __init__(self, dataset, target, **kwargs):
        try:
            self.controller = self.dataset_to_controller[dataset.__name__]()
        except Exception as err:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                f"Failed to initialize controller, unexpected type "
                f"`{type(self).__name__}`"
            ) from err

    def complete_initialization(
        self,
        controller_kwargs: Dict[str, Any],
        to_format: DataReturnFormat,
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            controller_kwargs: arguments to create controller
            to_format: format associated to expected return format
        """
        self.to_format = to_format
        self.controller(controller_kwargs=controller_kwargs)

        sample = self._controller.get_sample(0)
        self._validate_format_conversion(sample["data"])
        self._validate_format_conversion(sample["target"])

    def _validate_format_conversion(self, data: Any, for_: Optional[str] = None) -> Any:
        """Validates format conversion and applies `transform`

        Args:
            data: from `self._controller.get_sample`
            transform: `Callable` given at instantiation of cls

        """
        converter = self._get_format_conversion_callable()

        try:
            data = converter(data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Unable to perform type conversion of "
                f"data to {self._to_format.value} {'for ' + for_ if for_ else ''}"
            ) from e

        if not isinstance(data, self._to_format.value):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                f"Expected type conversion for the data to return "
                f"`{self._to_format.value}`, got {type(data).__name__} "
                f"{'for ' + for_ if for_ else ''}"
            )

        return data

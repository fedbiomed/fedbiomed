from pathlib import Path
from typing import Any, Dict, Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset._custom_dataset import CustomDataset
from fedbiomed.common.dataset_controller._controller import Controller
from fedbiomed.common.exceptions import FedbiomedError


class CustomController(Controller):
    """Custom dataset controller for MNIST dataset"""

    def __init__(
        self,
        root: Union[str, Path],
        **kwargs,
    ) -> Dict[str, Any]:
        """Constructor of the class

        Args:
            root: Root directory path
            train: If true then train files are used
            download: If true then downloads and extracts the files if they do not exist

        Raises:
            FedbiomedError: if `torchvision.datasets.MNIST` can not be initialized
        """
        self.root = root

        try:
            self._dataset = CustomDataset(root=self.root)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                "Failed to instantiate MnistDataset object. {e}"
            ) from e

        self._controller_kwargs = {
            "root": str(self.root),
            **kwargs,
        }

    def get_sample(self, index: int) -> Dict[str, Any]:
        """Retrieve a data sample without applying transforms"""
        try:
            data, target = self._dataset.get_item(index)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item at index {index}"
            ) from e
        return {"data": data, "target": target}

    def __len__(self) -> int:
        return len(self._dataset)

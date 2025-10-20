from abc import abstractmethod
from typing import Any, Dict, Tuple

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError

from ._dataset import Dataset


class CustomDataset(Dataset):
    """A class representing a custom dataset.

    This class allows users to create and manage their own datasets
    for use in federated learning scenarios.
    """

    def __init_subclass__(cls, **kwargs):
        """Ensures that subclasses implement required methods"""
        super().__init_subclass__(**kwargs)

        # Ensure __getitem__ is not overwritten
        if (
            "__getitem__" in cls.__dict__
            and cls.__dict__["__getitem__"] is not CustomDataset.__getitem__
        ):
            raise FedbiomedError(
                "Overriding __getitem__ in CustomDataset subclasses is not allowed. "
                "Please overwrite get_item instead."
            )

        if (
            "__init__" in cls.__dict__
            and cls.__dict__["__init__"] is not CustomDataset.__init__
        ):
            raise FedbiomedError(
                "Overriding __init__ in CustomDataset subclasses is not allowed. "
                "Please overwrite `read` method instead. Using path attribute to "
                "access dataset location."
            )

        # Ensure read and get_item are implemented by subclass
        if "read" not in cls.__dict__ or not callable(cls.__dict__.get("read", None)):
            raise FedbiomedError(
                "CustomDataset subclass must implement a 'read' method. "
                "This method is required to load the dataset and must be defined in your subclass."
            )
        if "get_item" not in cls.__dict__ or not callable(
            cls.__dict__.get("get_item", None)
        ):
            raise FedbiomedError(
                "CustomDataset subclass must implement a 'get_item' method. "
                "This method is required to retrieve samples and must be defined in your subclass."
            )

        # Ensure __len__ is implemented by subclass
        if "__len__" not in cls.__dict__ or not callable(
            cls.__dict__.get("__len__", None)
        ):
            raise FedbiomedError(
                "CustomDataset subclass must implement a '__len__' method. "
                "This method is required to return the number of samples and must be "
                "defined in your subclass."
            )

    @abstractmethod
    def read(self) -> None:
        """Reads the dataset from the specified path.

        This method should be implemented by subclasses to load the dataset
        from the given path and prepare it for use.
        """
        pass

    @abstractmethod
    def get_item(self, index):
        """Retrieves a sample and its label by index.

        Args:
            index (int): The index of the sample to retrieve.
        """
        pass

    def complete_initialization(
        self, controller_kwargs: Dict[str, Any], to_format: DataReturnFormat
    ) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            path: path to dataset
            to_format: format associated to expected return format
        """

        self.path = controller_kwargs.get("root", None)
        if self.path is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Custom Dataset ERROR: 'root' must be provided in controller_kwargs to specify dataset location."
            )
        self._to_format = to_format

        # Call user defined read function to read the dataset
        try:
            self.read()
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to read "
                f"from dataset using read method. Please see error: {e}"
            ) from e

        # Following line is just to check that dataset is well implemented
        # and it return correct data type respecting to to_format
        try:
            sample = self[0]
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item "
                f"from dataset using get_item method. Please see error: {e}"
            ) from e
        if not isinstance(sample, tuple) or len(sample) != 2:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: get_item method must return a tuple of two elements"
                f" (data, target), but got {type(sample).__name__} with length {len(sample) if isinstance(sample, (list, tuple)) else 'N/A'}"
            )

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """Retrieves a sample and its target by index."""
        data, target = self.get_item(idx)

        def check_type(sample: Any, type_) -> None:
            """Check if sample is of expected type"""
            if not isinstance(sample, self._to_format.value):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: "
                    f"Expected return type for the {type_} is {self._to_format.value}, "
                    f"but got {type(sample).__name__} "
                )  # Applies transformations if any

        check_type(data, "data")
        check_type(target, "target")

        return data, target

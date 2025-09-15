import ast
import inspect
import textwrap
from abc import abstractmethod
from typing import Any, Tuple

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

        # Ensure get_item returns a tuple
        get_item_method = cls.__dict__.get("get_item", None)
        if get_item_method is not None:

            def returns_tuple_from_ast(method):
                source = inspect.getsource(method)
                # Requires to be unindented to be parsed correctly
                source = textwrap.dedent(source)
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Return):
                        if (
                            isinstance(node.value, ast.Tuple)
                            and len(node.value.elts) == 2
                        ):
                            return True
                return False

            if get_item_method is not None and not returns_tuple_from_ast(
                get_item_method
            ):
                raise FedbiomedError(
                    "CustomDataset subclass must implement get_item to "
                    "return a tuple of two elements for respectively data and "
                    "target e.g. 'return data, target'"
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

    def complete_initialization(self, path: str, to_format: DataReturnFormat) -> None:
        """Finalize initialization of object to be able to recover items

        Args:
            path: path to dataset
            to_format: format associated to expected return format
        """

        self.path = path
        self._to_format = to_format

        # Call user defined read function to read the dataset
        self.read()

        # Following line is just to check that dataset is well implemented
        # and it return correct data type respecting to to_format
        try:
            _ = self[0]
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item"
                f"from dataset using get_item method. Please see error: {e}"
            ) from e

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

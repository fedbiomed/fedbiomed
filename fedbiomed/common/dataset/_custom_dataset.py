# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

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
        """Prevents subclasses from overriding reserved methods and enforces required ones."""
        super().__init_subclass__(**kwargs)

        if "__getitem__" in cls.__dict__:
            raise FedbiomedError(
                "Overriding __getitem__ in CustomDataset subclasses is not allowed. "
                "Please overwrite get_item instead."
            )

        if "__init__" in cls.__dict__:
            raise FedbiomedError(
                "Overriding __init__ in CustomDataset subclasses is not allowed. "
                "Please overwrite `read` method instead. Using path attribute to "
                "access dataset location."
            )

        for method, label in [
            ("read", "read"),
            ("get_item", "get_item"),
            ("__len__", "__len__"),
        ]:
            implemented = any(
                method in base.__dict__
                for base in cls.__mro__
                if base is not CustomDataset and base not in CustomDataset.__mro__
            )
            if not implemented:
                raise FedbiomedError(
                    f"CustomDataset subclass '{cls.__name__}' must implement a '{label}' method."
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
        """Return a (data, target) tuple for the given index.

        Args:
            index (int): Index of the sample to retrieve.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        pass

    def complete_initialization(
        self, controller_kwargs: Dict[str, Any], to_format: DataReturnFormat
    ) -> None:
        """Finalize initialization of object to be able to recover items.

        Args:
            controller_kwargs: must contain a ``"root"`` key with the path to the dataset.
            to_format: expected format of data returned by ``__getitem__``.
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

        if len(self) == 0:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Custom Dataset ERROR: dataset is empty (len == 0)."
            )

        try:
            sample = self.get_item(0)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to retrieve item "
                f"from dataset using get_item method. Please see error: {e}"
            ) from e

        if not isinstance(sample, tuple) or len(sample) != 2:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: get_item method must return a tuple of two elements"
                f" (data, target), but got {type(sample).__name__} with"
                f" length {len(sample) if isinstance(sample, (list, tuple)) else 'N/A'}"
            )

        data, target = sample
        self._check_type(data, "data")
        self._check_type(target, "target")

    def _check_type(self, sample: Any, type_: str) -> None:
        """Check if sample is of expected type"""
        if not isinstance(sample, self._to_format.value):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                f"Expected return type for the {type_} is {self._to_format.value}, "
                f"but got {type(sample).__name__} "
            )

    def _apply_default_types(self, data: Any, _type: str) -> Any:
        """Applies default types for training plan framework to data

        Args:
            data: data to convert
            _type: label identifying the data role (e.g. ``"data"`` or ``"target"``),
                used in error messages.

        Returns:
            Converted data
        """
        try:
            data = self._get_default_types_callable()(data)
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply default training plan types "
                f"to `{_type}` in {self._to_format.value} format."
            ) from e
        return data

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """Retrieves a sample and its target by index."""
        data, target = self.get_item(idx)

        self._check_type(data, "data")
        self._check_type(target, "target")

        # Apply default types for training plan framework
        data = self._apply_default_types(data, "data")
        target = self._apply_default_types(target, "target")

        return data, target

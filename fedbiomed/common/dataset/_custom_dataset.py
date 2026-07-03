# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np

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
                f"{ErrorNumbers.FB632.value}: Overriding __getitem__ in CustomDataset "
                "subclasses is not allowed. Please overwrite get_item instead."
            )

        if "__init__" in cls.__dict__:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Overriding __init__ in CustomDataset "
                "subclasses is not allowed. Please overwrite `read` method instead. "
                "Using path attribute to access dataset location."
            )

        for reserved in ("load", "path"):
            if reserved in cls.__dict__:
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: Overriding `{reserved}` in "
                    "CustomDataset subclasses is not allowed; it is reserved for "
                    "internal use and managed by the node."
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
        """Return the sample for the given index.

        May return either ``data`` alone, or a ``(data, target)`` tuple. When
        only ``data`` is returned, the target is treated as ``None``.

        Args:
            index (int): Index of the sample to retrieve.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        pass

    @property
    def path(self) -> Union[str, Path]:
        """Dataset location (file or folder) on the node; set by `load`, read-only."""
        return self.__path

    def load(
        self,
        root: Union[str, Path],
        to_format: DataReturnFormat,
    ) -> None:
        """Finalize initialization of object to be able to recover items.

        Args:
            root: path to the dataset (must not be ``None``).
            to_format: expected format of data returned by ``__getitem__``.
        """

        if root is None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Custom Dataset ERROR: 'root' must be provided to specify dataset location."
            )
        self.__path = root
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

        data, target = self._split_sample(sample)
        target = self._normalize_target(target)
        self._composed: dict[str, Union[bool, None]] = {
            "data": None,
            "target": None,
        }
        self._check_type(data, "data")
        if target is not None:
            self._check_type(target, "target")

    def _split_sample(self, sample: Any) -> Tuple[Any, Any]:
        """Split a ``get_item`` return into ``(data, target)``.

        A 2-element tuple is read as ``(data, target)``; anything else is
        treated as ``data`` with a ``None`` target.
        """
        if isinstance(sample, tuple) and len(sample) == 2:
            return sample
        return sample, None

    def _normalize_target(self, target: Any) -> Any:
        """Coerce a numpy scalar `target` to a 0-d ``np.ndarray`` (scikit-learn only).

        Indexing a 1-D label array yields a numpy scalar; treat it as a 0-d array
        so it passes the type check. No-op for torch, dict targets, and ``None``.
        """
        if self._to_format.value is np.ndarray and isinstance(target, np.generic):
            return np.asarray(target)
        return target

    def _check_type(self, sample: Any, type_: str) -> None:
        """Check if sample is of expected type"""
        # First time we see a sample of this type, determine if it is composed or not
        if self._composed[type_] is None:
            self._composed[type_] = not isinstance(sample, self._to_format.value)

        # Check non-composed sample
        if self._composed[type_] is False and not isinstance(
            sample, self._to_format.value
        ):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: "
                f"Expected return type for the {type_} is {self._to_format.value}, "
                f"but got {type(sample).__name__} "
            )

        # Check composed sample
        if self._composed[type_] is True:
            if not isinstance(sample, dict):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: "
                    f"Expected return type for the {type_} is dict of {self._to_format.value}, "
                    f"but got {type(sample).__name__} "
                )
            if not all(
                isinstance(elem, self._to_format.value) for elem in sample.values()
            ):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB632.value}: "
                    f"Expected return type for the {type_} is dict of {self._to_format.value}, "
                    f"but got elements of type {list(map(type, sample.values()))} "
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
            if not self._composed[_type]:
                data = self._get_default_types_callable()(data)
            else:
                data = {
                    key: self._get_default_types_callable()(value)
                    for key, value in data.items()
                }
        except Exception as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Failed to apply default training plan types "
                f"to `{_type}` in {self._to_format.value} format."
            ) from e
        return data

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """Retrieves a sample and its target by index."""
        data, target = self._split_sample(self.get_item(idx))
        target = self._normalize_target(target)

        self._check_type(data, "data")
        data = self._apply_default_types(data, "data")

        if target is not None:
            self._check_type(target, "target")
            target = self._apply_default_types(target, "target")

        return data, target

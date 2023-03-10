# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""JSON serialization utils, wrapped into a namespace class."""

import json
from typing import Any

import numpy as np
import torch


__all__ = [
    "JsonSerializer",
]


class JsonSerializer:
    """JSON-based (de)serialization utils, wrapped into a namespace class.

    This class has no value being instantiated: it merely acts as a namespace
    to pack together encoding and decoding utils to convert data to and from
    JSON dump strings or text files.

    The JSON encoding and decoding capabilities are enhanced to add support
    for the following non-standard object types:
        - numpy arrays and scalars
        - torch tensors (that are always loaded on CPU)
    """

    @classmethod
    def dumps(cls, obj: Any) -> str:
        """Serialize data into a JSON-encoded string.

        Args:
            obj: Data that needs encoding.

        Returns:
            JSON-encoded string that contains the input data.
        """
        return json.dumps(obj, default=cls._default)

    @classmethod
    def dump(cls, obj: Any, path: str) -> None:
        """Serialize data into a JSON dump file.

        Args:
            obj: Data that needs encoding.
            path: Path to the created dump file.
        """
        with open(path, "w", encoding="utf-8") as file:
            json.dump(obj, file, default=cls._default)

    @classmethod
    def loads(cls, string: str) -> Any:
        """Load serialized data from a JSON-encoded string.

        Args:
            string: JSON-encoded string that needs decoding.

        Returns:
            Data loaded and decoded from the input string.
        """
        return json.loads(string, object_hook=cls._object_hook)

    @classmethod
    def load(cls, path: str) -> Any:
        """Load serialized data from a JSON dump file.

        Args:
            path: Path to a JSON file, the contents of which to decode.

        Returns:
            Data loaded and decoded from the target file.
        """
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file, object_hook=cls._object_hook)

    @staticmethod
    def _default(obj: Any) -> Any:
        """Encode non-default object types into MsgPack-serializable data.

        The counterpart static method `unpack` may be used to recover
        the input objects from their encoded data.
        """
        if isinstance(obj, np.ndarray):
            spec = [obj.tobytes().hex(), obj.dtype.name, list(obj.shape)]
            return {"__type__": "np.ndarray", "value": spec}
        if isinstance(obj, np.generic):
            spec = [obj.tobytes().hex(), obj.dtype.name]
            return {"__type__": "np.generic", "value": spec}
        if isinstance(obj, torch.Tensor):
            obj = obj.cpu().numpy()
            spec = [obj.tobytes().hex(), obj.dtype.name, list(obj.shape)]
            return {"__type__": "torch.Tensor", "value": spec}
        # Raise on unsupported types.
        raise TypeError(f"Cannot serialize object of type '{type(obj)}'.")

    @staticmethod
    def _object_hook(obj: Any) -> Any:
        """De-serialize non-default object types encoded with `_default`."""
        if not (isinstance(obj, dict) and "__type__" in obj):
            return obj
        objtype = obj["__type__"]
        if objtype == "tuple":
            return tuple(obj["value"])
        if objtype == "np.ndarray":
            data, dtype, shape = obj["value"]
            data = bytes.fromhex(data)
            return np.frombuffer(data, dtype=dtype).reshape(shape)
        if objtype == "np.generic":
            data, dtype = obj["value"]
            data = bytes.fromhex(data)
            return np.frombuffer(data, dtype=dtype)[0]
        if objtype == "torch.Tensor":
            data, dtype, shape = obj["value"]
            data = bytes.fromhex(data)
            array = np.frombuffer(data, dtype=dtype).reshape(shape)
            return torch.from_numpy(array)
        return obj

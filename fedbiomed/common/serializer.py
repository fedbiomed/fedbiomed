# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""MsgPack serialization utils, wrapped into a namespace class."""

import dataclasses
from math import ceil
from typing import Any, Optional

import msgpack
import numpy as np
import torch
from declearn.model.api import Vector, VectorSpec
from declearn.utils import json_pack, json_unpack

from fedbiomed.common.exceptions import FedbiomedTypeError
from fedbiomed.common.logger import logger
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.optimizers import AuxVar, EncryptedAuxVar

__all__ = [
    "Serializer",
]


class Serializer:
    """MsgPack-based (de)serialization utils, wrapped into a namespace class.

    This class has no value being instantiated: it merely acts as a namespace
    to pack together encoding and decoding utils to convert data to and from
    MsgPack dump bytes or binary files.

    The MsgPack encoding and decoding capabilities are enhanced to add support
    for the following non-standard object types:
        - numpy arrays and scalars
        - torch tensors (that are always loaded on CPU)
        - tuples (which would otherwise be converted to lists)
    """

    @classmethod
    def dumps(cls, obj: Any, write_to: Optional[str] = None) -> bytes:
        """Serialize data into MsgPack-encoded bytes.

        Args:
            obj: Data that needs encoding.

        Returns:
            MsgPack-encoded bytes that contains the input data.
        """
        ser = msgpack.packb(obj, default=cls._default, strict_types=True)

        if write_to:
            with open(write_to, "wb") as file:
                file.write(ser)
                file.close()

        return ser

    @classmethod
    def dump(cls, obj: Any, path: str) -> None:
        """Serialize data into a MsgPack binary dump file.

        Args:
            obj: Data that needs encoding.
            path: Path to the created dump file.
        """
        with open(path, "wb") as file:
            msgpack.pack(obj, file, default=cls._default, strict_types=True)

    @classmethod
    def loads(cls, data: bytes) -> Any:
        """Load serialized data from a MsgPack-encoded string.

        Args:
            data: MsgPack-encoded bytes that needs decoding.

        Returns:
            Data loaded and decoded from the input bytes.
        """
        return msgpack.unpackb(data, object_hook=cls._object_hook, strict_map_key=False)

    @classmethod
    def load(cls, path: str) -> Any:
        """Load serialized data from a MsgPack dump file.

        Args:
            path: Path to a MsgPack file, the contents of which to decode.

        Returns:
            Data loaded and decoded from the target file.
        """
        with open(path, "rb") as file:
            return msgpack.unpack(
                file, object_hook=cls._object_hook, strict_map_key=False
            )

    @staticmethod
    def _default(obj: Any) -> Any:
        """Encode non-default object types into MsgPack-serializable data.

        The counterpart static method `unpack` may be used to recover
        the input objects from their encoded data.
        """
        # Big integer
        if isinstance(obj, int):
            return {
                "__type__": "int",
                "value": obj.to_bytes(
                    length=ceil(obj.bit_length() / 8) + 1, byteorder="big", signed=True
                ),
            }
        if isinstance(obj, tuple):
            return {"__type__": "tuple", "value": list(obj)}
        if isinstance(obj, set):
            return {"__type__": "set", "value": list(obj)}
        if isinstance(obj, np.ndarray):
            spec = [obj.tobytes(), obj.dtype.name, list(obj.shape)]
            return {"__type__": "np.ndarray", "value": spec}
        if isinstance(obj, np.generic):
            spec = [obj.tobytes(), obj.dtype.name]
            return {"__type__": "np.generic", "value": spec}
        if isinstance(obj, torch.Tensor):
            obj = obj.cpu().numpy()
            spec = [obj.tobytes(), obj.dtype.name, list(obj.shape)]
            return {"__type__": "torch.Tensor", "value": spec}
        if isinstance(obj, Vector):
            return {"__type__": "Vector", "value": obj.coefs}
        if isinstance(obj, VectorSpec):
            return {"__type__": "VectorSpec", "value": dataclasses.asdict(obj)}
        if isinstance(obj, AuxVar):
            return {"__type__": "AuxVar", "value": json_pack(obj)}
        if isinstance(obj, MetricTypes):
            return {"__type__": "MetricTypes", "value": obj.name}
        if isinstance(obj, EncryptedAuxVar):
            return {"__type__": "EncryptedAuxVar", "value": obj.to_dict()}
        # Raise on unsupported types.
        raise FedbiomedTypeError(f"Cannot serialize object of type '{type(obj)}'.")

    @staticmethod
    def _object_hook(obj: Any) -> Any:
        """De-serialize non-default object types encoded with `_default`."""
        if not (isinstance(obj, dict) and "__type__" in obj):
            return obj
        objtype = obj["__type__"]
        if objtype == "tuple":
            return tuple(obj["value"])
        if objtype == "int":
            return int.from_bytes(obj["value"], byteorder="big", signed=True)
        if objtype == "set":
            return set(obj["value"])
        if objtype == "np.ndarray":
            data, dtype, shape = obj["value"]
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        if objtype == "np.generic":
            data, dtype = obj["value"]
            return np.frombuffer(data, dtype=dtype)[0]
        if objtype == "torch.Tensor":
            data, dtype, shape = obj["value"]
            array = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            return torch.from_numpy(array)
        if objtype == "Vector":
            return Vector.build(obj["value"])
        if objtype == "VectorSpec":
            return VectorSpec(**obj["value"])
        if objtype == "AuxVar":
            return json_unpack(obj["value"])
        if objtype == "MetricTypes":
            return MetricTypes.get_metric_type_by_name(obj["value"])
        if objtype == "EncryptedAuxVar":
            return EncryptedAuxVar.from_dict(obj["value"])
        logger.warning(
            "Encountered an object that cannot be properly deserialized."
        )
        return obj

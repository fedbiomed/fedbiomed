# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for 'fedbiomed.common.serializer.Serializer'."""

import os
import tempfile
import unittest
from typing import Any, Callable, Optional
from unittest import mock

import numpy as np
import torch

from fedbiomed.common.exceptions import FedbiomedTypeError
from fedbiomed.common.logger import logger
from fedbiomed.common.serializer import Serializer


class TestSerializer(unittest.TestCase):
    """Unit tests for 'fedbiomed.common.serializer.Serializer'."""

    def assert_serializable(
        self,
        obj: Any,
        op_equal: Optional[Callable[[Any, Any], bool]] = None,
    ) -> None:
        """Test that a given value is (de)serializable using 'Serializer'.

        Args:
            obj: Instance that needs serializing and deserializing.
            op_equal: Optional equality-checking function, overriding the
                default python equality operator when checking that the
                de-serialized object is similar to `obj`.
        """
        # Test that the object can be serialized into bytes data.
        data = Serializer.dumps(obj)
        self.assertIsInstance(data, bytes)
        # Test that the object can be de-serialized from the data.
        bis = Serializer.loads(data)
        self.assertIsInstance(bis, type(obj))
        if op_equal is None:
            self.assertEqual(bis, obj)
        else:
            self.assertTrue(op_equal(bis, obj))

    def test_serializer_01_scalar(self) -> None:
        """Test that 'Serializer' operates well on python scalar types.

        Test that int, float, bytes, string and bool values are serializable.
        """
        self.assert_serializable(0)
        self.assert_serializable(1.0)
        self.assert_serializable("string")
        self.assert_serializable(b"bytes")
        self.assert_serializable(True)

    def test_serializer_02_struct(self) -> None:
        """Test that 'Serializer' operates well on python structure types.

        Assert that tuples, lists and dict are serializable (recursively).
        """
        self.assert_serializable((4, 2))
        self.assert_serializable([4, 2.0, "0"])
        self.assert_serializable({0: "0", 1.0: [1], "2": (2,)})

    def test_serializer_03_numpy(self) -> None:
        """Test that 'Serializer' operates well on numpy arrays and scalars."""
        array = np.random.normal(size=(32, 128))
        self.assert_serializable(array, lambda x, y: bool(np.all(x == y)))
        scalar = np.mean(array)
        self.assert_serializable(scalar)

    def test_serializer_04_torch(self) -> None:
        """Test that 'Serializer' operates well on torch tensors."""
        tensor = torch.randn(size=(32, 128))
        self.assert_serializable(tensor, lambda x, y: bool(torch.all(x == y)))

    def test_serializer_05_file_dump_load(self) -> None:
        """Test that 'Serializer.load' and 'dump' work properly."""
        data = {
            "scalar": 42,
            "tuples": ((0, 1), (2, 3), (4, 5), (6, 7)),
            "tensor": torch.randn(size=(4, 8)),
            "arrays": [np.random.normal(size=(2,)) for _ in range(3)],
        }
        with tempfile.TemporaryDirectory() as folder:
            # Test that the data can be serialized to the target file.
            path = os.path.join(folder, "serialized.dat")
            self.assertFalse(os.path.isfile(path))
            Serializer.dump(data, path)
            self.assertTrue(os.path.isfile(path))
            # Test that the data can properly be recovered from the file.
            datb = Serializer.load(path)
        # Check that the recovered data is equal to the initial one.
        self.assertIsInstance(datb, dict)
        self.assertEqual(data.keys(), datb.keys())
        self.assertEqual(data["scalar"], datb["scalar"])
        self.assertEqual(data["tuples"], datb["tuples"])
        self.assertTrue(bool(torch.all(data["tensor"] == datb["tensor"])))
        self.assertTrue(
            all(np.all(a == b) for a, b in zip(data["arrays"], datb["arrays"]))
        )

    def test_serializer_06_raises_dump_error(self) -> None:
        """Test that 'Serializer.dumps' raises the expected error."""

        class UnsupportedType:
            """Empty custom type."""

        with self.assertRaises(FedbiomedTypeError):
            Serializer.dumps(UnsupportedType())

    def test_serializer_07_warns_load_error(self) -> None:
        """Test that 'Serializer.loads' logs the expected warning."""
        # Build a dict that looks like the specification for a non-standard
        # type dump (e.g. numpy array, torch tensor...).
        obj = {"__type__": "toto", "value": "mock"}
        data = Serializer.dumps(obj)
        # Test that loading such a structure logs a warning.
        with mock.patch("fedbiomed.common.serializer.logger") as p_logger:
            bis = Serializer.loads(data)
        p_logger.warning.assert_called_once()
        self.assertDictEqual(obj, bis)


if __name__ == "__main__":
    unittest.main()

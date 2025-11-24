# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from fedbiomed.common.dataset._native_dataset import NativeDataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError

# --- Minimal helper datasets -------------------------------------------------


class _ArrayDataset:
    """Dataset-like wrapper around a numpy array (len + getitem)."""

    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return self.arr[idx]


class _ListDataset:
    def __init__(self, seq):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return self.seq[idx]


class _TorchDs(TorchDataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class _TorchSupervised(TorchDataset):
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


# --- New helpers for error/branch coverage ----------------------------------


class _NoCollectionInterface:
    """Missing __len__ and/or __getitem__ to trip constructor guard."""

    pass


class _ProbeBoomDataset:
    """Has len, but __getitem__(0) raises -> probe failure in __init__."""

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        raise RuntimeError("boom during probe")


class _UnsupervisedBadFirstData:
    """First element is not convertible to torch/np -> data conversion error in CI."""

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {"bad": "data"} if idx == 0 else 1  # idx 0 triggers failure


class _SupervisedBadOnDemand(TorchDataset):
    """Index 0 good (lets CI succeed); later indices trigger conversion errors."""

    def __init__(self):
        self.items = [
            (1, 0),  # good
            ({"bad": "x"}, 1),  # bad data -> __getitem__ data conversion error
            (2, {"bad": "y"}),  # bad target -> __getitem__ target conversion error
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# --- Tests -------------------------------------------------------------------


class TestNativeDataset(unittest.TestCase):
    # ----- Existing tests (kept) -----

    def test_dataset_is_array(self):
        """Unsupervised dataset provided as a numpy array + separate target of same length."""
        X = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50])
        # Numpy arrays already implement __len__/__getitem__, so pass directly
        nd = NativeDataset(X, target=y)

        self.assertEqual(len(nd), 5)

        # Convert to Torch (lazy in __getitem__)
        nd.complete_initialization(
            controller_kwargs={}, to_format=DataReturnFormat.TORCH
        )
        d2, t2 = nd[2]
        self.assertIsInstance(d2, torch.Tensor)
        self.assertIsInstance(t2, torch.Tensor)
        self.assertEqual(d2.item(), 3)
        self.assertEqual(t2.item(), 30)

    def test_len_and_get_item_list_dataset(self):
        """Same as above but using a simple list-backed dataset (unsupervised)."""
        X = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        ds = _ListDataset(X)

        nd = NativeDataset(ds, target=y)
        self.assertEqual(len(nd), 5)

        # Ensure conversion only happens after complete_initialization
        nd.complete_initialization(
            controller_kwargs={}, to_format=DataReturnFormat.TORCH
        )
        d0, t0 = nd[0]
        self.assertIsInstance(d0, torch.Tensor)
        self.assertIsInstance(t0, torch.Tensor)
        self.assertEqual(d0.item(), 1)
        self.assertEqual(t0.item(), 10)

    def test_list_with_sklearn_format(self):
        """Torch dataset but output format is SKLEARN (lazy conversion to numpy)."""
        ds = [1, 2, 3]
        y = [0, 1, 0]

        nd = NativeDataset(ds, target=y)
        nd.complete_initialization(
            controller_kwargs={}, to_format=DataReturnFormat.SKLEARN
        )

        d1, t1 = nd[1]
        self.assertIsInstance(d1, np.ndarray)
        self.assertIsInstance(t1, np.ndarray)
        # Scalars will be 0-d arrays; cast to int to compare
        self.assertEqual(int(d1), 2)
        self.assertEqual(int(t1), 1)

    def test_supervised_dataset_detection(self):
        """Dataset that returns (data, target) tuples should not require explicit target."""
        xs = [1, 3, 5]
        ys = [0, 1, 0]
        ds = _TorchSupervised(xs, ys)

        nd = NativeDataset(ds)  # no target argument
        self.assertEqual(len(nd), 3)

        nd.complete_initialization(
            controller_kwargs={}, to_format=DataReturnFormat.TORCH
        )
        d, t = nd[2]
        self.assertIsInstance(d, torch.Tensor)
        self.assertIsInstance(t, torch.Tensor)
        self.assertEqual(d.item(), 5)
        self.assertEqual(t.item(), 0)

    def test_conflicting_target_raises(self):
        """If dataset is supervised and a target is also passed -> error."""
        xs = [1, 2]
        ys = [0, 1]
        ds = _TorchSupervised(xs, ys)
        with self.assertRaises(FedbiomedError):
            NativeDataset(ds, target=[9, 9])

    def test_target_length_mismatch_raises(self):
        """Unsupervised dataset must have target with same length."""
        ds = _ListDataset([1, 2, 3])
        with self.assertRaises(FedbiomedError):
            NativeDataset(ds, target=[0, 1])  # length mismatch

    # ----- New tests to cover missing branches/lines -----

    def test_constructor_requires_collection_interface(self):
        """Covers lines 46-49: dataset must have __len__ and __getitem__."""
        with self.assertRaises(FedbiomedError):
            NativeDataset(_NoCollectionInterface())

    def test_probe_failure_raises_fedbiomed_error(self):
        """Covers lines 57-59: failure when probing dataset[0] in __init__."""
        with self.assertRaises(FedbiomedError):
            NativeDataset(_ProbeBoomDataset())

    def test_complete_initialization_unsupervised_without_target(self):
        """
        Covers lines 101-103 branch in complete_initialization:
        unsupervised dataset with no external target -> target is None, only data validated.
        """
        ds = _ArrayDataset(np.array([1, 2, 3]))
        nd = NativeDataset(ds)  # no target
        # Should validate data conversion only and not crash
        nd.complete_initialization(
            controller_kwargs={}, to_format=DataReturnFormat.SKLEARN
        )
        d0, t0 = nd[0]
        self.assertIsInstance(d0, np.ndarray)
        self.assertIsNone(t0)

    def test_complete_initialization_data_conversion_error(self):
        """
        Covers lines 105-111: error raised when first data item cannot be converted.
        Use TORCH to force tensorization and fail on dict data.
        """
        nd = NativeDataset(_UnsupervisedBadFirstData())
        with self.assertRaises(FedbiomedError):
            nd.complete_initialization(
                controller_kwargs={}, to_format=DataReturnFormat.TORCH
            )

    def test_complete_initialization_target_conversion_error(self):
        """
        Covers lines 113-120: target conversion error during CI when external targets present.
        Provide a target whose first element is not tensorizable.
        """
        data = _ArrayDataset(np.array([1, 2]))
        bad_target = [{"bad": "t0"}, 1]
        nd = NativeDataset(data, target=bad_target)
        with self.assertRaises(FedbiomedError):
            nd.complete_initialization(
                controller_kwargs={}, to_format=DataReturnFormat.TORCH
            )

    def test_getitem_data_conversion_error(self):
        """
        Covers lines 138-143: __getitem__ conversion failure on the DATA part.
        First sample is OK so CI passes; request idx=1 to trigger failure.
        """
        nd = NativeDataset(_SupervisedBadOnDemand())
        nd.complete_initialization(
            controller_kwargs={}, to_format=DataReturnFormat.TORCH
        )
        with self.assertRaises(FedbiomedError):
            _ = nd[1]  # bad data

    def test_getitem_target_conversion_error(self):
        """
        Covers lines 144-149: __getitem__ conversion failure on the TARGET part.
        First sample is OK so CI passes; request idx=2 to trigger failure.
        """
        nd = NativeDataset(_SupervisedBadOnDemand())
        nd.complete_initialization(
            controller_kwargs={}, to_format=DataReturnFormat.TORCH
        )
        with self.assertRaises(FedbiomedError):
            _ = nd[2]  # bad target


if __name__ == "__main__":
    unittest.main()

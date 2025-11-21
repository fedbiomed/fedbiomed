import unittest
from unittest.mock import patch

import numpy as np

from fedbiomed.common.dataloader import SkLearnDataLoader
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.exceptions import FedbiomedError


class SimpleDataset(Dataset):
    """Simple dataset returning 1D numpy arrays for data and scalar targets."""

    def __init__(self, length=5):
        self._length = length

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        pass

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # data: 1D array, target: scalar (0-dim array) -> valid format
        data = np.array([idx, idx + 1], dtype=float)
        target = np.array(idx, dtype=float)
        return data, target


class DictDataset(Dataset):
    """Dataset returning dicts with a single modality."""

    def __init__(self, length=4):
        self._length = length

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        pass

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        data = {"modality": np.array([idx, idx + 1], dtype=float)}
        target = {"modality": np.array([idx], dtype=float)}
        return data, target


class BadModalityDataset(Dataset):
    """Dataset that returns an invalid dict with multiple modalities on first sample."""

    def __init__(self):
        self._length = 1

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        pass

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # multiple modalities -> _initialize should fail
        data = {
            "a": np.array([1.0]),
            "b": np.array([2.0]),
        }
        target = np.array([0.0])
        return data, target


class InconsistentShapeDataset(Dataset):
    """Dataset with inconsistent sample shapes to trigger batching errors."""

    def __init__(self):
        self._length = 2

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        pass

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if idx == 0:
            data = np.array([0.0, 1.0])  # shape (2,)
        else:
            data = np.array([0.0, 1.0, 2.0])  # shape (3,) -> inconsistent for vstack
        target = np.array(idx, dtype=float)
        return data, target


class BadTargetDictDataset(Dataset):
    """Dataset with valid first sample but bad target keys in second sample."""

    def __init__(self):
        self._length = 2

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        pass

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if idx == 0:
            data = {"modality": np.array([0.0, 1.0], dtype=float)}
            target = {"modality": np.array([0.0], dtype=float)}
        else:
            data = {"modality": np.array([2.0, 3.0], dtype=float)}
            # mismatching key -> should trigger target format error in _check_sample_format
            target = {"other_modality": np.array([1.0], dtype=float)}
        return data, target


class FailingGetItemDataset(Dataset):
    """Dataset whose __getitem__ raises an exception to test wrapping in FedbiomedError."""

    def __init__(self):
        self._length = 3

    def complete_initialization(self):
        pass

    def _apply_transforms(self, sample):
        pass

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        raise RuntimeError("Synthetic failure in __getitem__")


class TestSkLearnDataLoader(unittest.TestCase):
    def setUp(self):
        self.dataset = SimpleDataset(length=5)

    def test_init_invalid_arguments(self):
        # bad batch_size type
        with self.assertRaises(FedbiomedError):
            SkLearnDataLoader(self.dataset, batch_size="2")

        # non-positive batch_size
        with self.assertRaises(FedbiomedError):
            SkLearnDataLoader(self.dataset, batch_size=0)

        # bad shuffle type
        with self.assertRaises(FedbiomedError):
            SkLearnDataLoader(self.dataset, shuffle="yes")

        # bad drop_last type
        with self.assertRaises(FedbiomedError):
            SkLearnDataLoader(self.dataset, drop_last="no")

    def test_len_and_remainder_no_drop_last(self):
        loader = SkLearnDataLoader(self.dataset, batch_size=2, drop_last=False)
        # 5 samples, batch_size 2 -> 3 batches (2 + 2 + 1)
        self.assertEqual(loader.n_remainder_samples(), 1)
        self.assertEqual(len(loader), 3)

    def test_len_and_remainder_drop_last(self):
        loader = SkLearnDataLoader(self.dataset, batch_size=2, drop_last=True)
        # 5 samples, batch_size 2, drop_last -> 2 full batches only
        self.assertEqual(loader.n_remainder_samples(), 1)
        self.assertEqual(len(loader), 2)

    def test_iteration_simple_dataset(self):
        batch_size = 2
        loader = SkLearnDataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        seen_indices = []
        for batch_data, batch_target in loader:
            self.assertIsInstance(batch_data, np.ndarray)
            self.assertIsInstance(batch_target, np.ndarray)
            self.assertEqual(batch_data.ndim, 2)

            for i in range(batch_data.shape[0]):
                seen_indices.append(int(batch_data[i, 0]))

        self.assertListEqual(seen_indices, list(range(5)))

    @patch("fedbiomed.common.dataloader.SkLearnDataLoader.shuffle")
    def test_shuffle_calls_numpy_shuffle(self, mock_shuffle):
        loader = SkLearnDataLoader(
            self.dataset, batch_size=2, shuffle=True, drop_last=False
        )

        _ = iter(loader)
        mock_shuffle.assert_called_once()

    def test_iteration_dict_dataset(self):
        dict_dataset = DictDataset(length=4)
        loader = SkLearnDataLoader(
            dict_dataset, batch_size=2, shuffle=False, drop_last=False
        )

        for batch_data, batch_target in loader:
            self.assertIsInstance(batch_data, np.ndarray)
            self.assertIsInstance(batch_target, np.ndarray)
            self.assertEqual(batch_data.ndim, 2)

    def test_initialize_raises_on_multiple_modalities(self):
        bad_dataset = BadModalityDataset()
        loader = SkLearnDataLoader(
            bad_dataset, batch_size=1, shuffle=False, drop_last=False
        )

        it = iter(loader)
        with self.assertRaises(FedbiomedError):
            next(it)

    def test_inconsistent_shapes_raise_fedbiomederror(self):
        bad_dataset = InconsistentShapeDataset()
        loader = SkLearnDataLoader(
            bad_dataset, batch_size=2, shuffle=False, drop_last=False
        )

        it = iter(loader)
        with self.assertRaises(FedbiomedError):
            next(it)

    def test_bad_target_dict_keys_raise_fedbiomederror(self):
        bad_target_dataset = BadTargetDictDataset()
        loader = SkLearnDataLoader(
            bad_target_dataset, batch_size=1, shuffle=False, drop_last=False
        )

        it = iter(loader)
        # first batch should be fine
        _ = next(it)
        # second batch should fail due to mismatching target keys
        with self.assertRaises(FedbiomedError):
            next(it)

    def test_dataset_getitem_exception_wrapped_in_fedbiomederror(self):
        failing_dataset = FailingGetItemDataset()
        loader = SkLearnDataLoader(
            failing_dataset, batch_size=1, shuffle=False, drop_last=False
        )

        it = iter(loader)
        with self.assertRaises(FedbiomedError):
            next(it)

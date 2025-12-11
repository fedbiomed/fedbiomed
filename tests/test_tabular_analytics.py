# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from fedbiomed.common.dataset import TabularDataset


class MockTabularDataset(TabularDataset):
    """Mock dataset that uses TabularAnalytics mixin

    This mock stores pre-built data/target lists and provides simple
    __len__ and __getitem__ so that the TabularAnalytics.mean() mixin
    can operate on it without a controller.
    """

    def __init__(self, data_list, target_list, input_columns=None, target_columns=None):
        """Initialize the mock dataset.

        Args:
            data_list: list of numpy arrays or torch tensors
            target_list: list of numpy arrays or torch tensors
            input_columns: list of input column names (default: ['feature_0', 'feature_1', ...])
            target_columns: list of target column names (default: ['target_0', 'target_1', ...])
        """

        # Store samples directly
        # TabularAnalytics relies only on __len__/__getitem__

        if len(data_list) != len(target_list):
            raise ValueError("data_list and target_list must have the same length")

        self.data_list = data_list
        self.target_list = target_list

        # Determine input/target column dimensions from first sample if available
        if len(data_list) > 0:
            first_data = data_list[0]
            first_target = target_list[0]
            # Handle both arrays and scalars
            data_size = (
                first_data.shape[0]
                if hasattr(first_data, "shape") and len(first_data.shape) > 0
                else 1
            )
            target_size = (
                first_target.shape[0]
                if hasattr(first_target, "shape") and len(first_target.shape) > 0
                else 1
            )
        else:
            data_size = 0
            target_size = 0

        # Set default column names if not provided
        self._input_columns = input_columns or [
            f"feature_{i}" for i in range(data_size)
        ]
        self._target_columns = target_columns or [
            f"target_{i}" for i in range(target_size)
        ]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.target_list[idx]


@pytest.fixture
def mock_dataset_with_numpy():
    """Fixture for mock dataset with numpy arrays"""
    data_samples = [
        np.array([25.0, 1.0]),
        np.array([30.0, 3.0]),
        np.array([35.0, 5.0]),
        np.array([40.0, 7.0]),
        np.array([45.0, 9.0]),
    ]
    target_samples = [
        np.array([50000.0]),
        np.array([60000.0]),
        np.array([70000.0]),
        np.array([80000.0]),
        np.array([90000.0]),
    ]
    return MockTabularDataset(
        data_samples,
        target_samples,
        input_columns=["age", "weight"],
        target_columns=["salary"],
    )


@pytest.fixture
def mock_dataset_with_single_sample():
    """Fixture for mock dataset with single sample"""
    data_samples = [np.array([10.0, 20.0])]
    target_samples = [np.array([100.0])]
    return MockTabularDataset(
        data_samples,
        target_samples,
        input_columns=["age", "weight"],
        target_columns=["salary"],
    )


@pytest.fixture
def mock_dataset_empty():
    """Fixture for empty mock dataset"""
    return MockTabularDataset([], [])


def test_mean_with_numpy_arrays(mock_dataset_with_numpy):
    """Test mean calculation with numpy arrays"""
    result = mock_dataset_with_numpy.mean()

    # Check structure
    assert isinstance(result, dict)
    assert set(result.keys()) == {"age", "weight", "salary"}

    # Verify values are correct
    assert np.isclose(result["age"], 35.0)
    assert np.isclose(result["weight"], 5.0)
    assert np.isclose(result["salary"], 70000.0)


def test_mean_with_single_sample(mock_dataset_with_single_sample):
    """Test mean with only one sample"""
    result = mock_dataset_with_single_sample.mean()

    assert np.isclose(result["age"], 10.0)
    assert np.isclose(result["weight"], 20.0)
    assert np.isclose(result["salary"], 100.0)


def test_mean_with_empty_dataset(mock_dataset_empty):
    """Test mean with empty dataset"""
    result = mock_dataset_empty.mean()

    assert all(v is None for v in result.values())


# -------------------- Torch tests (mirror numpy tests) --------------------


@pytest.fixture
def mock_dataset_with_torch():
    """Fixture for mock dataset with torch tensors"""
    data_samples = [
        torch.tensor([25.0, 1.0]),
        torch.tensor([30.0, 3.0]),
        torch.tensor([35.0, 5.0]),
        torch.tensor([40.0, 7.0]),
        torch.tensor([45.0, 9.0]),
    ]
    target_samples = [
        torch.tensor([50000.0]),
        torch.tensor([60000.0]),
        torch.tensor([70000.0]),
        torch.tensor([80000.0]),
        torch.tensor([90000.0]),
    ]
    return MockTabularDataset(
        data_samples,
        target_samples,
        input_columns=["age", "weight"],
        target_columns=["salary"],
    )


@pytest.fixture
def mock_dataset_with_single_torch_sample():
    data_samples = [torch.tensor([10.0, 20.0])]
    target_samples = [torch.tensor([100.0])]
    return MockTabularDataset(
        data_samples,
        target_samples,
        input_columns=["age", "weight"],
        target_columns=["salary"],
    )


@pytest.fixture
def mock_dataset_torch_empty():
    return MockTabularDataset([], [])


def test_mean_with_torch_tensors(mock_dataset_with_torch):
    result = mock_dataset_with_torch.mean()

    # Check structure
    assert isinstance(result, dict)
    assert set(result.keys()) == {"age", "weight", "salary"}

    # Verify values are correct
    assert torch.isclose(result["age"], torch.tensor(35.0))
    assert torch.isclose(result["weight"], torch.tensor(5.0))
    assert torch.isclose(result["salary"], torch.tensor(70000.0))


def test_mean_with_single_torch_sample(mock_dataset_with_single_torch_sample):
    result = mock_dataset_with_single_torch_sample.mean()
    assert torch.isclose(result["age"], torch.tensor(10.0))
    assert torch.isclose(result["weight"], torch.tensor(20.0))
    assert torch.isclose(result["salary"], torch.tensor(100.0))


def test_mean_with_empty_torch_dataset(mock_dataset_torch_empty):
    result = mock_dataset_torch_empty.mean()
    assert all(v is None for v in result.values())

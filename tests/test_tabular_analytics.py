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

    def __init__(self, data_list, target_list):
        """Initialize the mock dataset.

        Args:
            data_list: list of numpy arrays or torch tensors
            target_list: list of numpy arrays or torch tensors
        """

        # Store samples directly
        # TabularAnalytics relies only on __len__/__getitem__

        if len(data_list) != len(target_list):
            raise ValueError("data_list and target_list must have the same length")

        self.data_list = data_list
        self.target_list = target_list

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
    return MockTabularDataset(data_samples, target_samples)


@pytest.fixture
def mock_dataset_with_single_sample():
    """Fixture for mock dataset with single sample"""
    data_samples = [np.array([10.0, 20.0])]
    target_samples = [np.array([100.0])]
    return MockTabularDataset(data_samples, target_samples)


@pytest.fixture
def mock_dataset_empty():
    """Fixture for empty mock dataset"""
    return MockTabularDataset([], [])


def test_mean_with_numpy_arrays(mock_dataset_with_numpy):
    """Test mean calculation with numpy arrays"""
    result = mock_dataset_with_numpy.mean()

    # Check structure
    assert isinstance(result, dict)
    assert set(result.keys()) == {"data", "target"}

    # Verify values are correct
    assert np.allclose(result["data"], np.array([35.0, 5.0]))
    assert np.allclose(result["target"], np.array([70000.0]))

    assert isinstance(result["data"], np.ndarray)
    assert isinstance(result["target"], np.ndarray)


def test_mean_with_single_sample(mock_dataset_with_single_sample):
    """Test mean with only one sample"""
    result = mock_dataset_with_single_sample.mean()

    assert np.allclose(result["data"], np.array([10.0, 20.0]))
    assert np.allclose(result["target"], np.array([100.0]))


def test_mean_with_empty_dataset(mock_dataset_empty):
    """Test mean with empty dataset"""
    result = mock_dataset_empty.mean()

    assert result["data"] is None
    assert result["target"] is None


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
    return MockTabularDataset(data_samples, target_samples)


@pytest.fixture
def mock_dataset_with_single_torch_sample():
    data_samples = [torch.tensor([10.0, 20.0])]
    target_samples = [torch.tensor([100.0])]
    return MockTabularDataset(data_samples, target_samples)


@pytest.fixture
def mock_dataset_torch_empty():
    return MockTabularDataset([], [])


def test_mean_with_torch_tensors(mock_dataset_with_torch):
    result = mock_dataset_with_torch.mean()

    # Check structure
    assert isinstance(result, dict)
    assert set(result.keys()) == {"data", "target"}

    # Verify values are correct
    assert torch.allclose(result["data"], torch.tensor([35.0, 5.0]))
    assert torch.allclose(result["target"], torch.tensor([70000.0]))

    assert isinstance(result["data"], torch.Tensor)
    assert isinstance(result["target"], torch.Tensor)


def test_mean_with_single_torch_sample(mock_dataset_with_single_torch_sample):
    result = mock_dataset_with_single_torch_sample.mean()
    assert torch.allclose(result["data"], torch.tensor([10.0, 20.0]))
    assert torch.allclose(result["target"], torch.tensor([100.0]))


def test_mean_with_empty_torch_dataset(mock_dataset_torch_empty):
    result = mock_dataset_torch_empty.mean()
    assert result["data"] is None and result["target"] is None

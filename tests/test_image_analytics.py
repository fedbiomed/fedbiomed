# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from fedbiomed.common.analytics import ImageAnalytics
from fedbiomed.common.exceptions import FedbiomedError


class MockImageDataset(ImageAnalytics):
    """Mock image dataset class that implements ImageAnalytics for testing"""

    def __init__(self, image_data_list):
        """
        Args:
            image_data_list: List of images where image is torch.Tensor or np.ndarray
        """
        self._image_data_list = image_data_list

    def __len__(self):
        return len(self._image_data_list)

    def __getitem__(self, idx):
        return self._image_data_list[idx], None


@pytest.fixture
def dataset():
    """Simple image dataset with numpy arrays"""
    images = [
        np.array([20.0, 50.0, 80.0]),
        np.array([80.0, 60.0, 100.0]),
        np.array([50.0, 10.0, 30.0]),
    ]
    return MockImageDataset(images)


# ==================== MINMAX TESTS ====================


def test_image_analytics_min_max(dataset):
    """Test minmax calculation"""
    result = dataset.min_max()

    assert isinstance(result, dict)
    assert "pixel_values" in result
    min_val, max_val = result["pixel_values"]
    assert min_val == pytest.approx(10.0)
    assert max_val == pytest.approx(100.0)


# ==================== HISTOGRAM TESTS ====================


def test_image_analytics_histogram(dataset):
    """Test histogram with array bin_edges on both numpy and torch datasets"""
    bin_edges = np.array([20, 40, 60, 80, 100])
    result = dataset.histogram(bin_edges)

    assert isinstance(result, dict)
    assert "pixel_values" in result
    counts = result["pixel_values"]
    assert isinstance(counts, np.ndarray)
    assert len(counts) == 4  # 4 bins
    assert np.sum(counts) == 9  # 3 images x 3 pixels each


def test_image_analytics_histogram_invalid_bin_edges(dataset):
    """Test histogram with invalid bin_edges"""
    # Non-1D array should raise error
    with pytest.raises(FedbiomedError):
        bin_edges = np.array([[1, 2], [3, 4]])
        dataset.histogram(bin_edges)


def test_image_analytics_histogram_non_increasing_edges(dataset):
    """Test histogram with non-increasing bin_edges"""
    # Non-increasing edges should raise error
    with pytest.raises(FedbiomedError):
        bin_edges = np.array([120.5, 100.5, 70.5, 40.5, 9.5])
        dataset.histogram(bin_edges)


# ==================== QUANTILE TESTS ====================


def test_image_analytics_quantile_single(dataset):
    """Test quantile with single quantile value"""
    bin_edges = np.array([20, 40, 60, 80, 100])
    result = dataset.quantile(bin_edges, q=0.5)["pixel_values"]

    assert isinstance(result, dict)
    assert 0.5 in result
    assert isinstance(result[0.5], float)
    assert 20 <= result[0.5] <= 100


def test_image_analytics_quantile_multiple(dataset):
    """Test quantile with multiple quantile values"""
    bin_edges = np.array([20, 40, 60, 80, 100])
    result = dataset.quantile(bin_edges, q=[0.25, 0.5, 0.75])["pixel_values"]

    assert isinstance(result, dict)
    assert 0.25 in result
    assert 0.5 in result
    assert 0.75 in result
    for q_val in [0.25, 0.5, 0.75]:
        assert isinstance(result[q_val], float)
        assert 20 <= result[q_val] <= 100


def test_image_analytics_quantile_invalid_q(dataset):
    """Test quantile with invalid q values"""
    bin_edges = np.array([20, 40, 60, 80, 100])

    # q > 1 should raise error
    with pytest.raises(FedbiomedError):
        dataset.quantile(bin_edges, q=1.5)

    # q < 0 should raise error
    with pytest.raises(FedbiomedError):
        dataset.quantile(bin_edges, q=-0.1)

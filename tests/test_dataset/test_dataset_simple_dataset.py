from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from fedbiomed.common.dataset import ImageFolderDataset
from fedbiomed.common.dataset._simple_dataset import _ImageLabelDataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedValueError


@pytest.fixture
def mock_controller():
    """Fixture for a fake ImageFolderController with minimal behavior."""
    controller = MagicMock()
    controller.get_sample.side_effect = lambda index: {
        "data": Image.new("RGB", (28, 28), color=128),
        "target": 1,
    }
    return controller


@pytest.fixture(params=[DataReturnFormat.SKLEARN, DataReturnFormat.TORCH])
def dataset_with_mock_controller(request, mock_controller, tmp_path):
    """ImageFolderDataset initialized with either sklearn or torch return format."""
    dataset = ImageFolderDataset()
    dataset._controller_cls = lambda **kwargs: mock_controller
    dataset.complete_initialization(
        controller_kwargs={"root": tmp_path},
        to_format=request.param,
    )
    return dataset


# === Tests for Dataset ===


def test_to_format_setter_getter():
    dataset = ImageFolderDataset()
    dataset._to_format = None
    dataset.to_format = DataReturnFormat.SKLEARN
    assert dataset.to_format == DataReturnFormat.SKLEARN


def test_to_format_invalid_type():
    dataset = ImageFolderDataset()
    with pytest.raises(FedbiomedValueError):
        dataset.to_format = "not an enum"


def test_init_controller_invalid_kwargs_type():
    dataset = ImageFolderDataset()
    with pytest.raises(FedbiomedError):
        dataset._init_controller(controller_kwargs="not a dict")


def test_init_controller_instantiation_failure(tmp_path):
    dataset = ImageFolderDataset()
    dataset._controller_cls = lambda **kwargs: (_ for _ in ()).throw(Exception("Fail"))
    with pytest.raises(FedbiomedError):
        dataset._init_controller({"root": tmp_path})


# === Tests for SimpleDataset ===


def test_simple_dataset_cannot_be_instantiated():
    with pytest.raises(FedbiomedError) as excinfo:
        _ = _ImageLabelDataset()
    assert "cannot be instantiated directly" in str(excinfo.value)


def test_transform_invalid_type():
    with pytest.raises(FedbiomedValueError):
        _ = ImageFolderDataset(transform="not callable")


def test_target_transform_invalid_type():
    with pytest.raises(FedbiomedValueError):
        _ = ImageFolderDataset(target_transform="not callable")


def test_validate_transform_success_default(dataset_with_mock_controller):
    """if transform is None, the identity function is used by default"""
    dataset = dataset_with_mock_controller
    sample = dataset._controller.get_sample(0)
    dataset._validate_format_and_transformations(
        data=sample["data"],
        transform=dataset._transform,
    )


def test_validate_transform_succeeds(dataset_with_mock_controller):
    dataset = dataset_with_mock_controller
    sample = dataset._controller.get_sample(0)
    transform = (
        transforms.Normalize((0.1307,), (0.3081,))
        if dataset_with_mock_controller.to_format == DataReturnFormat.TORCH
        else lambda x: x.astype(np.float32) / 255
    )
    dataset._validate_format_and_transformations(
        data=sample["data"], transform=transform
    )


def test_validate_transform_fails(dataset_with_mock_controller):
    dataset = dataset_with_mock_controller
    sample = dataset._controller.get_sample(0)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    with pytest.raises(FedbiomedError):
        dataset._validate_format_and_transformations(
            data=sample["data"], transform=transform
        )


def test_validate_target_transform_success_default(dataset_with_mock_controller):
    """if transform is None, the identity function is used by default"""
    dataset = dataset_with_mock_controller
    sample = dataset._controller.get_sample(0)
    dataset._validate_format_and_transformations(
        data=sample["target"],
        transform=dataset._target_transform,
    )


def test_validate_target_transform_fails(dataset_with_mock_controller):
    dataset = dataset_with_mock_controller
    sample = dataset._controller.get_sample(0)
    target_transform = (
        torch.tensor
        if dataset_with_mock_controller.to_format == DataReturnFormat.SKLEARN
        else np.array
    )
    with pytest.raises(FedbiomedError):
        dataset._validate_format_and_transformations(
            data=sample["target"], transform=target_transform
        )


def test_apply_transforms_success(dataset_with_mock_controller):
    dataset = dataset_with_mock_controller
    dataset._transform = (
        transforms.Normalize((0.1307,), (0.3081,))
        if dataset_with_mock_controller.to_format == DataReturnFormat.TORCH
        else lambda x: x.astype(np.float32) / 255
    )
    data, target = dataset[0]
    assert isinstance(data, dataset_with_mock_controller._to_format.value)
    assert isinstance(target, dataset_with_mock_controller._to_format.value)


# === Tests for complete_initialization ===


def test_complete_initialization_missing_keys(tmp_path):
    # incomplete controller
    mock_controller = MagicMock()
    mock_controller.get_sample.side_effect = lambda index: {
        "data": Image.new("RGB", (28, 28), color=128),
    }
    dataset = ImageFolderDataset()
    dataset._controller_cls = lambda **kwargs: mock_controller
    with pytest.raises(KeyError):
        dataset.complete_initialization({"root": tmp_path}, DataReturnFormat.SKLEARN)


def test_complete_initialization_success(dataset_with_mock_controller):
    dataset = dataset_with_mock_controller
    data, target = dataset[0]
    assert isinstance(data, dataset_with_mock_controller._to_format.value)
    assert isinstance(target, dataset_with_mock_controller._to_format.value)


def test_getitem(dataset_with_mock_controller):
    sample = dataset_with_mock_controller[0]
    assert all(
        isinstance(item, dataset_with_mock_controller._to_format.value)
        for item in sample
    )

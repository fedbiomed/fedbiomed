from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from fedbiomed.common.dataset import ImageFolderDataset
from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    DatasetDataItemModality,
    DataType,
)
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def mock_imagefolder(mocker):
    mock_dataset = MagicMock()
    mock_dataset.class_to_idx = {"class_0": 0}
    mock_dataset.samples = [
        ("/fake/class_0/image_0.jpg", 0),
        ("/fake/class_0/image_1.jpg", 1),
        ("/fake/class_0/image_2.jpg", 2),
    ]
    mock_dataset.loader = lambda path: Image.fromarray(
        np.zeros((28, 28, 3), dtype=np.uint8),
        mode="RGB",
    )
    return mocker.patch(
        "fedbiomed.common.dataset_controller._image_folder_controller.datasets.ImageFolder",
        return_value=mock_dataset,
    )


@pytest.mark.parametrize(
    "format_type", [DataReturnFormat.DEFAULT, DataReturnFormat.TORCH]
)
def test_getitem_returns_expected_format(
    mocker, mock_imagefolder, tmp_path, format_type
):
    dataset = ImageFolderDataset(root=tmp_path)
    dataset._to_format = format_type
    data_item, target_item = dataset[0]

    if format_type == DataReturnFormat.DEFAULT:
        assert isinstance(data_item["data"], DatasetDataItemModality)
        assert data_item["data"].modality_name == "data"
        assert data_item["data"].type == DataType.IMAGE
        assert np.array_equal(data_item["data"].data, np.zeros((28, 28, 3)))
        assert np.array_equal(target_item["target"], np.array(0))
    else:
        assert isinstance(data_item["data"], torch.Tensor)
        assert torch.equal(data_item["data"], torch.zeros((3, 28, 28)))
        assert torch.equal(target_item["target"], torch.tensor(0))


def test_getitem_raises_on_unsupported_format(mocker, mock_imagefolder, tmp_path):
    dataset = ImageFolderDataset(root=tmp_path)
    dataset._to_format = DataReturnFormat.SKLEARN

    with pytest.raises(FedbiomedError):
        dataset[0]

from unittest.mock import MagicMock
from urllib.error import URLError

import numpy as np
import pytest
from PIL import Image

from fedbiomed.common.dataset_controller import ImageFolderController
from fedbiomed.common.dataset_types import DataType
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
    mock_dataset.loader = mock_dataset.loader = lambda path: Image.fromarray(
        np.zeros((28, 28, 3), dtype=np.uint8),
        mode="RGB",
    )
    return mocker.patch(
        "fedbiomed.common.dataset_controller._image_folder_controller.datasets.ImageFolder",
        return_value=mock_dataset,
    )


def test_init_loads_dataset(tmp_path, mocker, mock_imagefolder):
    controller = ImageFolderController(root=tmp_path)
    assert isinstance(controller._class_to_idx, dict)
    assert isinstance(controller._samples, list)
    assert callable(controller._loader)


def test_get_nontransformed_item_success(tmp_path, mocker, mock_imagefolder):
    controller = ImageFolderController(root=tmp_path)
    data, target = controller._get_nontransformed_item(0)
    assert "data" in data
    assert data["data"].size == (28, 28)
    assert len(data["data"].getbands()) == 3
    assert "target" in target
    assert isinstance(target["target"], int)


def test_get_nontransformed_item_error(tmp_path, mocker, mock_imagefolder):
    controller = ImageFolderController(root=tmp_path)
    controller._loader = MagicMock(side_effect=Exception("load error"))
    with pytest.raises(FedbiomedError):
        controller._get_nontransformed_item(0)


def test_get_dataset_data_meta(tmp_path, mocker, mock_imagefolder):
    controller = ImageFolderController(root=tmp_path)
    meta = controller._get_dataset_data_meta()
    assert "data" in meta.data
    assert meta.data["data"].modality_name == "data"
    assert meta.data["data"].type == DataType.IMAGE
    assert meta.data["data"].shape == (28, 28, 3)
    assert "target" in meta.target
    assert meta.target["target"].modality_name == "target"
    assert meta.target["target"].type == DataType.TABULAR
    assert meta.target["target"].shape == ()
    assert meta.len == 3


def test_download_mednist_failure(tmp_path, mocker, mock_imagefolder):
    controller = ImageFolderController(root=tmp_path)
    mocker.patch(
        "fedbiomed.common.dataset_controller._image_folder_controller.urlretrieve",
        side_effect=URLError("Fail"),
    )
    with pytest.raises(FedbiomedError):
        controller._download_mednist()

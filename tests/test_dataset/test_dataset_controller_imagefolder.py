from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

from fedbiomed.common.dataset_controller import ImageFolderController
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def mock_folder_methods():
    with (
        mock.patch(
            "fedbiomed.common.dataset_controller._image_folder_controller.folder.find_classes"
        ) as mock_find_classes,
        mock.patch(
            "fedbiomed.common.dataset_controller._image_folder_controller.folder.make_dataset"
        ) as mock_make_dataset,
        mock.patch(
            "fedbiomed.common.dataset_controller._image_folder_controller.folder.default_loader"
        ) as mock_default_loader,
    ):
        mock_find_classes.return_value = (["class_a"], {"class_a": 0})
        mock_make_dataset.return_value = [("image_a.jpg", 0), ("image_b.jpg", 1)]
        mock_default_loader.side_effect = lambda path: Image.new(
            "L", (28, 28), color=128
        )

        yield {
            "find_classes": mock_find_classes,
            "make_dataset": mock_make_dataset,
            "default_loader": mock_default_loader,
        }


def test_init_success(mock_folder_methods, tmp_path):
    controller = ImageFolderController(tmp_path)

    assert controller.root == Path(tmp_path)
    assert controller._class_to_idx == {"class_a": 0}
    assert len(controller._samples) == 2
    assert controller._controller_kwargs == {
        "root": str(tmp_path),
    }


def test_init_failure_find_classes(mock_folder_methods, tmp_path):
    with mock.patch(
        "fedbiomed.common.dataset_controller._image_folder_controller.folder.find_classes",
        side_effect=Exception("fail"),
    ):
        with pytest.raises(FedbiomedError):
            _ = ImageFolderController(tmp_path)


def test_init_failure_make_dataset(mock_folder_methods, tmp_path):
    with mock.patch(
        "fedbiomed.common.dataset_controller._image_folder_controller.folder.make_dataset",
        side_effect=Exception("fail"),
    ):
        with pytest.raises(FedbiomedError):
            _ = ImageFolderController(tmp_path)


def test_get_sample_success(mocker, mock_folder_methods, tmp_path):
    controller = ImageFolderController(tmp_path)
    controller._loader = lambda path: Image.new("L", (28, 28), color=255)
    sample = controller.get_sample(0)

    assert isinstance(sample["data"], Image.Image)
    assert isinstance(sample["target"], int)


def testget_sample_failure_loader(mock_folder_methods, tmp_path):
    with mock.patch.object(ImageFolderController, "_loader") as mock_loader:
        mock_loader.side_effect = Exception("Loader failed")

        controller = ImageFolderController(tmp_path)
        with pytest.raises(FedbiomedError) as excinfo:
            controller.get_sample(0)
        assert "Failed to retrieve item at index" in str(excinfo.value)


def test_len_and_shape(mock_folder_methods, tmp_path):
    controller = ImageFolderController(tmp_path)
    controller._loader = lambda path: Image.new("L", (28, 28), color=255)
    assert len(controller) == 2
    shape = controller.shape()
    assert isinstance(shape, dict)
    assert all(item in shape for item in ["data", "target"])
    assert shape["data"] == {"size": (28, 28), "mode": "L"}
    assert shape["target"] == 1

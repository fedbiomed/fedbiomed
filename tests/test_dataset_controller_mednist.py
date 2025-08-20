from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

from fedbiomed.common.dataset_controller import MedNistController
from fedbiomed.common.dataset_controller._mednist_controller import download_mednist
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def mock_torch_mednist(mocker):
    mock_mednist = mocker.patch(
        "fedbiomed.common.dataset_controller._mednist_controller.ImageFolder"
    )
    instance = mock_mednist.return_value
    instance.__getitem__.return_value = (
        Image.new("L", (28, 28), color=128),
        0,
    )
    instance.__len__.return_value = 10
    return instance


def test_mnist_controller_init_success(mocker, mock_torch_mednist, tmp_path):
    with mock.patch.object(Path, "exists", return_value=True):
        controller = MedNistController(root=tmp_path)
        assert controller._controller_kwargs["root"] == str(tmp_path)


def test_get_nontransformed_item(mocker, mock_torch_mednist, tmp_path):
    with mock.patch.object(Path, "exists", return_value=True):
        controller = MedNistController(root=tmp_path)
        sample = controller._get_nontransformed_item(index=1)
        assert isinstance(sample["data"], Image.Image)
        assert isinstance(sample["target"], int)


def test_len_and_shape(mocker, mock_torch_mednist, tmp_path):
    with mock.patch.object(Path, "exists", return_value=True):
        controller = MedNistController(tmp_path)
        assert len(controller) == 10
        shape = controller.shape()
        assert isinstance(shape, dict)
        assert all(item in shape for item in ["data", "target"])
        assert shape["data"] == {"size": (28, 28), "mode": "L"}
        assert shape["target"] == 1


def test_raises_on_dataset_failure(mocker, mock_torch_mednist, tmp_path):
    with mock.patch.object(Path, "exists", return_value=True):
        mocker.patch(
            "fedbiomed.common.dataset_controller._mednist_controller.ImageFolder",
            side_effect=RuntimeError("Fail"),
        )
        with pytest.raises(FedbiomedError):
            MedNistController(root=tmp_path)


# === download_mednist ===


@mock.patch("fedbiomed.common.dataset_controller._mednist_controller.urlretrieve")
@mock.patch("fedbiomed.common.dataset_controller._mednist_controller.tarfile.open")
@mock.patch("fedbiomed.common.dataset_controller._mednist_controller.os.remove")
def test_download_mednist_success(
    mock_remove, mock_tarfile_open, mock_urlretrieve, tmp_path
):
    # Mock tarfile object
    mock_tar = mock.MagicMock()
    mock_tarfile_open.return_value.__enter__.return_value = mock_tar

    # Call the function
    download_mednist(tmp_path)

    # Assert download, extraction, and cleanup were called
    mock_urlretrieve.assert_called_once()
    mock_tar.extractall.assert_called_once_with(tmp_path)
    mock_remove.assert_called_once_with(tmp_path / "MedNIST.tar.gz")


def test_download_mednist_invalid_path_type():
    with pytest.raises(FedbiomedError) as excinfo:
        download_mednist("not_a_path")
    assert "Expected `root` to be of type `Path`" in str(excinfo.value)


@mock.patch(
    "fedbiomed.common.dataset_controller._mednist_controller.urlretrieve",
    side_effect=Exception("Download failed"),
)
def test_download_mednist_download_error(mock_urlretrieve, tmp_path):
    with pytest.raises(FedbiomedError) as excinfo:
        download_mednist(tmp_path)
    assert "error raised while downloading" in str(excinfo.value)


@mock.patch("fedbiomed.common.dataset_controller._mednist_controller.urlretrieve")
@mock.patch(
    "fedbiomed.common.dataset_controller._mednist_controller.tarfile.open",
    side_effect=Exception("Extraction failed"),
)
def test_download_mednist_extraction_error(
    mock_tarfile_open, mock_urlretrieve, tmp_path
):
    with pytest.raises(FedbiomedError) as excinfo:
        download_mednist(tmp_path)
    assert "error raised while extracting" in str(excinfo.value)

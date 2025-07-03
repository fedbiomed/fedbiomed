import pytest
import torch

from fedbiomed.common.dataset_controller import MnistController
from fedbiomed.common.dataset_types import DataType
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def mock_mnist_controller(mocker):
    mock_dataset = mocker.patch(
        "fedbiomed.common.dataset_controller._mnist_controller.datasets.MNIST"
    )
    instance = mock_dataset.return_value
    instance.data = torch.zeros((10, 28, 28))
    instance.targets = torch.arange(10)
    return instance


def test_init_loads_dataset(mocker, mock_mnist_controller, tmp_path):
    controller = MnistController(root=tmp_path)
    assert controller._data.shape == (10, 28, 28)
    assert torch.equal(controller._targets, torch.arange(10))


def test_get_nontransformed_item_returns_data(mocker, mock_mnist_controller, tmp_path):
    controller = MnistController(root=tmp_path)
    data, target = controller._get_nontransformed_item(index=1)

    assert isinstance(data["data"], torch.Tensor)
    assert data["data"].shape == (28, 28)
    assert target["target"] == 1


def test_dataset_data_meta_structure(mocker, mock_mnist_controller, tmp_path):
    controller = MnistController(root=tmp_path)
    meta = controller._get_dataset_data_meta()

    assert meta.len == 10
    assert "data" in meta.data
    assert meta.data["data"].modality_name == "data"
    assert meta.data["data"].type == DataType.IMAGE
    assert meta.data["data"].shape == (28, 28)
    assert "target" in meta.target
    assert meta.target["target"].modality_name == "target"
    assert meta.target["target"].type == DataType.TABULAR
    assert meta.target["target"].shape == (1,)
    assert meta.len == 10


def test_raises_on_dataset_failure(mocker, mock_mnist_controller, tmp_path):
    mocker.patch(
        "fedbiomed.common.dataset_controller._mnist_controller.datasets.MNIST",
        side_effect=RuntimeError("Fail"),
    )
    with pytest.raises(FedbiomedError):
        MnistController(root=tmp_path)

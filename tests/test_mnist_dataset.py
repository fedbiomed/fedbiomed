import numpy as np
import pandas as pd
import pytest
import torch

from fedbiomed.common.dataset import MnistDataset
from fedbiomed.common.dataset_types import (
    DataReturnFormat,
    DatasetDataItemModality,
    DataType,
)
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


@pytest.mark.parametrize(
    "format_type", [DataReturnFormat.DEFAULT, DataReturnFormat.TORCH]
)
def test_getitem_returns_expected_format(
    mocker, mock_mnist_controller, tmp_path, format_type
):
    dataset = MnistDataset(root=tmp_path)
    dataset._to_format = format_type
    data_item, target_item = dataset[0]

    if format_type == DataReturnFormat.DEFAULT:
        assert isinstance(data_item["data"], DatasetDataItemModality)
        assert data_item["data"].modality_name == "data"
        assert data_item["data"].type == DataType.IMAGE
        assert np.array_equal(data_item["data"].data, np.zeros((28, 28)))
        assert isinstance(target_item["target"], DatasetDataItemModality)
        assert target_item["target"].modality_name == "target"
        assert target_item["target"].type == DataType.TABULAR
        assert pd.DataFrame([0]).equals(target_item["target"].data)
    else:
        assert isinstance(data_item["data"], torch.Tensor)
        assert torch.equal(data_item["data"], torch.zeros((28, 28)))
        assert torch.equal(target_item["target"], torch.tensor(0))


def test_getitem_raises_on_unsupported_format(mocker, mock_mnist_controller, tmp_path):
    dataset = MnistDataset(root=tmp_path)
    dataset._to_format = DataReturnFormat.SKLEARN

    with pytest.raises(FedbiomedError):
        dataset[0]

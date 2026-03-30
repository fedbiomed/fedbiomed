from unittest.mock import patch

import pytest

from fedbiomed.common.dataset_controller import CustomController
from fedbiomed.common.exceptions import FedbiomedError


def test_init_success(tmp_path):
    controller = CustomController(root=tmp_path)
    assert controller._controller_kwargs == {"root": str(tmp_path)}
    assert controller.root == tmp_path


def test_init_invalid_path():
    with pytest.raises(FedbiomedError):
        CustomController(root="/nonexistent/path")


def test_init_invalid_type():
    with pytest.raises(FedbiomedError):
        CustomController(root=123)


def test_dlp_initialised(tmp_path):
    # DataLoadingPlanMixin.__init__ must have run — _dlp should be None, not missing
    controller = CustomController(root=tmp_path)
    assert controller._dlp is None


def test_shape_warns_and_returns_none(tmp_path):
    controller = CustomController(root=tmp_path)
    with patch(
        "fedbiomed.common.dataset_controller._custom_controller.logger.warning"
    ) as mock_warn:
        result = controller.shape()
    assert result is None
    mock_warn.assert_called_once()
    assert "shape" in mock_warn.call_args[0][0]


def test_get_sample_warns_and_returns_none(tmp_path):
    controller = CustomController(root=tmp_path)
    with patch(
        "fedbiomed.common.dataset_controller._custom_controller.logger.warning"
    ) as mock_warn:
        result = controller.get_sample(0)
    assert result is None
    mock_warn.assert_called_once()
    assert "get_sample" in mock_warn.call_args[0][0]


def test_get_types_warns_and_returns_none(tmp_path):
    controller = CustomController(root=tmp_path)
    with patch(
        "fedbiomed.common.dataset_controller._custom_controller.logger.warning"
    ) as mock_warn:
        result = controller.get_types()
    assert result is None
    mock_warn.assert_called_once()
    assert "get_types" in mock_warn.call_args[0][0]


def test_len_raises(tmp_path):
    controller = CustomController(root=tmp_path)
    with pytest.raises(FedbiomedError):
        len(controller)

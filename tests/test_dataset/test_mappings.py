from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset._mappings import (
    ControllerParametersBase,
    MedicalFolderParameters,
    get_controller,
)
from fedbiomed.common.exceptions import FedbiomedError


def test_controller_parameters_base_to_dict():
    """Test to_dict method removes None values"""
    params = ControllerParametersBase(root="/tmp/test")
    assert params.to_dict() == {"root": "/tmp/test"}

    @dataclass
    class TestParams(ControllerParametersBase):
        opt: str = None

    params_sub = TestParams(root="/tmp/test", opt=None)
    assert params_sub.to_dict() == {"root": "/tmp/test"}

    params_sub_val = TestParams(root="/tmp/test", opt="value")
    assert params_sub_val.to_dict() == {"root": "/tmp/test", "opt": "value"}


def test_controller_parameters_base_from_dict():
    """Test from_dict method filters extra keys"""
    data = {"root": "/tmp/test", "extra": "ignored"}
    params = ControllerParametersBase.from_dict(data)
    assert params.root == "/tmp/test"
    assert not hasattr(params, "extra")


def test_medical_folder_parameters():
    """Test MedicalFolderParameters specifics"""
    params = MedicalFolderParameters(root="/tmp/med")
    assert params.dlp is None
    assert params.tabular_file is None

    # Test with DLP
    dlp_mock = MagicMock(spec=DataLoadingPlan)
    params = MedicalFolderParameters(root="/tmp/med", dlp=dlp_mock)
    assert params.dlp == dlp_mock

    # Test conversion
    dct = params.to_dict()
    assert "dlp" in dct
    assert dct["dlp"] is not None

    # asdict deepcopies fields, so we check type rather than identity for the mock
    assert isinstance(dct["dlp"], MagicMock)


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_success(mock_registry):
    """Test successful controller creation"""
    mock_controller_cls = MagicMock()
    mock_params_cls = MagicMock()
    mock_params_instance = MagicMock()
    mock_params_instance.to_dict.return_value = {"param1": "value1"}
    mock_params_cls.from_dict.return_value = mock_params_instance

    mock_registry.__getitem__.return_value = (
        mock_controller_cls,
        mock_params_cls,
        MagicMock(),
    )
    mock_registry.__contains__.side_effect = lambda k: True

    result = get_controller("csv", {"param1": "value1"})

    mock_params_cls.from_dict.assert_called_once_with({"param1": "value1"})
    mock_controller_cls.assert_called_once_with(param1="value1")
    assert result == mock_controller_cls.return_value


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_unknown_type(mock_registry):
    """Test handling of unknown dataset type"""
    with pytest.raises(FedbiomedError) as excinfo:
        get_controller("unknown_type", {})
    assert ErrorNumbers.FB632.value in str(excinfo.value)
    assert "Unknown 'data_type'" in str(excinfo.value)


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_param_parse_error(mock_registry):
    """Test error during parameter parsing"""
    mock_params_cls = MagicMock()
    mock_params_cls.from_dict.side_effect = Exception("Parse error")
    mock_registry.__getitem__.return_value = (MagicMock(), mock_params_cls, MagicMock())
    mock_registry.__contains__.return_value = True

    with pytest.raises(FedbiomedError) as excinfo:
        get_controller("csv", {})
    assert ErrorNumbers.FB632.value in str(excinfo.value)
    assert "Failed to parse dataset_parameters" in str(excinfo.value)


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_instantiation_error(mock_registry):
    """Test error during controller instantiation"""
    mock_controller_cls = MagicMock(side_effect=Exception("Init error"))
    mock_params_cls = MagicMock()
    mock_params_cls.from_dict.return_value.to_dict.return_value = {}
    mock_registry.__getitem__.return_value = (
        mock_controller_cls,
        mock_params_cls,
        MagicMock(),
    )
    mock_registry.__contains__.return_value = True

    with pytest.raises(FedbiomedError) as excinfo:
        get_controller("csv", {})
    assert ErrorNumbers.FB632.value in str(excinfo.value)
    assert "Unhandled exception occurred" in str(excinfo.value)


@patch("fedbiomed.common.dataset._mappings.REGISTRY_CONTROLLERS")
def test_get_controller_fedbiomed_error_propagation(mock_registry):
    """Test that FedbiomedError is propagated as is"""
    mock_controller_cls = MagicMock(side_effect=FedbiomedError("Original error"))
    mock_params_cls = MagicMock()
    mock_params_cls.from_dict.return_value.to_dict.return_value = {}
    mock_registry.__getitem__.return_value = (
        mock_controller_cls,
        mock_params_cls,
        MagicMock(),
    )
    mock_registry.__contains__.return_value = True

    with pytest.raises(FedbiomedError) as excinfo:
        get_controller("csv", {})
    assert "Original error" in str(excinfo.value)

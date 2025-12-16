from unittest.mock import MagicMock, patch

import pytest

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.dataset_manager._registry_controllers import (
    REGISTRY_CONTROLLERS,
    ControllerParametersBase,
    MedicalFolderParameters,
    get_controller,
)


def test_controller_parameters_base():
    """
    Test ControllerParametersBase serialization and deserialization
    """
    # Test from_dict filters extra fields and to_dict
    data = {"root": "/tmp/test", "extra_field": "should_be_ignored"}
    params = ControllerParametersBase.from_dict(data)
    assert params.to_dict() == {"root": "/tmp/test"}
    assert not hasattr(params, "extra_field")


def test_medical_folder_parameters():
    """
    Test MedicalFolderParameters specific fields
    """
    # Test defaults (None values should be stripped by to_dict)
    params = MedicalFolderParameters(root="/tmp/med")
    assert params.to_dict() == {"root": "/tmp/med"}

    # Test with optional fields
    params = MedicalFolderParameters(root="/tmp/med", tabular_file="tab.csv")
    assert params.to_dict() == {
        "root": "/tmp/med",
        "tabular_file": "tab.csv",
    }


def test_get_controller_unknown_type():
    """
    Test get_controller raises NotImplementedError for unknown types
    """
    with pytest.raises(FedbiomedError, match="Unknown 'data_type'"):
        get_controller("unknown-type", {"root": "/tmp"})


def test_get_controller_success():
    """
    Test get_controller successfully instantiates a controller
    """
    # Mock the registry components
    mock_controller_cls = MagicMock()
    mock_params_cls = MagicMock()
    mock_params_instance = MagicMock()

    # Setup mocks
    mock_params_instance.to_dict.return_value = {"root": "/tmp/mock"}
    mock_params_cls.from_dict.return_value = mock_params_instance

    # Create a fake registry entry
    fake_registry = {"mock-type": (mock_controller_cls, mock_params_cls, MagicMock())}

    # Patch the registry dictionary
    with patch.dict(REGISTRY_CONTROLLERS, fake_registry):
        controller = get_controller("mock-type", {"root": "/tmp/mock"})

        # Verify interactions
        mock_params_cls.from_dict.assert_called_once_with({"root": "/tmp/mock"})
        mock_controller_cls.assert_called_once_with(root="/tmp/mock")
        assert controller == mock_controller_cls.return_value


def test_get_controller_param_parsing_error():
    """
    Test get_controller handles parameter parsing errors
    """
    mock_controller_cls = MagicMock()
    mock_params_cls = MagicMock()

    # Simulate error during parameter parsing
    mock_params_cls.from_dict.side_effect = Exception("Parsing error")

    fake_registry = {"mock-type": (mock_controller_cls, mock_params_cls, MagicMock())}

    with patch.dict(REGISTRY_CONTROLLERS, fake_registry):
        with pytest.raises(FedbiomedError) as excinfo:
            get_controller("mock-type", {})

        assert ErrorNumbers.FB632.value in str(excinfo.value)
        assert "Failed to parse dataset_parameters" in str(excinfo.value)


def test_get_controller_instantiation_error():
    """
    Test get_controller handles generic instantiation errors
    """
    mock_controller_cls = MagicMock()
    mock_params_cls = MagicMock()
    mock_params_instance = MagicMock()

    mock_params_instance.to_dict.return_value = {}
    mock_params_cls.from_dict.return_value = mock_params_instance

    # Simulate generic exception during controller init
    mock_controller_cls.side_effect = Exception("Init error")

    fake_registry = {"mock-type": (mock_controller_cls, mock_params_cls, MagicMock())}

    with patch.dict(REGISTRY_CONTROLLERS, fake_registry):
        with pytest.raises(FedbiomedError) as excinfo:
            get_controller("mock-type", {})

        assert ErrorNumbers.FB632.value in str(excinfo.value)
        assert "Unhandled exception occurred" in str(excinfo.value)


def test_get_controller_instantiation_fedbiomed_error():
    """
    Test get_controller re-raises FedbiomedError during instantiation
    """
    mock_controller_cls = MagicMock()
    mock_params_cls = MagicMock()
    mock_params_instance = MagicMock()

    mock_params_instance.to_dict.return_value = {}
    mock_params_cls.from_dict.return_value = mock_params_instance

    # Simulate FedbiomedError during controller init
    fb_error = FedbiomedError("Specific error")
    mock_controller_cls.side_effect = fb_error

    fake_registry = {"mock-type": (mock_controller_cls, mock_params_cls, MagicMock())}

    with patch.dict(REGISTRY_CONTROLLERS, fake_registry):
        with pytest.raises(FedbiomedError) as excinfo:
            get_controller("mock-type", {})

        # Should be the exact same exception object
        assert excinfo.value is fb_error

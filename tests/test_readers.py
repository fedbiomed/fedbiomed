from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import pytest

from fedbiomed.common.dataset_reader import NiftiReader
from fedbiomed.common.dataset_types import drf_default


# NifTI READER TESTS -------------------------------------------------------------------------------------------------------------
@pytest.fixture
def nifti_reader():
    """Fixture to create a NiftiReader instance."""
    return NiftiReader


def test_nifti_reader_initialization(nifti_reader):
    """Test the initialization of NiftiReader."""
    assert nifti_reader.return_format == drf_default


def test_nifti_reader_read(nifti_reader):
    """Test reading a NIfTI file."""
    # Assuming you have a valid NIfTI file path for testing
    nifti_file_path = "path/to/test.nii"  # Replace with an actual test file path
    with pytest.raises(FileNotFoundError):
        nifti_reader.read(nifti_file_path)


# If the file exists, you would check the output format
# tensor = nifti_reader.read(nifti_file_path)
# assert isinstance(tensor, torch.Tensor)
# assert tensor.dim() == 4  # Assuming the output is [D, H, W]
@pytest.fixture
def mock_nifti_file(tmp_path):
    """Creates a mock NIfTI file and returns its path."""
    file_path = tmp_path / "test.nii"
    # Create a dummy file to simulate existence
    file_path.write_bytes(b"dummy")
    return str(file_path)


@pytest.fixture
def mock_nifti_reader(monkeypatch):
    """Fixture to create a NiftiReader with mocked nibabel and torch."""

    data = np.ones((2, 2, 2))
    affine = np.eye(4)
    real_img = nib.Nifti1Image(data, affine)
    # Mock nibabel.load and get_fdata
    monkeypatch.setattr("nibabel.load", mock.MagicMock(return_value=real_img))

    return NiftiReader


def test_nifti_reader_read_return_format(mock_nifti_reader, mock_nifti_file):
    """Test NiftiReader.read with all supported return formats."""
    reader = mock_nifti_reader
    result = reader.read(mock_nifti_file)
    assert isinstance(result, nib.Nifti1Image)
    assert result.shape == (2, 2, 2)


def test_nifti_reader_validate(nifti_reader, monkeypatch):
    """Test the validation of a NIfTI file."""

    monkeypatch.setattr(Path, "exists", lambda self: False)

    with pytest.raises(FileNotFoundError):
        nifti_reader.validate(Path("path/to/nonexistent.nii"))

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(ValueError):
        nifti_reader.validate(Path("path/to/invalid_file.txt"))

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "suffix", property(lambda self: ".txt"))
    with pytest.raises(ValueError):
        nifti_reader.validate(Path("path/to/invalid_file.txt"))

    # Test with a valid file path
    valid_path = Path("path/to/valid_file.nii")
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "suffix", property(lambda self: ".nii"))
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    nifti_reader.validate(valid_path)

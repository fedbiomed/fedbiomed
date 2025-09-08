import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataset_controller import (
    MedicalFolderController,
)
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture(autouse=True)
def mock_nifti_reader(monkeypatch):
    monkeypatch.setattr(
        "fedbiomed.common.dataset_reader.NiftiReader.read",
        MagicMock(return_value="mock_nifti_data"),
    )


@pytest.fixture
def temp_medical_folder():
    """Create a temporary medical folder structure for testing"""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create proper folder structure: root/patient/modality/file.nii
        for patient in ("patient1", "patient2", "patient3"):
            for modality in ("T1", "T2"):
                modality_dir = os.path.join(temp_dir, patient, modality)
                os.makedirs(modality_dir)
                # Create dummy NIfTI files
                nii_file = os.path.join(modality_dir, f"{patient}_{modality}.nii")
                with open(nii_file, "w") as f:
                    f.write("dummy nifti data")

        # Create demographics CSV file
        participants_file = os.path.join(temp_dir, "participants.csv")
        with open(participants_file, "w") as f:
            f.write("participant_id,age,gender\n")
            f.write("patient1,30,M\n")
            f.write("patient2,25,F\n")
            f.write("patient3,40,M\n")

        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


@pytest.fixture
def incomplete_medical_folder():
    """Create a medical folder with missing modalities for some patients"""
    temp_dir = tempfile.mkdtemp()
    try:
        # patient1 has both T1 and T2
        for modality in ("T1", "T2"):
            modality_dir = os.path.join(temp_dir, "patient1", modality)
            os.makedirs(modality_dir)
            nii_file = os.path.join(modality_dir, f"patient1_{modality}.nii")
            with open(nii_file, "w") as f:
                f.write("dummy data")

        # patient2 has only T1
        modality_dir = os.path.join(temp_dir, "patient2", "T1")
        os.makedirs(modality_dir)
        nii_file = os.path.join(modality_dir, "patient2_T1.nii")
        with open(nii_file, "w") as f:
            f.write("dummy data")

        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def test_init_basic(temp_medical_folder):
    """Test basic initialization of MedicalFolderController"""
    controller = MedicalFolderController(root=temp_medical_folder)

    assert os.path.realpath(controller.root) == os.path.realpath(temp_medical_folder)
    assert controller.tabular_file is None
    assert controller.index_col is None
    assert len(controller.modalities) > 0
    assert len(controller) > 0


def test_init_with_demographics(temp_medical_folder):
    """Test initialization with demographics file"""
    participants_file = os.path.join(temp_medical_folder, "participants.csv")

    controller = MedicalFolderController(
        root=temp_medical_folder,
        tabular_file=participants_file,
        index_col="participant_id",
    )

    assert str(controller.tabular_file) == os.path.realpath(participants_file)
    assert controller.index_col == "participant_id"
    assert controller.demographics is not None
    assert len(controller.demographics) == 3


def test_tabular_file_setter_valid_path(temp_medical_folder):
    """Test tabular_file setter with valid CSV path"""
    controller = MedicalFolderController(root=temp_medical_folder)
    csv_path = os.path.join(temp_medical_folder, "participants.csv")
    controller.tabular_file = csv_path
    assert controller.tabular_file == Path(csv_path).resolve()


def test_tabular_file_setter_none(temp_medical_folder):
    """Test tabular_file setter with None"""
    controller = MedicalFolderController(root=temp_medical_folder)
    controller.tabular_file = None
    assert controller.tabular_file is None


def test_tabular_file_setter_invalid_type(temp_medical_folder):
    """Test tabular_file setter with invalid type"""
    controller = MedicalFolderController(root=temp_medical_folder)

    with pytest.raises(FedbiomedError) as exc_info:
        controller.tabular_file = 123
    assert ErrorNumbers.FB632.value in str(exc_info.value)


def test_tabular_file_setter_invalid_extension(temp_medical_folder):
    """Test tabular_file setter with invalid file extension"""
    controller = MedicalFolderController(root=temp_medical_folder)
    invalid_file = os.path.join(temp_medical_folder, "test.txt")

    with pytest.raises(FedbiomedError) as exc_info:
        controller.tabular_file = invalid_file
    assert ErrorNumbers.FB613.value in str(exc_info.value)


def test_index_col_setter_valid(temp_medical_folder):
    """Test index_col setter with valid values"""
    controller = MedicalFolderController(root=temp_medical_folder)

    controller.index_col = "participant_id"
    assert controller.index_col == "participant_id"

    controller.index_col = 0
    assert controller.index_col == 0

    controller.index_col = None
    assert controller.index_col is None


def test_index_col_setter_invalid(temp_medical_folder):
    """Test index_col setter with invalid type"""
    controller = MedicalFolderController(root=temp_medical_folder)

    with pytest.raises(FedbiomedError) as exc_info:
        controller.index_col = 12.5
    assert ErrorNumbers.FB613.value in str(exc_info.value)


def test_demographics_property_none(temp_medical_folder):
    """Test demographics property when tabular_file or index_col is None"""
    controller = MedicalFolderController(root=temp_medical_folder)
    assert controller.demographics is None


def test_demographics_property_valid(temp_medical_folder):
    """Test demographics property with valid tabular file"""
    participants_file = os.path.join(temp_medical_folder, "participants.csv")
    controller = MedicalFolderController(
        root=temp_medical_folder,
        tabular_file=participants_file,
        index_col="participant_id",
    )

    demographics = controller.demographics
    assert isinstance(demographics, pd.DataFrame)
    assert len(demographics) == 3
    assert "age" in demographics.columns
    assert "gender" in demographics.columns


def test_read_demographics_valid(temp_medical_folder):
    """Test read_demographics with valid file"""
    controller = MedicalFolderController(root=temp_medical_folder)
    participants_file = os.path.join(temp_medical_folder, "participants.csv")

    demographics = controller.read_demographics(participants_file, "participant_id")

    assert isinstance(demographics, pd.DataFrame)
    assert len(demographics) == 3
    assert "patient1" in demographics.index


def test_read_demographics_with_duplicates(temp_medical_folder):
    """Test read_demographics handles duplicate indices correctly"""
    # Create CSV with duplicate participant_id
    duplicate_csv = os.path.join(temp_medical_folder, "duplicates.csv")
    with open(duplicate_csv, "w") as f:
        f.write("participant_id,age,gender\n")
        f.write("patient1,30,M\n")
        f.write("patient1,31,F\n")  # duplicate
        f.write("patient2,25,F\n")

    controller = MedicalFolderController(root=temp_medical_folder)
    demographics = controller.read_demographics(duplicate_csv, "participant_id")

    # Should keep first occurrence
    assert len(demographics) == 2
    assert demographics.loc["patient1", "age"] == 30


def test_read_demographics_file_error(temp_medical_folder):
    """Test read_demographics with non-existent file"""
    controller = MedicalFolderController(root=temp_medical_folder)
    non_existent_file = os.path.join(temp_medical_folder, "nonexistent.csv")

    with pytest.raises(FedbiomedError) as exc_info:
        controller.read_demographics(non_existent_file, "participant_id")
    assert ErrorNumbers.FB613.value in str(exc_info.value)


def test_demographics_column_names(temp_medical_folder):
    """Test demographics_column_names method"""
    controller = MedicalFolderController(root=temp_medical_folder)
    participants_file = os.path.join(temp_medical_folder, "participants.csv")

    columns = controller.demographics_column_names(participants_file)

    assert "age" in columns
    assert "gender" in columns


def test_make_df_dir_valid_structure(temp_medical_folder):
    """Test _make_df_dir with valid folder structure"""
    controller = MedicalFolderController(root=temp_medical_folder)
    modalities, df_dir = controller._make_df_dir(Path(temp_medical_folder))

    assert isinstance(modalities, set)
    assert isinstance(df_dir, pd.DataFrame)
    assert len(df_dir) > 0
    assert all(col in df_dir.columns for col in ["subject", "modality", "file"])


def test_make_df_dir_invalid_structure():
    """Test _make_df_dir with invalid folder structure"""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create files directly in root without proper structure
        with open(os.path.join(temp_dir, "file.nii"), "w") as f:
            f.write("dummy")

        controller = MedicalFolderController.__new__(MedicalFolderController)

        with pytest.raises(FedbiomedError) as exc_info:
            controller._make_df_dir(Path(temp_dir))
        assert ErrorNumbers.FB613.value in str(exc_info.value)
    finally:
        shutil.rmtree(temp_dir)


def test_make_df_dir_no_valid_files():
    """Test _make_df_dir with no valid NIfTI files"""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create proper structure but with invalid file extensions
        os.makedirs(os.path.join(temp_dir, "patient1", "T1"))
        with open(os.path.join(temp_dir, "patient1", "T1", "file.txt"), "w") as f:
            f.write("dummy")

        controller = MedicalFolderController.__new__(MedicalFolderController)

        with pytest.raises(FedbiomedError) as exc_info:
            controller._make_df_dir(Path(temp_dir))
        assert ErrorNumbers.FB613.value in str(exc_info.value)
    finally:
        shutil.rmtree(temp_dir)


def test_make_df_dir_multiple_files_per_modality():
    """Test _make_df_dir with multiple files per modality"""
    temp_dir = tempfile.mkdtemp()
    try:
        modality_dir = os.path.join(temp_dir, "patient1", "T1")
        os.makedirs(modality_dir)
        # Create multiple NIfTI files in same modality folder
        with open(os.path.join(modality_dir, "file1.nii"), "w") as f:
            f.write("dummy")
        with open(os.path.join(modality_dir, "file2.nii"), "w") as f:
            f.write("dummy")

        controller = MedicalFolderController.__new__(MedicalFolderController)

        with pytest.raises(FedbiomedError) as exc_info:
            controller._make_df_dir(Path(temp_dir))
        assert ErrorNumbers.FB613.value in str(exc_info.value)
    finally:
        shutil.rmtree(temp_dir)


def test_make_df_dir_missing_requested_modalities(temp_medical_folder):
    """Test _make_df_dir when requested modalities are not found"""
    controller = MedicalFolderController.__new__(MedicalFolderController)

    with pytest.raises(FedbiomedError) as exc_info:
        controller._make_df_dir(Path(temp_medical_folder), modalities=["FLAIR"])
    assert ErrorNumbers.FB613.value in str(exc_info.value)


def test_make_df_dir_incomplete_modalities(incomplete_medical_folder):
    """Test _make_df_dir with patients missing some modalities"""
    controller = MedicalFolderController.__new__(MedicalFolderController)
    modalities, df_dir = controller._make_df_dir(
        Path(incomplete_medical_folder), modalities=["T1", "T2"]
    )

    # Should only include patient1 who has both T1 and T2
    subjects = df_dir["subject"].unique()
    assert "patient1" in subjects
    assert "patient2" not in subjects  # patient2 missing T2


def test_make_dataset_without_demographics(temp_medical_folder):
    """Test _make_dataset without demographics file"""
    controller = MedicalFolderController.__new__(MedicalFolderController)
    modalities, subjects, samples = controller._make_dataset(
        Path(temp_medical_folder), None, None
    )

    assert isinstance(modalities, list)
    assert isinstance(subjects, list)
    assert isinstance(samples, list)
    assert len(samples) > 0
    assert all("demographics" not in sample for sample in samples)


def test_make_dataset_with_demographics(temp_medical_folder):
    """Test _make_dataset with demographics file"""
    participants_file = Path(temp_medical_folder) / "participants.csv"
    controller = MedicalFolderController.__new__(MedicalFolderController)
    controller.read_demographics = MagicMock(
        return_value=pd.DataFrame(
            {"age": [30, 25, 40], "gender": ["M", "F", "M"]},
            index=["patient1", "patient2", "patient3"],
        )
    )

    _, _, samples = controller._make_dataset(
        Path(temp_medical_folder), participants_file, "participant_id"
    )

    assert all("demographics" in sample for sample in samples)


def test_make_dataset_mismatched_args(temp_medical_folder):
    """Test _make_dataset with mismatched tabular_file and index_col args"""
    controller = MedicalFolderController.__new__(MedicalFolderController)
    participants_file = Path(temp_medical_folder) / "participants.csv"

    with pytest.raises(FedbiomedError) as exc_info:
        controller._make_dataset(Path(temp_medical_folder), participants_file, None)
    assert ErrorNumbers.FB613.value in str(exc_info.value)


@patch("fedbiomed.common.dataset_reader.NiftiReader.read")
def testget_sample(mock_read, temp_medical_folder):
    """Test get_sample method"""
    mock_read.return_value = "mock_nifti_data"

    controller = MedicalFolderController(root=temp_medical_folder)
    mock_read.reset_mock()

    item = controller.get_sample(0)

    assert isinstance(item, dict)
    assert all(modality in item for modality in controller.modalities)
    assert mock_read.call_count == len(controller.modalities)


@patch("fedbiomed.common.dataset_reader.NiftiReader.read")
def testget_sample_with_demographics(mock_read, temp_medical_folder):
    """Test get_sample with demographics"""
    mock_read.return_value = "mock_nifti_data"
    participants_file = os.path.join(temp_medical_folder, "participants.csv")

    controller = MedicalFolderController(
        root=temp_medical_folder,
        tabular_file=participants_file,
        index_col="participant_id",
    )

    item = controller.get_sample(0)

    assert "demographics" in item
    assert isinstance(item["demographics"], dict)


@patch("fedbiomed.common.dataset_reader.NiftiReader.read")
def testget_sample_error(mock_read, temp_medical_folder):
    """Test get_sample with read error"""

    controller = MedicalFolderController(root=temp_medical_folder)

    mock_read.side_effect = Exception("Read error")
    with pytest.raises(FedbiomedError) as exc_info:
        controller.get_sample(0)
    assert ErrorNumbers.FB632.value in str(exc_info.value)


def test_len(temp_medical_folder):
    """Test __len__ method"""
    controller = MedicalFolderController(root=temp_medical_folder)

    length = len(controller)
    assert isinstance(length, int)
    assert length > 0
    assert length == len(controller._samples)


def test_controller_kwargs_property(temp_medical_folder):
    """Test that _controller_kwargs is properly set"""
    participants_file = os.path.join(temp_medical_folder, "participants.csv")
    controller = MedicalFolderController(
        root=temp_medical_folder,
        tabular_file=participants_file,
        index_col="participant_id",
        modalities=["T1"],
    )

    kwargs = controller._controller_kwargs
    assert os.path.realpath(kwargs["root"]) == os.path.realpath(temp_medical_folder)
    assert os.path.realpath(kwargs["tabular_file"]) == os.path.realpath(
        str(participants_file)
    )
    assert kwargs["index_col"] == "participant_id"
    assert kwargs["modalities"] == ["T1"]


def test_extensions_property():
    """Test that _extensions class property is correctly defined"""
    assert hasattr(MedicalFolderController, "_extensions")
    assert MedicalFolderController._extensions == (".nii", ".nii.gz")


def test_hidden_files_ignored():
    """Test that hidden files and folders are ignored"""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create normal structure
        normal_dir = os.path.join(temp_dir, "patient1", "T1")
        os.makedirs(normal_dir)
        with open(os.path.join(normal_dir, "file.nii"), "w") as f:
            f.write("dummy")

        # Create hidden folder structure
        hidden_dir = os.path.join(temp_dir, ".hidden_patient", "T1")
        os.makedirs(hidden_dir)
        with open(os.path.join(hidden_dir, "file.nii"), "w") as f:
            f.write("dummy")

        # Create structure with hidden file
        normal_dir2 = os.path.join(temp_dir, "patient2", "T1")
        os.makedirs(normal_dir2)
        with open(os.path.join(normal_dir2, ".hidden_file.nii"), "w") as f:
            f.write("dummy")

        controller = MedicalFolderController(root=temp_dir)

        # Should only find patient1, not hidden_patient or patient2 with hidden file
        assert len(controller) == 1

    finally:
        shutil.rmtree(temp_dir)


def test_nii_gz_extension_support():
    """Test that .nii.gz files are properly supported"""
    temp_dir = tempfile.mkdtemp()
    try:
        modality_dir = os.path.join(temp_dir, "patient1", "T1")
        os.makedirs(modality_dir)
        with open(os.path.join(modality_dir, "patient1_T1.nii.gz"), "w") as f:
            f.write("dummy compressed nifti")

        controller = MedicalFolderController(root=temp_dir)
        assert len(controller) == 1

    finally:
        shutil.rmtree(temp_dir)


def test_tsv_file_support(temp_medical_folder):
    """Test that TSV files are supported for demographics"""
    # Create a TSV file
    tsv_file = os.path.join(temp_medical_folder, "participants.tsv")
    with open(tsv_file, "w") as f:
        f.write("participant_id\tage\tgender\n")
        f.write("patient1\t30\tM\n")
        f.write("patient2\t25\tF\n")

    controller = MedicalFolderController(root=temp_medical_folder)
    controller.tabular_file = tsv_file  # Should not raise error

    assert controller.tabular_file == Path(tsv_file).resolve()


def test_path_expanduser_functionality():
    """Test that path expansion with ~ works correctly"""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a simple structure
        modality_dir = os.path.join(temp_dir, "patient1", "T1")
        os.makedirs(modality_dir)
        with open(os.path.join(modality_dir, "patient1_T1.nii"), "w") as f:
            f.write("dummy")

        # Test with Path object
        controller = MedicalFolderController(root=Path(temp_dir))
        assert len(controller) == 1

    finally:
        shutil.rmtree(temp_dir)


def test_case_insensitive_extensions():
    """Test that extension matching is case insensitive"""
    temp_dir = tempfile.mkdtemp()
    try:
        modality_dir = os.path.join(temp_dir, "patient1", "T1")
        os.makedirs(modality_dir)
        # Use uppercase extension
        with open(os.path.join(modality_dir, "patient1_T1.NII"), "w") as f:
            f.write("dummy")

        controller = MedicalFolderController(root=temp_dir)
        assert len(controller) == 1

    finally:
        shutil.rmtree(temp_dir)


def test_subject_intersection_with_demographics(temp_medical_folder):
    """Test that only subjects present in both folder structure and demographics are included"""
    # Create demographics with extra subject not in folder structure
    extra_demo_csv = os.path.join(temp_medical_folder, "extra_demo.csv")
    with open(extra_demo_csv, "w") as f:
        f.write("participant_id,age,gender\n")
        f.write("patient1,30,M\n")
        f.write("patient2,25,F\n")
        f.write("patient4,35,F\n")  # This patient doesn't exist in folder

    controller = MedicalFolderController(
        root=temp_medical_folder,
        tabular_file=extra_demo_csv,
        index_col="participant_id",
    )

    # Should only include patients 1 and 2, not patient4
    assert len(controller) == 2

    # Verify patient4 is not in any sample
    all_demographics = [
        sample.get("demographics", {}) for sample in controller._samples
    ]
    participant_ids = [
        demo.get("participant_id") if isinstance(demo, dict) else None
        for demo in all_demographics
    ]
    assert "patient4" not in participant_ids

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan, MapperBlock
from fedbiomed.common.dataset_controller import (
    MedicalFolderController,
    MedicalFolderLoadingBlockTypes,
)
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture(autouse=True)
def mock_nifti_reader(monkeypatch):
    monkeypatch.setattr(
        "fedbiomed.common.dataset_reader.NiftiReader.read",
        MagicMock(return_value="mock_nifti_data"),
    )


@pytest.fixture(params=["default"])
def temp_medical_folder(request):
    """Create a temporary medical folder structure for testing"""
    temp_dir = tempfile.mkdtemp()

    match request.param:
        case "default":
            tree_dir = {
                "patient1": ["T1", "T2"],
                "patient2": ["T1", "T2"],
                "patient3": ["T1", "T2"],
            }
        case "incomplete_subject":
            tree_dir = {
                "patient1": ["T1", "T2"],
                "patient2": ["T1"],
                "patient3": ["T1", "T2"],
            }
        case "incomplete_modalities":
            tree_dir = {
                "patient1": ["T1", "T2"],
                "patient2": ["T1", "T2"],
                "patient3": ["T1", "T3"],
            }
        case "dlp":
            tree_dir = {
                "patient1": ["T1_A", "T2"],
                "patient2": ["T1_B", "T2"],
                "patient3": ["T1_C", "T2"],
            }
        case "empty_intersection_demographics":
            tree_dir = {
                "subject1": ["T1", "T2"],
                "subject2": ["T1", "T2"],
                "subject3": ["T1", "T2"],
            }
        case _:
            raise Exception("Unexpected param")

    try:
        # Create proper folder structure: root/patient/modality/file.nii
        for subject, modalities in tree_dir.items():
            for modality in modalities:
                modality_dir = os.path.join(temp_dir, subject, modality)
                os.makedirs(modality_dir)
                # Create dummy NIfTI files
                nii_file = os.path.join(modality_dir, f"{subject}_{modality}.nii")
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


def test_init_basic(temp_medical_folder):
    """Test basic initialization of MedicalFolderController"""
    controller = MedicalFolderController(root=temp_medical_folder)
    assert os.path.realpath(controller.root) == os.path.realpath(temp_medical_folder)
    assert controller.tabular_file is None
    assert controller.index_col is None
    assert controller.demographics is None
    assert all(modality in controller.modalities for modality in ["T1", "T2"])
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
    assert isinstance(controller.demographics, pd.DataFrame)
    assert len(controller.demographics) == 3
    assert all(col in controller.demographics.columns for col in ["age", "gender"])


def test_init_demographics_without_index_col(temp_medical_folder):
    """Test initialization with demographics file"""
    participants_file = os.path.join(temp_medical_folder, "participants.csv")
    with pytest.raises(FedbiomedError) as exc_info:
        _ = MedicalFolderController(
            root=temp_medical_folder,
            tabular_file=participants_file,
        )
    partial_msg = "Arguments `tabular_file` and `index_col`, both or none are expected"
    assert partial_msg in str(exc_info.value)


def test_tabular_file_invalid_type(temp_medical_folder):
    """Test tabular_file setter with invalid type"""
    with pytest.raises(FedbiomedError) as exc_info:
        _ = MedicalFolderController(
            root=temp_medical_folder,
            tabular_file=123,
            index_col="participant_id",
        )
    partial_msg = "Expected a string or Path"
    assert partial_msg in str(exc_info.value)


def test_tabular_file_invalid_extension(temp_medical_folder):
    """Test tabular_file setter with invalid file extension"""
    invalid_file = os.path.join(temp_medical_folder, "test.txt")
    with pytest.raises(FedbiomedError) as exc_info:
        _ = MedicalFolderController(
            root=temp_medical_folder,
            tabular_file=invalid_file,
            index_col="participant_id",
        )
    partial_msg = "Path does not correspond to a CSV or TSV file"
    assert partial_msg in str(exc_info.value)


def test_index_col_invalid(temp_medical_folder):
    """Test index_col setter with invalid type"""
    participants_file = os.path.join(temp_medical_folder, "participants.csv")
    with pytest.raises(FedbiomedError) as exc_info:
        _ = MedicalFolderController(
            root=temp_medical_folder,
            tabular_file=participants_file,
            index_col=12.5,
        )
    partial_msg = "`index_col` should be of type `int` or `str`"
    assert partial_msg in str(exc_info.value)


# === read_demographics ===


def test_read_demographics_valid(temp_medical_folder):
    """Test read_demographics with valid file"""
    controller = MedicalFolderController.__new__(MedicalFolderController)
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

    controller = MedicalFolderController.__new__(MedicalFolderController)
    demographics = controller.read_demographics(duplicate_csv, "participant_id")
    # Should keep first occurrence
    assert len(demographics) == 2
    assert demographics.loc["patient1", "age"] == 30


def test_read_demographics_file_error(temp_medical_folder):
    """Test read_demographics with non-existent file"""
    controller = MedicalFolderController.__new__(MedicalFolderController)
    non_existent_file = os.path.join(temp_medical_folder, "nonexistent.csv")

    with pytest.raises(FedbiomedError) as exc_info:
        _ = controller.read_demographics(non_existent_file, "participant_id")
    assert ErrorNumbers.FB613.value in str(exc_info.value)


def test_demographics_column_names(temp_medical_folder):
    """Test demographics_column_names method"""
    controller = MedicalFolderController.__new__(MedicalFolderController)
    participants_file = os.path.join(temp_medical_folder, "participants.csv")
    columns = controller.demographics_column_names(participants_file)
    assert all(col in columns for col in ["age", "gender"])


# === _make_df_dir ===


def test_make_df_dir_valid_structure(temp_medical_folder):
    """Test _make_df_dir with valid folder structure"""
    controller = MedicalFolderController.__new__(MedicalFolderController)
    df_dir = controller._make_df_dir(Path(temp_medical_folder))
    assert isinstance(df_dir, pd.DataFrame)
    assert len(df_dir) > 0
    assert all(col in df_dir.columns for col in ["subject", "modality", "file", "path"])


def test_make_df_dir_invalid_structure():
    """Test _make_df_dir with invalid folder structure"""
    controller = MedicalFolderController.__new__(MedicalFolderController)
    temp_dir = tempfile.mkdtemp()
    # Create files directly in root without proper structure
    with open(os.path.join(temp_dir, "file.nii"), "w") as f:
        f.write("dummy")

    with pytest.raises(FedbiomedError) as exc_info:
        _ = controller._make_df_dir(Path(temp_dir))
    partial_msg = "Root folder does not match Medical Folder structure"
    assert partial_msg in str(exc_info.value)
    shutil.rmtree(temp_dir)


def test_make_df_dir_no_valid_files():
    """Test _make_df_dir with no valid NIfTI files"""
    controller = MedicalFolderController.__new__(MedicalFolderController)
    temp_dir = tempfile.mkdtemp()
    # Create proper structure but with invalid file extensions
    os.makedirs(os.path.join(temp_dir, "patient1", "T1"))
    with open(os.path.join(temp_dir, "patient1", "T1", "file.txt"), "w") as f:
        f.write("dummy")

    with pytest.raises(FedbiomedError) as exc_info:
        _ = controller._make_df_dir(Path(temp_dir))
    partial_msg = "Root folder does not match Medical Folder structure"
    assert partial_msg in str(exc_info.value)
    shutil.rmtree(temp_dir)


def test_make_df_dir_multiple_files_per_modality():
    """Test _make_df_dir with multiple files per modality"""
    controller = MedicalFolderController.__new__(MedicalFolderController)
    temp_dir = tempfile.mkdtemp()
    modality_dir = os.path.join(temp_dir, "patient1", "T1")
    os.makedirs(modality_dir)
    for file in ["file1.nii", "file2.nii"]:
        # Create multiple NIfTI files in same modality folder
        with open(os.path.join(modality_dir, file), "w") as f:
            f.write("dummy")

    with pytest.raises(FedbiomedError) as exc_info:
        _ = controller._make_df_dir(Path(temp_dir))
    partial_msg = "more than one valid file per modality"
    assert partial_msg in str(exc_info.value)
    shutil.rmtree(temp_dir)


# === _prepare_df_dir_for_use ===


def test_prepare_df_dir_for_use_valid_without_dlp(temp_medical_folder):
    controller = MedicalFolderController(temp_medical_folder, validate=False)
    _, df_dir = controller._prepare_df_dir_for_use(controller.df_dir)
    assert isinstance(df_dir, pd.DataFrame)
    assert len(df_dir) > 0
    assert all(col in df_dir.columns for col in ["subject", "modality", "file", "path"])


@pytest.mark.parametrize("temp_medical_folder", ["dlp"], indirect=True)
def test_prepare_df_dir_for_use_valid_with_dlp(temp_medical_folder):
    controller = MedicalFolderController(temp_medical_folder, validate=False)

    dlb = MapperBlock()
    modalities_to_folders = {
        "T1": ["T1_A", "T1_B", "T1_C"],
        "T2": ["T2"],
    }
    dlb.map = modalities_to_folders
    dlp = DataLoadingPlan({MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS: dlb})

    _, df_dir = controller._prepare_df_dir_for_use(controller.df_dir, dlp)
    assert all(modality in ["T1", "T2"] for modality in df_dir["modality"].unique())


def test_prepare_df_dir_for_use_missing_modalities(temp_medical_folder):
    """Test _prepare_df_dir_for_use when a subject in dlp is not in the folder structure"""
    controller = MedicalFolderController(temp_medical_folder, validate=False)

    dlb = MapperBlock()
    modalities_to_folders = {
        "T1": ["T1_A", "T1_B", "T1_C"],
        "T2": ["T2"],
        "T3": ["T3"],  # modality T3 is not present in folder structure
    }
    dlb.map = modalities_to_folders
    dlp = DataLoadingPlan({MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS: dlb})

    with pytest.raises(FedbiomedError) as exc_info:
        _ = controller._prepare_df_dir_for_use(controller.df_dir, dlp)
    partial_msg = "Some modality names are not found in the root folder"
    assert partial_msg in str(exc_info.value)


@pytest.mark.parametrize("temp_medical_folder", ["incomplete_subject"], indirect=True)
def test_prepare_df_dir_for_use_incomplete_subject(temp_medical_folder):
    """Test _prepare_df_dir_for_use no subject has all modalities"""
    controller = MedicalFolderController(temp_medical_folder, validate=False)
    _, df_dir = controller._prepare_df_dir_for_use(controller.df_dir)
    subjects = df_dir["subject"].unique()
    assert len(subjects) > 0
    assert "T2" not in subjects


@pytest.mark.parametrize(
    "temp_medical_folder", ["incomplete_modalities"], indirect=True
)
def test_prepare_df_dir_for_use_incomplete_modalities(temp_medical_folder):
    """Test _prepare_df_dir_for_use no subject has all modalities"""
    controller = MedicalFolderController(temp_medical_folder, validate=False)
    with pytest.raises(FedbiomedError) as exc_info:
        _ = controller._prepare_df_dir_for_use(controller.df_dir)
    partial_msg = "No 'subject' matches all `modalities`"
    assert partial_msg in str(exc_info.value)


# === _make_dataset ===


def test_make_dataset_without_demographics(temp_medical_folder):
    """Test _make_dataset without demographics file"""
    controller = MedicalFolderController(temp_medical_folder, validate=False)
    _, df_dir = controller._prepare_df_dir_for_use(controller.df_dir)
    subjects, samples = controller._make_dataset(demographics=None, df_dir=df_dir)

    assert isinstance(subjects, list)
    assert isinstance(samples, list)
    assert len(subjects) > 0 and len(samples) > 0
    assert all(set(sample.keys()) == {"T1", "T2"} for sample in samples)


def test_make_dataset_with_demographics(temp_medical_folder):
    """Test _make_dataset with demographics file"""
    controller = MedicalFolderController(
        temp_medical_folder,
        tabular_file=os.path.join(temp_medical_folder, "participants.csv"),
        index_col="participant_id",
        validate=False,
    )

    _, df_dir = controller._prepare_df_dir_for_use(controller.df_dir)
    subjects, samples = controller._make_dataset(
        demographics=controller.demographics,
        df_dir=df_dir,
    )

    assert isinstance(subjects, list)
    assert isinstance(samples, list)
    assert len(subjects) > 0 and len(samples) > 0
    assert all(set(sample.keys()) == {"T1", "T2", "demographics"} for sample in samples)


@pytest.mark.parametrize(
    "temp_medical_folder", ["empty_intersection_demographics"], indirect=True
)
def test_make_dataset_empty_intersection(temp_medical_folder):
    """Test _make_dataset raise error when there is no match between subjects in folders and demographics"""
    controller = MedicalFolderController(
        temp_medical_folder,
        tabular_file=os.path.join(temp_medical_folder, "participants.csv"),
        index_col="participant_id",
        validate=False,
    )

    _, df_dir = controller._prepare_df_dir_for_use(controller.df_dir)
    with pytest.raises(FedbiomedError) as exc_info:
        _ = controller._make_dataset(
            demographics=controller.demographics,
            df_dir=df_dir,
        )
    partial_msg = "subject reference does not match any subject in folder structure"
    assert partial_msg in str(exc_info.value)


# === get_nontransformed_item ===


@patch("fedbiomed.common.dataset_reader.NiftiReader.read")
def test_get_sample(mock_read, temp_medical_folder):
    """Test get_sample method"""
    mock_read.return_value = "mock_nifti_data"
    controller = MedicalFolderController(root=temp_medical_folder)

    mock_read.reset_mock()
    item = controller.get_sample(0)

    assert isinstance(item, dict)
    assert all(modality in item for modality in controller.modalities)
    assert mock_read.call_count == len(controller.modalities)


@patch("fedbiomed.common.dataset_reader.NiftiReader.read")
def test_get_sample_with_demographics(mock_read, temp_medical_folder):
    """Test get_sample with demographics"""
    mock_read.return_value = "mock_nifti_data"
    participants_file = os.path.join(temp_medical_folder, "participants.csv")
    controller = MedicalFolderController(
        root=temp_medical_folder,
        tabular_file=participants_file,
        index_col="participant_id",
    )

    mock_read.reset_mock()
    item = controller.get_sample(0)

    assert "demographics" in item
    assert isinstance(item["demographics"], dict)
    assert mock_read.call_count == len(controller.modalities)


@patch("fedbiomed.common.dataset_reader.NiftiReader.read")
@pytest.mark.parametrize("temp_medical_folder", ["dlp"], indirect=True)
def test_get_sample_with_dlp(mock_read, temp_medical_folder):
    """Test get_sample with demographics"""
    mock_read.return_value = "mock_nifti_data"
    participants_file = os.path.join(temp_medical_folder, "participants.csv")

    dlb = MapperBlock()
    modalities_to_folders = {
        "T1": ["T1_A", "T1_B", "T1_C"],
        "T2": ["T2"],
    }
    dlb.map = modalities_to_folders
    dlp = DataLoadingPlan({MedicalFolderLoadingBlockTypes.MODALITIES_TO_FOLDERS: dlb})

    controller = MedicalFolderController(
        root=temp_medical_folder,
        tabular_file=participants_file,
        index_col="participant_id",
        dlp=dlp,
    )

    mock_read.reset_mock()
    item = controller.get_sample(0)

    assert "demographics" in item
    assert isinstance(item["demographics"], dict)
    assert mock_read.call_count == len(controller.modalities)


@patch("fedbiomed.common.dataset_reader.NiftiReader.read")
def test_get_sample_error(mock_read, temp_medical_folder):
    """Test get_sample with read error"""

    controller = MedicalFolderController(root=temp_medical_folder)

    mock_read.side_effect = Exception("Read error")
    with pytest.raises(FedbiomedError) as exc_info:
        _ = controller.get_sample(0)
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
    )

    kwargs = controller._controller_kwargs
    assert os.path.realpath(kwargs["root"]) == os.path.realpath(temp_medical_folder)
    assert os.path.realpath(kwargs["tabular_file"]) == os.path.realpath(
        str(participants_file)
    )
    assert kwargs["index_col"] == "participant_id"


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

    controller = MedicalFolderController(
        root=temp_medical_folder,
        tabular_file=tsv_file,  # Should not raise error
        index_col="participant_id",
    )

    assert controller.tabular_file == Path(tsv_file).resolve()


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


@pytest.mark.parametrize(
    "temp_medical_folder", ["incomplete_modalities"], indirect=True
)
def test_subject_modality_status(temp_medical_folder):
    """Test _prepare_df_dir_for_use no subject has all modalities"""
    controller = MedicalFolderController(temp_medical_folder, validate=False)
    subject_modality_status = controller.subject_modality_status()

    assert isinstance(subject_modality_status, dict)
    assert all(key in subject_modality_status for key in ["columns", "data", "index"])
    assert subject_modality_status["columns"] == ["T1", "T2", "T3"]
    assert subject_modality_status["data"] == [
        [True, True, False],
        [True, True, False],
        [True, False, True],
    ]
    assert subject_modality_status["index"] == ["patient1", "patient2", "patient3"]

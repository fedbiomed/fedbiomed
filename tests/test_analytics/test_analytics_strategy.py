import pytest

from fedbiomed.common.analytics._analytics_strategy import (
    AnalyticsStrategy,
    validate_dataset_arguments_for_fa,
)
from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


class TestValidateDatasetArgumentsForFA:
    """Test the validate_dataset_arguments_for_fa function."""

    def test_validate_dataset_arguments_none_required_ok(self):
        """Test validation for dataset types that require no arguments (None passed)."""
        # DatasetTypes.IMAGES maps to None in DatasetArgumentsFA
        validate_dataset_arguments_for_fa(None, DatasetTypes.IMAGES)
        validate_dataset_arguments_for_fa(None, DatasetTypes.DEFAULT)
        validate_dataset_arguments_for_fa(None, DatasetTypes.MEDNIST)

    def test_validate_dataset_arguments_none_required_fail(self):
        """Test validation fails if arguments are provided for types that require None."""
        args = {"arg1": "value"}
        with pytest.raises(FedbiomedError) as excinfo:
            validate_dataset_arguments_for_fa(args, DatasetTypes.IMAGES)
        assert ErrorNumbers.FB633.value in str(excinfo.value)

    def test_validate_dataset_arguments_medical_folder_ok(self):
        """Test valid arguments for MEDICAL_FOLDER."""
        args = {"modalities": ["T1", "T2"]}
        validate_dataset_arguments_for_fa(args, DatasetTypes.MEDICAL_FOLDER)

    def test_validate_dataset_arguments_medical_folder_fail_missing_args(self):
        """Test validation fails if required arguments are missing for MEDICAL_FOLDER."""
        # Case 1: None arguments
        with pytest.raises(FedbiomedError) as excinfo:
            validate_dataset_arguments_for_fa(None, DatasetTypes.MEDICAL_FOLDER)
        assert "Missing required dataset argument" in str(excinfo.value)

        # Case 2: Empty arguments
        with pytest.raises(FedbiomedError) as excinfo:
            validate_dataset_arguments_for_fa({}, DatasetTypes.MEDICAL_FOLDER)
        assert "Missing required dataset argument 'modalities'" in str(excinfo.value)

        # Case 3: Wrong argument
        with pytest.raises(FedbiomedError) as excinfo:
            validate_dataset_arguments_for_fa(
                {"wrong_arg": 1}, DatasetTypes.MEDICAL_FOLDER
            )
        assert "Missing required dataset argument 'modalities'" in str(excinfo.value)

    def test_validate_dataset_arguments_medical_folder_fail_unexpected_args(self):
        """Test validation fails if unexpected arguments are provided for MEDICAL_FOLDER."""
        args = {"modalities": ["T1"], "extra_arg": 123}
        with pytest.raises(FedbiomedError) as excinfo:
            validate_dataset_arguments_for_fa(args, DatasetTypes.MEDICAL_FOLDER)
        assert ErrorNumbers.FB633.value in str(excinfo.value)
        assert "Unexpected dataset argument(s)" in str(excinfo.value)

    def test_validate_dataset_arguments_tabular_ok(self):
        """Test valid arguments for TABULAR."""
        # Optional argument provided
        args = {"col_names": ["age", "height"]}
        validate_dataset_arguments_for_fa(args, DatasetTypes.TABULAR)

        # Optional argument omitted (None) - Should pass as it is not required
        validate_dataset_arguments_for_fa(None, DatasetTypes.TABULAR)

        # Optional argument omitted (Empty dict) - Should pass
        validate_dataset_arguments_for_fa({}, DatasetTypes.TABULAR)

    def test_validate_dataset_arguments_tabular_fail_unexpected(self):
        """Test validation fails if unexpected arguments are provided for TABULAR."""
        args = {"col_names": [], "unexpected": "value"}
        with pytest.raises(FedbiomedError) as excinfo:
            validate_dataset_arguments_for_fa(args, DatasetTypes.TABULAR)
        assert ErrorNumbers.FB633.value in str(excinfo.value)


class TestAnalyticsStrategy:
    """Test the AnalyticsStrategy class."""

    def test_analytics_strategy_structure(self):
        """Test that AnalyticsStrategy has the expected method."""
        # Since it is conceptually abstract but technically instantiable in the current implementation
        # we can check it can be instantiated or methods inspected.
        strategy = AnalyticsStrategy()
        assert hasattr(strategy, "basic_stats")

        # Verify it returns None (implicit return of pass) or pass
        assert strategy.basic_stats() is None

    def test_analytics_strategy_inheritance(self):
        """Test inheritance and implementation of AnalyticsStrategy."""

        class MyStrategy(AnalyticsStrategy):
            def basic_stats(self, **kwargs):
                return {"mean": 0}

        s = MyStrategy()
        assert s.basic_stats() == {"mean": 0}

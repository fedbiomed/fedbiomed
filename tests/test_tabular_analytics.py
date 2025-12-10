from unittest.mock import Mock

import polars as pl
import pytest

from fedbiomed.common.analytics.tabular_analytics import TabularAnalytics
from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.dataset_controller import TabularController
from fedbiomed.common.dataset_reader import CsvReader
from fedbiomed.common.exceptions import FedbiomedError


@pytest.fixture
def analytics():
    """Fixture for TabularAnalytics instance"""
    return TabularAnalytics()


@pytest.fixture
def sample_df():
    """Fixture for sample dataframe"""
    return pl.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 70000, 80000, 90000],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "experience": [1, 3, 5, 7, 9],
        }
    )


@pytest.fixture
def mock_dataset(sample_df):
    """Fixture for mock dataset with controller and reader"""
    mock_reader = Mock(spec=CsvReader)
    mock_reader.data = sample_df

    mock_controller = Mock(spec=TabularController)
    mock_controller._reader = mock_reader

    mock_dataset = Mock(spec=TabularDataset)
    mock_dataset._controller = mock_controller

    return mock_dataset


class TestTabularAnalyticsMean:
    """Tests for TabularAnalytics.mean() method"""

    def test_mean_with_all_numeric_columns(self, analytics, mock_dataset):
        """Test mean calculation with all numeric columns"""
        result = analytics.mean(mock_dataset)

        # Verify result contains all numeric columns
        assert "age" in result
        assert "salary" in result
        assert "experience" in result
        assert "name" not in result

        # Verify values are correct
        assert result["age"][0] == pytest.approx(35.0)  # (25+30+35+40+45)/5
        assert result["salary"][0] == pytest.approx(70000.0)
        assert result["experience"][0] == pytest.approx(5.0)

    def test_mean_with_selected_columns(self, analytics, mock_dataset):
        """Test mean calculation with specific columns selected via params"""
        params = {"cols": ["age", "salary"]}
        result = analytics.mean(mock_dataset, params)

        assert "age" in result
        assert "salary" in result
        assert "experience" not in result
        assert "name" not in result

    def test_mean_with_single_numeric_column(self, analytics, mock_dataset):
        """Test mean with only one numeric column selected"""
        params = {"cols": ["age"]}
        result = analytics.mean(mock_dataset, params)

        assert "age" in result
        assert len(result) == 1
        assert result["age"][0] == pytest.approx(35.0)

    def test_mean_with_no_numeric_columns(self, analytics, mock_dataset):
        """Test mean when no numeric columns are selected"""
        params = {"cols": ["name"]}
        result = analytics.mean(mock_dataset, params)

        assert result == {}

    def test_mean_with_invalid_controller_type(self, analytics):
        """Test that error is raised when controller is not TabularController"""
        mock_dataset = Mock(spec=TabularDataset)
        mock_dataset._controller = Mock()  # Not a TabularController

        with pytest.raises(FedbiomedError, match="Expected TabularController"):
            analytics.mean(mock_dataset)

    def test_mean_with_invalid_reader_type(self, analytics):
        """Test that error is raised when reader is not CsvReader"""
        mock_controller = Mock(spec=TabularController)
        mock_controller._reader = Mock()  # Not a CsvReader

        mock_dataset = Mock(spec=TabularDataset)
        mock_dataset._controller = mock_controller

        with pytest.raises(FedbiomedError, match="Expected CsvReader"):
            analytics.mean(mock_dataset)

    def test_mean_with_none_data(self, analytics):
        """Test that error is raised when data is None"""
        mock_reader = Mock(spec=CsvReader)
        mock_reader.data = None

        mock_controller = Mock(spec=TabularController)
        mock_controller._reader = mock_reader

        mock_dataset = Mock(spec=TabularDataset)
        mock_dataset._controller = mock_controller

        with pytest.raises(FedbiomedError, match="Dataset is empty"):
            analytics.mean(mock_dataset)

    def test_mean_with_missing_columns(self, analytics, mock_dataset):
        """Test that error is raised when requested columns don't exist"""
        params = {"cols": ["age", "nonexistent_col"]}

        with pytest.raises(KeyError, match="Columns not found"):
            analytics.mean(mock_dataset, params)

    def test_mean_with_mixed_numeric_and_string_params(self, analytics, mock_dataset):
        """Test mean with mix of numeric and non-numeric columns in params"""
        params = {"cols": ["age", "name", "salary"]}
        result = analytics.mean(mock_dataset, params)

        # Only numeric columns should be in result
        assert "age" in result
        assert "salary" in result
        assert "name" not in result

    def test_mean_with_empty_dataframe(self, analytics):
        """Test mean with an empty dataframe"""
        empty_df = pl.DataFrame(
            {
                "age": pl.Series([], dtype=pl.Int64),
                "salary": pl.Series([], dtype=pl.Float64),
            }
        )

        mock_reader = Mock(spec=CsvReader)
        mock_reader.data = empty_df

        mock_controller = Mock(spec=TabularController)
        mock_controller._reader = mock_reader

        mock_dataset = Mock(spec=TabularDataset)
        mock_dataset._controller = mock_controller

        result = analytics.mean(mock_dataset)

        # Result should contain numeric columns
        assert "age" in result
        assert "salary" in result

    def test_mean_with_none_params(self, analytics, mock_dataset):
        """Test that None params defaults to all numeric columns"""
        result = analytics.mean(mock_dataset, params=None)

        # Should include all numeric columns
        assert "age" in result
        assert "salary" in result
        assert "experience" in result
        assert "name" not in result

    def test_mean_with_empty_cols_in_params(self, analytics, mock_dataset):
        """Test that empty cols in params defaults to all numeric columns"""
        params = {"cols": None}
        result = analytics.mean(mock_dataset, params)

        # Should include all numeric columns
        assert "age" in result
        assert "salary" in result

    def test_mean_with_floating_point_values(self, analytics):
        """Test mean calculation with floating point numbers"""
        df = pl.DataFrame({"values": [1.5, 2.5, 3.5, 4.5]})

        mock_reader = Mock(spec=CsvReader)
        mock_reader.data = df

        mock_controller = Mock(spec=TabularController)
        mock_controller._reader = mock_reader

        mock_dataset = Mock(spec=TabularDataset)
        mock_dataset._controller = mock_controller

        result = analytics.mean(mock_dataset)

        assert result["values"][0] == pytest.approx(3.0)

    def test_mean_with_negative_values(self, analytics):
        """Test mean calculation with negative values"""
        df = pl.DataFrame({"values": [-10, -5, 0, 5, 10]})

        mock_reader = Mock(spec=CsvReader)
        mock_reader.data = df

        mock_controller = Mock(spec=TabularController)
        mock_controller._reader = mock_reader

        mock_dataset = Mock(spec=TabularDataset)
        mock_dataset._controller = mock_controller

        result = analytics.mean(mock_dataset)

        assert result["values"][0] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "cols,expected_columns",
        [
            (["age"], ["age"]),
            (["age", "salary"], ["age", "salary"]),
            (["experience"], ["experience"]),
        ],
    )
    def test_mean_with_various_column_selections(
        self, analytics, mock_dataset, cols, expected_columns
    ):
        """Parametrized test for various column selections"""
        params = {"cols": cols}
        result = analytics.mean(mock_dataset, params)

        for col in expected_columns:
            assert col in result

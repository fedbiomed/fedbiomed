# tests/test_csv_reader.py
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from fedbiomed.common.dataset_reader._csv_reader import CsvReader
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedUserInputError


# --- Fixtures ---
@pytest.fixture
def sample_csv():
    csv_path = Path(__file__).parent / "test-data" / "csv" / "data.csv"
    csv_path.write_text(
        "id,name,age\n1,Alice,30\n2,Bob,40\n3,Charlie,50\n",
        encoding="utf-8",
    )
    yield csv_path
    csv_path.unlink()


@pytest.fixture
def no_header_csv():
    # No header; 3 columns, 3 rows
    csv_path = Path(__file__).parent / "test-data" / "csv" / "no_header.csv"
    csv_path.write_text(
        "1,Alice,30\n2,Bob,40\n3,Charlie,50\n",
        encoding="utf-8",
    )
    yield csv_path
    csv_path.unlink()


# --- _pre_parse tests ---
def test_csvreader_01_pre_parse_detects_delimiter_and_header(sample_csv, no_header_csv):
    reader = CsvReader(sample_csv)
    reader_no_header = CsvReader(no_header_csv)
    assert reader._delimiter == ","
    assert reader_no_header._delimiter == ","
    assert reader.header
    assert not reader_no_header.header


def test_csvreader_02_pre_parse_empty_file(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(FedbiomedError):
        CsvReader(p)


def test_csvreader_03_pre_parse_cannot_detect_delimiter(mocker, sample_csv):
    mocker.patch.object(csv.Sniffer, "sniff", side_effect=csv.Error)
    with pytest.raises(FedbiomedError):
        CsvReader(sample_csv)


def test_csvreader_04_pre_parse_cannot_detect_header(mocker, sample_csv):
    mocker.patch.object(csv.Sniffer, "has_header", side_effect=csv.Error)
    with pytest.raises(FedbiomedError):
        CsvReader(sample_csv)


# --- read tests ---
def test_csvreader_05_read(sample_csv, no_header_csv):
    reader1 = CsvReader(sample_csv)
    reader2 = CsvReader(no_header_csv)
    df1 = reader1.read()
    df2 = reader2.read()
    assert isinstance(df1, pl.DataFrame)
    assert isinstance(df2, pl.DataFrame)


def test_csvreader_06_read_csv_error_case(sample_csv, monkeypatch):
    """Test that read_csv raises an error when the file does not exist."""

    def raise_compute_error(*args, **kwargs):
        raise pl.exceptions.ComputeError("Mocked error")

    monkeypatch.setattr(pl, "read_csv", raise_compute_error)
    with pytest.raises(FedbiomedError):
        reader = CsvReader(path=sample_csv)
        reader.read()


# --- test to numpy and pandas ---
def test_csvreader_07_to_numpy_and_pandas(sample_csv):
    reader = CsvReader(sample_csv)
    df_numpy = reader.to_numpy()
    df_pandas = reader.to_pandas()
    assert isinstance(df_numpy, np.ndarray)
    assert isinstance(df_pandas, pd.DataFrame)


# --- _validate_path ---
def test_csvreader_08_validate(tmp_path: Path):
    bad_path = tmp_path / "missing.csv"
    reader = CsvReader(bad_path)
    # reader = CsvReader.__new__(CsvReader)
    # reader._path = bad_path
    with pytest.raises(FedbiomedError):
        reader.validate()


# --- shape and len ---
def test_csvreader_09_shape_and_len(sample_csv):
    reader = CsvReader(sample_csv)
    shp = reader.shape()
    assert isinstance(shp, dict)
    assert "csv" in shp
    rows, cols = shp["csv"]
    assert rows == 3
    assert cols == 3
    assert reader.len() == 3


def test_csvreader_10_no_header_csv_shape_and_len(no_header_csv):
    reader = CsvReader(no_header_csv)
    shp = reader.shape()
    assert isinstance(shp, dict)
    assert "csv" in shp
    rows, cols = shp["csv"]
    assert rows == 3
    assert cols == 3
    assert reader.len() == 3


# --- get / _read_single_entry ---
def test_csvreader_11_get_by_row_index(sample_csv):
    reader = CsvReader(sample_csv)
    out = reader.get(indexes=1)
    assert out.shape[0] == 1


def test_csvreader_12_get_row_index_out_of_range(sample_csv):
    reader = CsvReader(sample_csv)
    with pytest.raises(FedbiomedUserInputError):
        reader.get(indexes=[99])


def test_csvreader_13_get_with_columns_by_labels_and_int(sample_csv):
    reader = CsvReader(sample_csv)
    out = reader.get(indexes=[0, 2], columns=["name", "age"])
    assert out.shape == (2, 2)
    assert out.columns == ["name", "age"]

    out = reader.get(indexes=[0, 2], columns=0)
    assert out.shape == (2, 1)
    assert out.columns == ["id"]


def test_csvreader_14_get_with_columns_by_indices(sample_csv):
    reader = CsvReader(sample_csv)
    out = reader.get(indexes=[0, 2], columns=[0, 2])
    assert out.shape == (2, 2)
    assert out.columns == ["id", "age"]


def test_csvreader_15_get_with_columns_by_indices_no_header(no_header_csv):
    reader = CsvReader(no_header_csv)
    out = reader.get(indexes=[0, 2], columns=[0, 2])
    assert out.shape == (2, 2)
    assert out[1]["column_3"].item() == 50


def test_csvreader_16_get_column_index_out_of_range(no_header_csv):
    reader = CsvReader(no_header_csv)
    with pytest.raises(FedbiomedUserInputError):
        reader.get(indexes=[1], columns=[99])


def test_csvreader_17_get_column_name_not_found(sample_csv):
    reader = CsvReader(sample_csv)
    with pytest.raises(FedbiomedUserInputError):
        reader.get(indexes=[1], columns=["not_a_col"])

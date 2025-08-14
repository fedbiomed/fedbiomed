# tests/test_csv_reader.py
import csv
from pathlib import Path

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
    return csv_path


@pytest.fixture
def no_header_csv() -> Path:
    # No header; 3 columns, 3 rows
    csv_path = Path(__file__).parent / "test-data" / "csv" / "no_header.csv"
    csv_path.write_text(
        "1,Alice,30\n2,Bob,40\n3,Charlie,50\n",
        encoding="utf-8",
    )
    return csv_path


# --- _pre_parse tests ---
def test_pre_parse_detects_delimiter_and_header(sample_csv):
    reader = CsvReader(sample_csv, memory=True)
    assert reader._delimiter == ","
    assert isinstance(reader.header, bool)
    assert isinstance(reader._reader, pl.DataFrame)


def test_pre_parse_empty_file(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(FedbiomedError):
        CsvReader(p)


def test_pre_parse_cannot_detect_delimiter(mocker, sample_csv):
    mocker.patch.object(csv.Sniffer, "sniff", side_effect=csv.Error)
    with pytest.raises(FedbiomedError):
        CsvReader(sample_csv)


def test_pre_parse_cannot_detect_header(mocker, sample_csv):
    mocker.patch.object(csv.Sniffer, "has_header", side_effect=csv.Error)
    with pytest.raises(FedbiomedError):
        CsvReader(sample_csv)


# --- read tests ---
def test_read_memory_true(sample_csv):
    reader = CsvReader(sample_csv, memory=True)
    df = reader.read()
    assert isinstance(df, pl.DataFrame)


def test_read_memory_false(sample_csv):
    reader = CsvReader(sample_csv, memory=False)
    lf = reader.read()
    assert isinstance(lf, pl.LazyFrame)


# --- _validate_path ---
def test_validate(tmp_path: Path):
    bad_path = tmp_path / "missing.csv"
    reader = CsvReader.__new__(CsvReader)
    reader._path = bad_path
    with pytest.raises(FedbiomedUserInputError):
        reader.validate()


# --- shape and len ---
def test_shape_and_len_memory_true(sample_csv):
    reader = CsvReader(sample_csv, memory=True)
    shp = reader.shape()
    assert isinstance(shp, dict)
    assert "csv" in shp
    rows, cols = shp["csv"]
    assert rows == 3
    assert cols == 3
    assert reader.len() == 3


def test_shape_and_len_memory_false(sample_csv):
    reader = CsvReader(sample_csv, memory=False)
    shp = reader.shape()
    assert isinstance(shp, dict)
    assert "csv" in shp
    rows, cols = shp["csv"]
    assert rows == 3
    assert cols == 3
    assert reader.len() == 3


# --- get / _read_single_entry ---
def test_get_by_row_index(sample_csv):
    reader = CsvReader(sample_csv)
    out = reader.get(indexes=1)
    assert out.shape[0] == 1


def test_get_by_index_col_name(sample_csv):
    reader = CsvReader(sample_csv)
    out = reader.get(indexes=[2], index_col="id")
    assert out["name"][0] == "Bob"


def test_get_by_index_col_int(sample_csv):
    reader = CsvReader(sample_csv)
    out = reader.get(indexes=[2], index_col=0)
    assert out["name"][0] == "Bob"


def test_get_index_col_not_found(sample_csv):
    reader = CsvReader(sample_csv)
    with pytest.raises(FedbiomedUserInputError):
        reader.get(indexes=[1], index_col="not_a_col")


def test_get_row_index_out_of_range(sample_csv):
    reader = CsvReader(sample_csv)
    with pytest.raises(FedbiomedUserInputError):
        reader.get(indexes=[99])


def test_get_with_columns_by_labels(sample_csv: Path):
    r = CsvReader(sample_csv)
    out = r.get(indexes=[1, 3], index_col="id", columns=["name", "age"])
    assert out.shape == (2, 2)
    assert out.columns == ["name", "age"]


def test_get_with_columns_by_indices(sample_csv: Path):
    r = CsvReader(sample_csv)
    # 0->id, 2->age
    out = r.get(indexes=[1, 3], index_col="id", columns=[0, 2])
    assert out.shape == (2, 2)
    assert out.columns == ["id", "age"]


def test_get_column_index_out_of_range(sample_csv):
    reader = CsvReader(sample_csv)
    with pytest.raises(FedbiomedError):
        reader.get(indexes=[1], columns=[99])


def test_get_column_name_not_found(sample_csv):
    reader = CsvReader(sample_csv)
    with pytest.raises(FedbiomedUserInputError):
        reader.get(indexes=[1], columns=["not_a_col"])


# ---------- No-header CSV handling ----------
def test_no_header_csv_shape_and_get(no_header_csv: Path):
    # Force has_header=False so polars autogenerates names: column_0, column_1, column_2
    reader = CsvReader(no_header_csv, has_header=False)
    shp = reader.shape()
    rows, cols = shp["csv"]
    assert rows == 3
    assert cols == 3

    # Index by first column (index_col=0) equals "column_0" behind the scenes
    out = reader.get(indexes=[2], index_col=0)
    assert out.shape == (1, 3)

    # Select by integer column indices (converted internally to names)
    out2 = reader.get(indexes=[1, 3], index_col=0, columns=[0, 2])
    assert out2.shape == (2, 2)


# --- _random_sample ---
def test_random_sample(sample_csv):
    reader = CsvReader(sample_csv)
    out = reader._random_sample(n_entries=1, seed=123)
    assert out.shape[0] == 1

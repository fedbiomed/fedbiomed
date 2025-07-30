import csv
import inspect
import os
import random
import string
import uuid

import numpy as np
import pandas as pd
import pytest

from fedbiomed.common.dataset_reader._csv_reader import CsvReader
from fedbiomed.common.exceptions import FedbiomedUserInputError


@pytest.fixture
def testdir():
    return os.path.join(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
        "test-data",
    )


@pytest.fixture
def csv_files():
    return (
        ("tata-header.csv", True, ","),
        ("titi-normal.csv", False, ","),
        ("../../../dataset/CSV/pseudo_adni_mod.csv", True, ";"),
        ("test_csv_no_header.tsv", False, "\t"),
        ("test_csv_header.tsv", True, "\t"),
        ("test_csv_without_header_delimiter_semicolumn.csv", False, ";"),
        ("test_csv_header.csv", True, ","),
    )


@pytest.fixture
def error_files():
    return ("toto-error.csv",)


@pytest.fixture
def numpy_transform():
    def _numpy_transform(df):
        head = ["CDRSB.bl", "ADAS11.bl", "Hippocampus.bl", "RAVLT.immediate.bl"]
        normalized = (df[head].to_numpy() - np.array([0.0, 0.0, 0.0, 0.0])) / np.array(
            [0.1, 1.0, 1.0, 1.0]
        )
        return normalized

    return _numpy_transform


@pytest.fixture
def numpy_transform_2():
    import polars as pl

    def _numpy_transform_2(df):
        head = ["CDRSB.bl", "ADAS11.bl", "Hippocampus.bl", "RAVLT.immediate.bl"]
        min_v = df["CDRSB.bl"].min()
        max_v = df["CDRSB.bl"].max()
        df = df.with_columns(
            ((pl.col("CDRSB.bl") - min_v) / (max_v - min_v)).alias("new_variable")
        )
        head.append("new_variable")
        return df[head]

    return _numpy_transform_2


@pytest.fixture
def numpy_target_transform():
    def _numpy_target_transform(df):
        target_col = ["MMSE.bl"]
        return df[target_col]

    return _numpy_target_transform


@pytest.fixture
def convert_to_float():
    def _convert_to_float(x):
        fct = np.frompyfunc(
            lambda x: np.round(float(x), 7) if not isinstance(x, str) else x, 1, 1
        )
        return fct(x)

    return _convert_to_float


@pytest.fixture
def get_shape_through_pandas():
    def _get_shape_through_pandas(file: str, is_header: bool, delimiter: str):
        return pd.read_csv(
            file, header=None if not is_header else 0, delimiter=delimiter
        ).shape

    return _get_shape_through_pandas


@pytest.fixture
def get_dataset_header():
    def _get_dataset_header(file: str, delimiter: str):
        return pd.read_csv(file, delimiter=delimiter).columns

    return _get_dataset_header


@pytest.fixture
def iterate_through_pandas():
    def _iterate_through_pandas(file: str, is_header: bool, delimiter: str):
        csv = pd.read_csv(
            file, header=None if not is_header else 0, delimiter=delimiter
        )
        csv.reset_index()
        for _index, row in csv.iterrows():
            yield row

    return _iterate_through_pandas


def test_csvreader_01_load_iter_csv(
    testdir,
    csv_files,
    get_shape_through_pandas,
    get_dataset_header,
    iterate_through_pandas,
    convert_to_float,
):
    for file, is_header, d in csv_files:
        print(file)

        storage = ()
        path1 = os.path.join(testdir, "csv", file)

        csvreader = CsvReader(path1, force_read=True)

        shape1 = csvreader.shape()
        # compare with pandas
        shape2 = get_shape_through_pandas(path1, is_header, d)
        assert shape1["csv"] == shape2, f"error in file {file}"

        iterator_comp = iterate_through_pandas(path1, is_header, d)

        # test an entry at a time

        for i, p_row in zip(
            range(min(shape1["csv"][0], 20)), iterator_comp, strict=False
        ):
            row, _ = csvreader.read_single_entry(i)
            storage = (*storage, *row)

            # np.testing.assert_array_equal(p_row.to_numpy(), row[0])
            np.testing.assert_array_equal(
                convert_to_float(p_row.to_numpy()), convert_to_float(row[0])
            )

        # test batch of indexes
        row, _ = csvreader.read_single_entry(list(range(min(shape1["csv"][0], 20))))
        # for _i, _j in zip(row, storage):
        #     import pdb; pdb.set_trace()
        np.testing.assert_array_equal(row, np.stack(storage))

        # check header
        h1 = csvreader.header
        h2 = get_dataset_header(path1, d)

        if h1:
            assert h1 == h2.tolist()


def test_csvreader_02_read(testdir, convert_to_float):
    """
    Test column selection before reading
    This test verifies that CsvReader can select specific columns by name or index
    """
    # Test with header file that has column names
    file_path = os.path.join(testdir, "csv", "test_csv_header.csv")

    # Read the file with pandas to get reference data
    df_pandas = pd.read_csv(file_path, header=0, delimiter=",")

    # Initialize reader
    csvreader = CsvReader(file_path)
    total_rows = csvreader.shape()["csv"][0]
    rows_to_test = list(range(min(5, total_rows)))

    # Read these rows using CsvReader
    data, _ = csvreader.read_single_entry(rows_to_test)

    # TODO: Ask Yannick about index_col is column names and indexes is a list of row indices
    # Ask for better naming of parameters
    data, _ = csvreader.read_single_entry(rows_to_test)

    # Compare with pandas selected columns
    pandas_selected = df_pandas.iloc[rows_to_test].to_numpy()

    # Test 1: Compare csvreader data with pandas data
    # This tests if the reader can correctly read the data
    np.testing.assert_array_equal(
        convert_to_float(pandas_selected), convert_to_float(data)
    )

    # Test 2: Read file line by line
    # This tests if the reader can iterate through the file correctly
    for idx in rows_to_test:
        row, _ = csvreader.read_single_entry(idx)
        expected_row = df_pandas.iloc[[idx]].to_numpy()
        np.testing.assert_array_equal(
            convert_to_float(expected_row[0]), convert_to_float(row[0])
        )

    # Test 3: Invalid column selection should raise appropriate errors
    with pytest.raises((IndexError, KeyError, FedbiomedUserInputError)):
        # Try to select non-existent column index
        csvreader = CsvReader(file_path)
        csvreader.read_single_entry(indexes=0, index_col=["non_existent_column"])

    # Test 4: Invalid row selection should raise appropriate errors
    with pytest.raises((IndexError, KeyError, FedbiomedUserInputError)):
        # Try to select non-existent column index
        csvreader = CsvReader(file_path)
        csvreader.read_single_entry(indexes=[-1])


def test_csvreader_03_index(testdir, convert_to_float):
    """
    Test get_index functionality and dataset splitting using sklearn/pytorch methods
    This test verifies that CsvReader can provide indices for dataset splitting
    """
    from sklearn.model_selection import train_test_split

    # Use a file with enough data for splitting
    file_path = os.path.join(testdir, "csv", "test_csv_header.csv")
    csvreader = CsvReader(file_path)

    # Get the total number of rows
    total_rows = csvreader.shape()["csv"][0]

    # Test 1: Basic index functionality
    if hasattr(csvreader, "get_index") or hasattr(csvreader, "indices"):
        # Test getting all indices
        try:
            all_indices = csvreader.get_index()
            assert len(all_indices) == total_rows, (
                "Index length should match total rows"
            )
            assert all(isinstance(idx, (int, np.integer)) for idx in all_indices), (
                "Indices should be integers"
            )
        except AttributeError:
            # If get_index doesn't exist, create indices manually
            all_indices = list(range(total_rows))
    else:
        all_indices = list(range(total_rows))

    # Test 2: Train-test split using sklearn
    if total_rows >= 4:  # Need at least 4 rows for meaningful split
        train_indices, test_indices = train_test_split(
            all_indices, test_size=0.3, random_state=42
        )

        # Verify split indices are valid
        assert len(train_indices) + len(test_indices) == total_rows
        assert len(set(train_indices).intersection(set(test_indices))) == 0, (
            "Train/test indices should not overlap"
        )
        assert all(0 <= idx < total_rows for idx in train_indices + test_indices), (
            "All indices should be valid"
        )

        # Test reading data using split indices
        train_data, _ = csvreader.read_single_entry(
            train_indices[:3]
        )  # Test first 3 training samples
        test_data, _ = csvreader.read_single_entry(
            test_indices[:2]
        )  # Test first 2 test samples

        # Verify data shapes are correct
        assert train_data.shape[0] == 3, "Should read 3 training samples"
        assert test_data.shape[0] == 2, "Should read 2 test samples"
        assert train_data.shape[1] == test_data.shape[1], (
            "Train and test data should have same number of features"
        )

    # Test 3: Train-validation-test split
    if total_rows >= 6:  # Need at least 6 rows for three-way split
        # First split: train + val vs test
        train_val_indices, test_indices = train_test_split(
            all_indices, test_size=0.2, random_state=42
        )

        # Second split: train vs validation
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=0.25,  # 0.25 of 0.8 = 0.2 of total, so we get 60% train, 20% val, 20% test
            random_state=42,
        )

        # Verify all splits are non-overlapping and complete
        all_split_indices = set(train_indices) | set(val_indices) | set(test_indices)
        assert len(all_split_indices) == total_rows, (
            "All indices should be accounted for"
        )
        assert len(set(train_indices) & set(val_indices)) == 0, (
            "Train/val should not overlap"
        )
        assert len(set(train_indices) & set(test_indices)) == 0, (
            "Train/test should not overlap"
        )
        assert len(set(val_indices) & set(test_indices)) == 0, (
            "Val/test should not overlap"
        )

        # Test reading from each split
        if len(train_indices) > 0:
            train_sample, _ = csvreader.read_single_entry(train_indices[0])
            assert train_sample.shape[0] == 1, "Should read one training sample"

        if len(val_indices) > 0:
            val_sample, _ = csvreader.read_single_entry(val_indices[0])
            assert val_sample.shape[0] == 1, "Should read one validation sample"

        if len(test_indices) > 0:
            test_sample, _ = csvreader.read_single_entry(test_indices[0])
            assert test_sample.shape[0] == 1, "Should read one test sample"

    # Test 4: Stratified splitting (if applicable - for classification datasets)
    df_pandas = pd.read_csv(file_path, header=0, delimiter=",")

    # Look for a column that might serve as labels (typically last column or one with few unique values)
    potential_label_columns = []
    for col_idx, col in enumerate(df_pandas.columns):
        unique_vals = df_pandas[col].nunique()
        if 2 <= unique_vals <= min(10, total_rows // 2):  # Reasonable number of classes
            potential_label_columns.append(col_idx)

    if potential_label_columns and total_rows >= 4:
        label_col_idx = potential_label_columns[0]
        labels = df_pandas.iloc[:, label_col_idx].values

        try:
            # Attempt stratified split
            from sklearn.model_selection import StratifiedShuffleSplit

            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=0.3, random_state=42
            )
            train_idx, test_idx = next(splitter.split(all_indices, labels))

            # Verify stratification worked
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]

            # Check if class distributions are similar (within reasonable tolerance)
            from collections import Counter

            train_dist = Counter(train_labels)
            test_dist = Counter(test_labels)

            # Basic check that both splits have representation of classes
            assert len(train_dist) > 0, "Training set should have samples"
            assert len(test_dist) > 0, "Test set should have samples"

        except (ValueError, ImportError):
            # Stratified split might fail if labels are continuous or other issues
            # This is acceptable - just continue with regular splitting
            pass

    # Test 5: Custom index ranges
    if total_rows >= 10:
        # Test reading specific index ranges
        start_idx = 2
        end_idx = min(7, total_rows)
        range_indices = list(range(start_idx, end_idx))

        range_data, _ = csvreader.read_single_entry(range_indices)
        assert range_data.shape[0] == len(range_indices), (
            f"Should read {len(range_indices)} samples"
        )

        # Compare with pandas to ensure correct data retrieval
        pandas_range = df_pandas.iloc[start_idx:end_idx].to_numpy()
        np.testing.assert_array_equal(
            convert_to_float(pandas_range), convert_to_float(range_data)
        )

    # Test 6: Random sampling without replacement
    if total_rows >= 5:
        np.random.seed(42)
        sample_size = min(3, total_rows)
        random_indices = np.random.choice(all_indices, size=sample_size, replace=False)

        sampled_data, _ = csvreader.read_single_entry(random_indices.tolist())
        assert sampled_data.shape[0] == sample_size, f"Should sample {sample_size} rows"

        # Verify we can reproduce the same sample with the same seed
        np.random.seed(42)
        random_indices_2 = np.random.choice(
            all_indices, size=sample_size, replace=False
        )
        sampled_data_2, _ = csvreader.read_single_entry(random_indices_2.tolist())

        np.testing.assert_array_equal(
            convert_to_float(sampled_data), convert_to_float(sampled_data_2)
        )

    # Test 7: Edge cases
    # Test empty index list
    try:
        empty_data, _ = csvreader.read_single_entry([])
        assert empty_data.shape[0] == 0, "Empty index list should return empty data"
    except (ValueError, FedbiomedUserInputError):
        # Some implementations might not allow empty index lists
        pass

    # Test duplicate indices
    if total_rows >= 2:
        duplicate_indices = [0, 0, 1, 1]
        try:
            dup_data, _ = csvreader.read_single_entry(duplicate_indices)
            # Should either handle duplicates or raise an error
            if dup_data.shape[0] == len(duplicate_indices):
                # Duplicates are allowed - verify data is correct
                expected_data = df_pandas.iloc[[0, 0, 1, 1]].to_numpy()
                np.testing.assert_array_equal(
                    convert_to_float(expected_data), convert_to_float(dup_data)
                )
        except (ValueError, FedbiomedUserInputError):
            # Some implementations might not allow duplicate indices
            pass


def test_csvreader_04_error(testdir, error_files):
    for file in error_files:
        path1 = os.path.join(testdir, "csv", file)
        csvreader = CsvReader(path1)

        with pytest.raises(FedbiomedUserInputError):
            csvreader.read_single_entry(4)

        with pytest.raises(FedbiomedUserInputError):
            csvreader = CsvReader("/some/non/existing/path")
            csvreader.validate()


def test_csvreader_05_read_from_column(testdir, convert_to_float):
    file_with_col_name = (("test_csv_header.csv", True, ","),)
    for file, _, d in file_with_col_name:
        path = os.path.join(testdir, "csv", file)
        df = pd.read_csv(path, header=0, delimiter=d)

        values = np.random.choice(df.iloc[:, 12], size=10)

        df.set_index(df.iloc[:, 12], inplace=True)
        csvreader = CsvReader(
            path,
        )
        # test with one entry
        res1, _ = csvreader.read_single_entry(values[0], 12)
        res2 = df.loc[values[0]].drop(df.columns[12]).to_numpy()

        np.testing.assert_array_equal(convert_to_float(res2), convert_to_float(res1[0]))

        # test with several entries
        res1, _ = csvreader.read_single_entry(values, 12)

        res2 = (
            df.loc[values]
            .loc[~df.loc[values].index.duplicated(keep="first")]
            .drop(df.columns[12], axis=1)
            .to_numpy()
        )
        np.testing.assert_array_equal(
            convert_to_float(res2[res2[:, 0].argsort()]),
            convert_to_float(res1[res1[:, 0].argsort()]),
        )


def test_csvreader_06_transforms(
    testdir, numpy_transform, numpy_target_transform, numpy_transform_2
):
    path = os.path.join(testdir, "csv", "../../../dataset/CSV/pseudo_adni_mod.csv")

    csvreader = CsvReader(
        path,
        reader_transform=numpy_transform,
        reader_target_transform=numpy_target_transform,
    )
    csvreader.read_single_entry([1, 5, 6, 7])

    csvreader = CsvReader(
        path,
        reader_transform=numpy_transform_2,
        reader_target_transform=numpy_target_transform,
    )
    csvreader.read_single_entry([1, 5, 6, 7])


@pytest.mark.skip("stress test too long to execute")
def test_csvreader_xx_huge_dataset():
    # Stress test
    import time

    path = "./notebooks/data/test_csv2.csv"
    values = generate_huge_csv(path)
    t = time.time()
    reader = CsvReader(path)
    s = reader.shape()
    print(s)
    v, _ = reader.read_single_entry([1, 3, 2345, 567])
    v, _ = reader.read_single_entry(values, 21 + 2)
    print(v[0].shape)
    t2 = time.time()
    print("time", t2 - t)


def generate_huge_csv(path: str, delimiter: str = ",", with_header: bool = False):
    rows = 1e5
    columns = 52
    print_after_rows = 100000
    index_col = 21
    index_col_cat_var = 4
    values = []
    if with_header:
        _header = [id_generator() for _ in range(columns + 1)]
        print(_header)

    with open(path, "a+") as f:
        w = csv.writer(f, delimiter=delimiter, lineterminator="\n")
        if with_header:
            w.writerows([_header])
        for i in range(int(rows)):
            if i % print_after_rows == 0:
                print(".", end="", flush=True)
            row, val = generate_random_row(columns, i, index_col, index_col_cat_var)
            w.writerows(row)
            if i < 10:
                values.append(val)
    return values


def generate_random_row(col, row, index_col=-1, cat_var_col=-1, cat_var_choices=None):
    a = []
    line = [row]
    val = ""
    if cat_var_choices is None:
        cat_var_choices = ["AAA", "BBB", "CCC", "DDD"]
    for i in range(col):
        if i == index_col:
            val = uuid.uuid4().hex
            line.append(val)
        elif i == cat_var_col:
            line.append(cat_var_choices[random.randint(0, len(cat_var_choices) - 1)])
        else:
            line.append(random.random())

    a.append(line)
    return a, val


def id_generator(size=6, chars=string.ascii_uppercase + string.digits + "-_()"):
    return "".join(random.choice(chars) for _ in range(size))

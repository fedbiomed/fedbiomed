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


def test_csvreader_02_read():
    # Select column before reading
    pass


def test_csvreader_03_index():
    # Test get_index
    # here: try to split dataset using sklearn/pytorch methods
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

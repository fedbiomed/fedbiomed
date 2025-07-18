import csv
import inspect
import os
import random
import string
import unittest
import uuid

import numpy as np

from fedbiomed.common.dataset_reader._csv_reader import CsvReader
from fedbiomed.common.exceptions import FedbiomedUserInputError


class TestCsvReader(unittest.TestCase):
    def setUp(self):
        self.testdir = os.path.join(
            os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
            "test-data",
        )

        self.csv_files = (
            (
                "tata-header.csv",
                True,
                ",",
            ),
            (
                "titi-normal.csv",
                False,
                ",",
            ),
            ("../../../dataset/CSV/pseudo_adni_mod.csv", True, ";"),
            ("test_csv_no_header.tsv", False, "\t"),
            ("test_csv_header.tsv", True, "\t"),
            ("test_csv_without_header_delimiter_semicolumn.csv", False, ";"),
            ("test_csv_header.csv", True, ","),
        )
        self.error_files = ("toto-error.csv",)

        # idea : load more files (tsv, other delimiters, 2 headers)

    @staticmethod
    def numpy_transform(df):
        head = ["CDRSB.bl", "ADAS11.bl", "Hippocampus.bl", "RAVLT.immediate.bl"]
        normalized = (df[head].to_numpy() - np.array([0.0, 0.0, 0.0, 0.0])) / np.array(
            [0.1, 1.0, 1.0, 1.0]
        )

        return normalized

    @staticmethod
    def numpy_transform_2(df):
        import polars as pl

        head = ["CDRSB.bl", "ADAS11.bl", "Hippocampus.bl", "RAVLT.immediate.bl"]
        min_v = df["CDRSB.bl"].min()
        max_v = df["CDRSB.bl"].max()
        df = df.with_columns(
            ((pl.col("CDRSB.bl") - min_v) / (max_v - min_v)).alias("new_variable")
        )
        head.append("new_variable")
        return df[head]

    @staticmethod
    def numpy_target_transform(df):
        target_col = ["MMSE.bl"]
        return df[target_col]

    def tearDown(self):
        return super().tearDown()

    def _get_shape_through_pandas(self, file: str, is_header: bool, delimiter: str):
        import pandas as pd

        is_header = None if not is_header else 0

        return pd.read_csv(file, header=is_header, delimiter=delimiter).shape

    def _get_dataset_header(self, file: str, delimiter: str):
        import pandas as pd

        return pd.read_csv(file, delimiter=delimiter).columns

    def _iterate_through_pandas(self, file: str):
        import pandas as pd

        csv = pd.read_csv(file)
        csv.reset_index()
        for _index, row in csv.iterrows():
            yield row

    def test_csvreader_01_load_iter_csv(self):
        for file, is_header, d in self.csv_files:
            print(file)

            storage = ()
            path1 = os.path.join(self.testdir, "csv", file)

            csvreader = CsvReader(path1, force_read=True)

            shape1 = csvreader.shape()
            # compare with pandas
            shape2 = self._get_shape_through_pandas(path1, is_header, d)
            self.assertEqual(shape1["csv"], shape2, "error in file " + file)

            iterator_comp = self._iterate_through_pandas(path1)

            # test an entry at a time
            for i, p_row in zip(
                range(min(shape1["csv"][0], 20)), iterator_comp, strict=False
            ):
                row, _ = csvreader.read_single_entry(i)
                # csvreader.multithread_read_single_entry(i)
                storage = (*storage, *row)

                np.array_equal(p_row.tolist(), row[0])

            # test bacth of indexes
            row = csvreader.read_single_entry(list(range(min(shape1["csv"][0], 20))))
            # csvreader.multithread_read_single_entry(list(range(min(shape1['csv'][0], 20))))
            for _i, _j in zip(row, storage, strict=False):
                np.array_equal(_i, _j)

            # check header
            h1 = csvreader.header
            h2 = self._get_dataset_header(path1, d)

            if h1:
                self.assertListEqual(h1, h2.tolist())
        # print(d, shape1)

    def test_csvreader_02_read(self):
        # here select column before reading
        pass

    def test_csvreader_03_index(self):
        pass
        # test get_index

    def test_csvreader_04_error(self):
        for file in self.error_files:
            path1 = os.path.join(self.testdir, "csv", file)
            csvreader = CsvReader(path1)

            # with self.assertRaises(FedbiomedUserInputError):
            #     csvreader.read_single_entry(0)

            # csvreader = CsvReader(path1, force_read=True)
            # csvreader.read_single_entry(0)
            # with self.assertRaises(FedbiomedError):

            #     csvreader.read_single_entry(2)

            with self.assertRaises(FedbiomedUserInputError):
                # go beyond size of dataset
                csvreader.read_single_entry(4)

            with self.assertRaises(FedbiomedUserInputError):
                csvreader = CsvReader("/some/non/existing/path")
                csvreader.validate()

    def test_csvreader_05_read_single_entry_from_column(self):
        pass

    def test_csvreader_05_convert_to_framework(self):
        pass

    def test_csvreader_06_transforms(self):
        path = os.path.join(
            self.testdir, "csv", "../../../dataset/CSV/pseudo_adni_mod.csv"
        )
        csvreader = CsvReader(
            path,
            native_transform=self.numpy_transform,
            native_target_transform=self.numpy_target_transform,
        )
        csvreader.read_single_entry([1, 5, 6, 7])

        csvreader = CsvReader(
            path,
            native_transform=self.numpy_transform_2,
            native_target_transform=self.numpy_target_transform,
        )
        csvreader.read_single_entry([1, 5, 6, 7])

    def test_csvreader_xx_huge_dataset(self):
        # stress test

        self.skipTest(
            "stress test too long to execute"
        )  # test is skipped because too long to execute
        import time

        # path = "./notebooks/data/test_csv_indx_col.csv"
        path = "./notebooks/data/test_csv2.csv"
        # if not os.path.exists(path):
        values = generate_huge_csv(path)
        t = time.time()
        reader = CsvReader(path)
        s = reader.shape()
        print(s)
        v, _ = reader.read_single_entry([1, 3, 2345, 567])
        # v = reader.multithread_read_single_entry([1, 3, 2345, 567, 8, 45, 6789, 2345678, 34567,2345])
        v, _ = reader.read_single_entry_from_column(21 + 2, values)
        print(v[0].shape)
        t2 = time.time()
        print("time", t2 - t)
        # lasts for approx 15 sec on my computer, with a file of size 37GB


def id_generator(size=6, chars=string.ascii_uppercase + string.digits + "-_()"):
    return "".join(random.choice(chars) for _ in range(size))


def generate_huge_csv(path: str, delimiter: str = ",", with_header: bool = False):
    # 1000000 and 52 == roughly 1GB (WARNING TAKES a while, 30s+)
    rows = 1e5
    columns = 52
    print_after_rows = 100000
    index_col = 21
    index_col_cat_var = 4
    values = []
    if with_header:
        _header = []
        for _ in range(columns):
            _header.append(id_generator())
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


# TODO: add delimiter
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

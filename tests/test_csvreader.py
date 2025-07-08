import csv
import inspect
import os
import random
import unittest
import uuid

import numpy as np

from fedbiomed.common.dataset_reader._csv_reader import CsvReader
from fedbiomed.common.exceptions import FedbiomedUserInputError


class TestCSVReader(unittest.TestCase):
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
        )
        self.error_files = ("toto-error.csv",)

        # idea : load more files (tsv, other delimiters, 2 headers)

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

            csvreader = CsvReader(path1)

            shape1 = csvreader.shape()
            # compare with pandas
            shape2 = self._get_shape_through_pandas(path1, is_header, d)
            self.assertEqual(shape1["csv"], shape2, "error in file " + file)

            iterator_comp = self._iterate_through_pandas(path1)
            # test an entry at a time
            for i, p_row in zip(
                range(min(shape1["csv"][0], 20)), iterator_comp, strict=False
            ):
                row = csvreader.read_single_entry(i)
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

    def test_csvreader_05_convert_to_framework(self):
        pass

    def test_csvreader_06_(self):
        pass

    def test_csvreader_xx_huge_dataset(self):
        # stress test

        # self.skipTest('stress test too long to execute')  # test is skipped because too long to execute
        import time

        path = "./notebooks/data/test_csv_indx_col.csv"
        # path = "./notebooks/data/test_csv.csv"
        # if not os.path.exists(path):
        values = generate_huge_csv(path)
        t = time.time()
        reader = CsvReader(path)
        s = reader.shape()
        print(s)
        v = reader.read_single_entry([1, 3, 2345, 567])
        # v = reader.multithread_read_single_entry([1, 3, 2345, 567, 8, 45, 6789, 2345678, 34567,2345])
        v = reader.read_single_entry_from_column(21 + 2, values)
        print(v[0].shape)
        t2 = time.time()
        print("time", t2 - t)
        # lasts for approx 15 sec on my computer, with a file of size 37GB


def generate_huge_csv(path: str):
    # 1000000 and 52 == roughly 1GB (WARNING TAKES a while, 30s+)
    rows = 1e5
    columns = 52
    print_after_rows = 100000
    index_col = 21
    values = []
    with open(path, "a+") as f:
        w = csv.writer(f, lineterminator="\n")

        for i in range(int(rows)):
            if i % print_after_rows == 0:
                print(".", end="", flush=True)
            row, val = generate_random_row(columns, i, index_col)
            w.writerows(row)
            if i < 10:
                values.append(val)
    return values


# TODO: add delimiter
def generate_random_row(col, row, index_col=-1):
    a = []
    line = [row]
    val = ""
    for i in range(col):
        if i == index_col:
            val = uuid.uuid4().hex
            line.append(val)
        else:
            line.append(random.random())

    a.append(line)
    return a, val

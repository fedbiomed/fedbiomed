import inspect
import os
import unittest

import numpy as np

from fedbiomed.common.dataset_reader._csv_reader import CsvReader


class TestCSVReader(unittest.TestCase):
    def setUp(self):
        self.testdir = os.path.join(
            os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
            "test-data",
        )

        self.csv_files = (
            "tata-header.csv",
            "titi-normal.csv",
        )
        self.error_files = ("toto-error.csv",)

        # idea : load more files (tsv, other delimiters, 2 headers)

    def tearDown(self):
        return super().tearDown()

    def _get_shape_through_pandas(self, file: str):
        import pandas as pd

        return pd.read_csv(file).shape

    def _iterate_through_pandas(self, file: str):
        import pandas as pd

        csv = pd.read_csv(file)
        csv.reset_index()
        for _index, row in csv.iterrows():
            yield row

    def test_csvreader_01_load_iter_csv(self):
        for file in self.csv_files:
            print(file)
            storage = ()
            path1 = os.path.join(self.testdir, "csv", file)

            csvreader = CsvReader(path1)
            shape1 = csvreader.shape()
            # compare with pandas
            shape2 = self._get_shape_through_pandas(path1)
            self.assertEqual(shape1, shape2)

            iterator_comp = self._iterate_through_pandas(path1)
            # test an entry at a time
            for i, p_row in zip(range(shape1[0]), iterator_comp, strict=False):
                row = csvreader.read_single_entry(i)
                storage = (*storage, *row)
                np.array_equal(p_row.tolist(), row[0])

            row = csvreader.read_single_entry(list(range(shape1[0])))

            for _i, _j in zip(row, storage, strict=False):
                np.array_equal(_i, _j)
        # print(d, shape1)

    def test_csvreader_02_read(self):
        pass

    def test_csvreader_03_index(self):
        pass
        # test get_index

    def test_csvreader_04_error(self):
        for file in self.error_files:
            path1 = os.path.join(self.testdir, "csv", file)
            csvreader = CsvReader(path1)
            csvreader.read_single_entry(0)

from typing import Union

import pandas as pd


class Responses:
    def __init__(self, data: Union[list, dict]):
        if isinstance(data, dict):
            self._data = [data]
        elif isinstance(data, list):
            self._data = []
            for d in data:
                if d not in self._data:
                    self._data.append(d)

    @property
    def data(self):
        return(self._data)

    @property
    def dataframe(self):
        return pd.DataFrame(self._data)

    def get_data(self):
        return(self._data)

    def set_data(self, data):
        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def append(self, other):
        if isinstance(other, list):
            self._data = self._data + other
        elif isinstance(other, dict):
            self._data = self._data + [other]
        else:
            self._data = self._data + other.data  # what if other has no data member ?

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self._data)

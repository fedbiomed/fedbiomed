from typing import Union

import pandas as pd


class Responses:
    """Class parsing Nodes' responses.
    """
    def __init__(self, data: Union[list, dict]):
        """Constructor of `Responses` class. Reconfigures 
        input data into either a dictionary in a list (List[dict]), or
        a list with unique values

        Args:
            data (Union[list, dict]): input data
        """
        if isinstance(data, dict):
            self._data = [data]
        elif isinstance(data, list):
            self._data = []
            # create a list containing unique fields
            for d in data:
                if d not in self._data:
                    self._data.append(d)
                    # FIXME: would using ` set()` function be a better idea?

    @property
    def data(self) -> list:
        """setter

        Returns:
            list:  data of the class `Responses`
        """
        return(self._data)

    @property
    def dataframe(self) -> pd.DataFrame:
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

    def __repr__(self) -> str:
        """Makes Responses object representable
        (one can use built-in `repr()` function)

        Returns:
            str: the representation of the data
        """
        return repr(self._data)

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self._data)

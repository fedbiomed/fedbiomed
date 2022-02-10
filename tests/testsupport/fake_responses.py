from typing import Union


class FakeResponses:

    def __init__(self, data: Union[list, dict]):
        """Constructor of fake `Responses` class. Reconfigures
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

    def __getitem__(self, item):
        """  """
        return self._data[item]

import pandas as pd
from typing import Union, List, Dict
from fedbiomed.common.exceptions import FedbiomedResponsesError


class Responses:
    """Class parsing Nodes' responses Reconfigures input data into either a dictionary in a list (List[dict]), or
    a list with unique values.
    """

    def __init__(self, data: Union[list, dict]):
        """Constructor of `Responses` class.

        Args:
            data: input response
        """
        if isinstance(data, dict):
            self._data = [data]
        elif isinstance(data, list):
            self._data = []
            # create a list containing unique fields
            for d in data:
                if d not in self._data:
                    self._data.append(d)

    def __getitem__(self, item: int):
        """ Magic method to get item by index

        Args:
            item: List index

        Returns:
            Single response that comes for node
        """

        return self._data[item]

    def __repr__(self) -> str:
        """Makes Responses object representable (one can use built-in `repr()` function)

        Returns:
            The representation of the data
        """
        return repr(self._data)

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self._data)

    def data(self) -> list:
        """Gets all responses that are received

        Returns:
            Data of the class `Responses`
        """
        return self._data

    def set_data(self, data: List[Dict]) -> List:
        """Setter for ._data attribute

        Args:
            data: List of responses as python dictionary

        Raises:
            FedbiomedResponsesError: When `data` argument is not in valid type

        Returns:
            List of responses as Dict
        """

        # TODO: Check elements of list are Dict
        if not isinstance(data, List):
            raise FedbiomedResponsesError(f'`data` argument should instance of list not {type(data)}')

        self._data = data

        return self._data

    def dataframe(self) -> pd.DataFrame:
        """ This method converts the list that includes responses to pandas dataframe

        Returns:
             Pandas DataFrame includes node responses. Each row of dataframe represent single response that comes
                from a node.
        """

        return pd.DataFrame(self._data)

    def append(self, response: Union[List, Dict]) -> list:
        """ Appends new responses to existing responses

        Args:
            response: List of response as dict or single response as dict that will be appended
        Raises:
            FedbiomedResponsesError: When `response` argument is not in valid type

        Returns:
            List of dict as responses
        """
        if isinstance(response, List):
            self._data = self._data + response
        elif isinstance(response, Dict):
            self._data = self._data + [response]
        elif isinstance(response, self.__class__):
            self._data = self._data + response.data()
        else:
            raise FedbiomedResponsesError(f'`The argument must be instance of List, '
                                          f'Dict or Responses` not {type(response)}')

        return self._data

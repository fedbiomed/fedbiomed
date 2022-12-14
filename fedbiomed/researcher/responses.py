# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from typing import Any, TypeVar, Union, List, Dict
from fedbiomed.common.exceptions import FedbiomedResponsesError

_R = TypeVar('Responses')


class Responses:
    """Class parsing Nodes' responses Reconfigures input data into either a dictionary in a list (List[dict]), or
    a list with unique values.
    """

    def __init__(self, data: Union[list, dict]):
        """Constructor of `Responses` class.

        Args:
            data: input response
        """
        self._map_node: Dict[str, int] = {}
        if isinstance(data, dict):
            self._data = [data]
            self._update_map_node(data)
        elif isinstance(data, list):
            self._data = []
            # create a list containing unique fields
            for d in data:
                if d not in self._data:
                    self._data.append(d)
                    self._update_map_node(d)
                    
        else:
            raise FedbiomedResponsesError(f"data argument should be of type list or dict, not {type(data)}")

    def __getitem__(self, item: int) -> list:
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

        Returns:
            List of responses as Dict

        Raises:
            FedbiomedResponsesError: When `data` argument is not in valid type
        """

        # TODO: Check elements of list are Dict
        if not isinstance(data, List):
            raise FedbiomedResponsesError(f'`data` argument should instance of list not {type(data)}')

        self._data = data
        for datum in data:
            self._update_map_node(datum)
        return self._data

    def dataframe(self) -> pd.DataFrame:
        """ This method converts the list that includes responses to pandas dataframe

        Returns:
            Pandas DataFrame includes node responses. Each row of dataframe represent single response that comes
                from a node.
        """

        return pd.DataFrame(self._data)

    def append(self, response: Union[List[Dict], Dict, _R]) -> list:
        """ Appends new responses to existing responses

        Args:
            response: List of response as dict or single response as dict that will be appended

        Returns:
            List of dict as responses

        Raises:
            FedbiomedResponsesError: When `response` argument is not in valid type
        """

        if isinstance(response, List):
            #self._data = self._data + response
            for resp in response:
                if isinstance(resp, (dict, self.__class__)):
                    self.append(resp)
                else:
                    self._data = self._data + response
        elif isinstance(response, Dict):
            self._data = self._data + [response]
        elif isinstance(response, self.__class__):
            self._data = self._data + response.data()
        else:
            raise FedbiomedResponsesError(f'`The argument must be instance of List, '
                                          f'Dict or Responses` not {type(response)}')

        self._update_map_node(response)
        return self._data

    def _update_map_node(self, response: Union[Dict, _R]):
        """
        Updates an internal mapping, so one can get a specific node response index
        of a list of nodes responses. 

        Args:
            response (Union[Dict, _R]): a response from a node,
            that should contain a `'node_id'`argument

        """
        if isinstance(response, dict):
            _tmp_node_id = response.get('node_id')
            if _tmp_node_id is not None:
                self._map_node[_tmp_node_id] = len(self._data) - 1
        if isinstance(response, self.__class__):
            self._map_node.update(response._map_node)

    def get_index_from_node_id(self, node_id: str) -> Union[int, None]:
        """
        Helper that allows to retrieve the index of a given node_id,
        assuming that all content of the object Responses are nodes' answers

        Args:
            node_id (str): id of the node

        Returns:
            Union[int, None]: returns the index of the corresponding
            node_id. If not found, returns None
        """
        return self._map_node.get(node_id)

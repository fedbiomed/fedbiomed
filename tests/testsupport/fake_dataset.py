""" This file contains dummy Classes for unit testing. It fakes FederatedDataSet class
(from fedbiomed.common.dataset) 
"""

from typing import Any


class FederatedDataSetMock():
    """Provides an interface that behave like the FederatedDataset,
    with a constructor and a `data` method
    """
    def __init__(self, data: Any):
        self._data = data

    def data(self) -> Any:
        """Returns data values stored in FederatedDataSetMock

        Returns:
            Any: values stored in class
        """
        return self._data
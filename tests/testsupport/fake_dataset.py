"""This file contains dummy Classes for unit testing. It fakes FederatedDataset class
(from fedbiomed.common.dataset)
"""

from typing import Any


class FederatedDatasetMock:
    """Provides an interface that behave like the FederatedDataset,
    with a constructor and a `data` method
    """

    def __init__(self, data: Any):
        self._data = data
        self._node_ids = []

    def data(self) -> Any:
        """Returns data values stored in FederatedDatasetMock

        Returns:
            Any: values stored in class
        """
        return self._data

    def node_ids(self):
        return self._node_ids

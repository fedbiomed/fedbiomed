import uuid
from typing import List, Dict


class FederatedDataSet:
    """A class that allows researcher to interact with
    remote datasets (federated datasets).
    It contains details about remote datasets,
    such as client ids, data size that can be useful for
    aggregating or sampling strategies on researcher's side
    """
    def __init__(self, data: dict):
        self._data = data

    def data(self):
        return self._data

    @property
    def node_ids(self) -> List[uuid.UUID]:
        """Returns a list with client ids"""
        return list(self._data.keys())

    @property
    def sample_sizes(self) -> List[int]:
        """Returns a list with data sample sizes"""
        pass

    @property
    def shapes(self) -> Dict[uuid.UUID, int]:
        shapes_dict = {}
        for node_id, node_data_size in zip(self.node_ids,
                                           self.sample_sizes):
            shapes_dict[node_id] = node_data_size
        return shapes_dict

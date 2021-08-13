from typing import List, Dict


class FederatedDataSet:
    def __init__(self, data: dict):
        self._data = data

    def data(self):
        return self._data

    @property
    def client_ids(self) -> List:
        """Returns a list with client ids"""
        return list(self._data.keys())

    @property
    def sample_sizes(self) -> List:
        """Returns a list with data sample sizes"""
        pass

    @property
    def shapes(self) -> Dict:
        shapes_dict = {}
        for client_id, client_data_size in zip(self.client_ids, self.sample_sizes):
            shapes_dict[client_id] = client_data_size
        return shapes_dict

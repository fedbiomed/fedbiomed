'''
class which allows researcher to interact with remote datasets (federated datasets)
'''

from typing import Any, List, Dict, Union
import uuid
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedDatasetError


class FederatedDataSet:
    """
    A class that allows researcher to interact with
    remote datasets (federated datasets).
    It contains details about remote datasets,
    such as client ids, data size that can be useful for
    aggregating or sampling strategies on researcher's side
    """
    def __init__(self,
                 data: Dict[str, List[Dict[str, Any]]],
                 test_ratio: float = .0
                ):
        """
        Constructor
        """
        self._data = data

        # testing facility attributes
        self.set_test_ratio(test_ratio, False)

    def data(self) -> Dict:
        """
        Getter for FederatedDataset

        Returns:
            Dict: Dict of federated datasets, keys as node ids
        """
        return self._data

    def test_ratio(self) -> float:
        return self._test_ratio

    def set_test_ratio(self, ratio: float, overwrite: bool = False) -> float:
        """
        Sets testing ratio. 

        Args:
            - ratio (float): testing ratio, that MUST be within interval [0, 1]
            - overwrite (bool, optional): modify ratio for the datasets that already
                have a ratio value

        Returns:
            float: set testing ratio
        """
        self._test_ratio = ratio
        for _, node_datasets in self._data.items():
            for ds in node_datasets:
                if overwrite is True or 'test_ratio' not in ds:
                    ds.update({'test_ratio': self._test_ratio})

        return self._test_ratio

    def node_ids(self) -> List[uuid.UUID]:
        """
        Getter for Node ids

        Returns:
            List[str]: list of node ids
        """
        return list(self._data.keys())

    def sample_sizes(self) -> List[int]:
        """
        Returns a list with data sample sizes

        Returns:
            List[int]: List of sample sizes in federated datasets
                       in the same order with node_ids()
        """

        sample_sizes = []
        for (key, val) in self._data.items():
            sample_sizes.append(val[0]["shape"][0])

        return sample_sizes

    def shapes(self) -> Dict[uuid.UUID, int]:
        """
        Getter for shapes of FederatedDatasets by node ids

        Returns:
            Dict[str, int]: Dict that includes sample_sizes by node_ids
        """

        shapes_dict = {}
        for node_id, node_data_size in zip(self.node_ids(),
                                           self.sample_sizes()):
            shapes_dict[node_id] = node_data_size

        return shapes_dict

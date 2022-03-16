'''
class which allows researcher to interact with remote datasets (federated datasets)
'''
import copy

from typing import Any, List, Dict, Union
import uuid
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedDatasetError

from fedbiomed.common.logger import logger


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
                 test_ratio: Union[float, Dict[str, float]] = .0):
        """
        simple constructor
        """

        self._data = data
        self._check_data_format()
        # at this point, data can be only dict or None

        # testing facility attributes
        if self._data is not None:
            # retrieve the keys of the dictionary
            _key_list = list(data.keys())
            if _key_list and 'test_ratio' in data[_key_list[0]][0]:
                # FIXME: in this version, `test_ratio` should be the same 
                # for each nodes. We don't handle cases where `test_ratio`
                # appears only in some of the node entries
                self._test_ratio = data[_key_list[0]][0]['test_ratio']
            else:
                self.set_test_ratio(test_ratio)

        # self._test_metric = test_metric
        # self._test_metric_args = {} if test_metric_args is None else test_metric_args
        # self.test_on_global_updates = test_on_global_updates
        # self.test_on_local_updates = test_on_local_updates

    def _check_data_format(self):
        _is_data_structure_ok = True
        if self._data is not None:
            
            for node_id in self._data:

                if isinstance(self._data[node_id], list):
                    if not isinstance(self._data[node_id][0], dict):
                        _is_data_structure_ok = False
                else:
                    _is_data_structure_ok = False
        if not _is_data_structure_ok:
            raise FedbiomedDatasetError(ErrorNumbers.FB414.value + ". Expected data of type "
                                        f"Dict[str, List[Dict[str, Any]]], but got {self._data}")

    def data(self) -> Dict:
        """
        Getter for FederatedDataset

        Returns:
            Dict: Dict of federated datasets, keys as node ids
        """
        data = copy.deepcopy(self._data)  # prevent user to change FederatedDataset value through references
        return data
    
    def test_ratio(self) -> Union[float, Dict[str, float]]:
        return self._test_ratio
    
    def set_test_ratio(self, ratio: float) -> float:
        self._test_ratio = ratio
        
        for node_id in self._data.keys(): 
            
            self._data[node_id][0].update({'test_ratio': self._test_ratio})

        return self._test_ratio
    
    # def test_metric(self) -> Tuple[str, Dict[str, Any]]:
    #     return self._test_metric, self._test_metric_args
    
    # def set_test_metric(self,
    #                     metric: str,
    #                     metric_args: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    #     self._test_metric = metric
    #     self._test_metric_args = metric_args
    #     return self._test_metric, self._test_metric_args
    
    
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

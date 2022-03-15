'''
class which allows researcher to interact with remote datasets (federated datasets)
'''
import copy

from typing import Any, List, Dict, Optional, Tuple, Union
import uuid

from fedbiomed.common.exceptions import FedbiomedError


class FederatedDataSet:
    """
    A class that allows researcher to interact with
    remote datasets (federated datasets).
    It contains details about remote datasets,
    such as client ids, data size that can be useful for
    aggregating or sampling strategies on researcher's side
    """
    def __init__(self,
                 data: Dict,
                 test_ratio: Union[float, Dict[str, float]] = .0):
        """
        simple constructor
        """
        self._data = data
        
        # testing attributes
        
        if 'test_ratio' in data:
            self._test_ratio = data['test_ratio']
        else:
            self._test_ratio = test_ratio
        # self._test_metric = test_metric
        # self._test_metric_args = {} if test_metric_args is None else test_metric_args
        # self.test_on_global_updates = test_on_global_updates
        # self.test_on_local_updates = test_on_local_updates

    def data(self) -> Dict:
        """
        Getter for FederatedDataset

        Returns:
            Dict: Dict of federated datasets, keys as node ids
        """
        data = copy.deepcopy(self._data)
        if isinstance(data, dict):
            for node_id in data.keys(): 
                if 'test_ratio' not in data[node_id]:
                    print("FATASET", type(data[node_id]), type(data[node_id][0]))
                    if isinstance(data[node_id], dict):
                        data[node_id].update({'test_ratio': self._test_ratio})
                        
                    elif isinstance(data[node_id][0], dict):
                        data[node_id][0].update({'test_ratio': self._test_ratio})
                    else:
                        raise FedbiomedError()  
        return data
    
    def test_ratio(self) -> Union[float, Dict[str, float]]:
        return self._test_ratio
    
    def set_test_ratio(self, ratio: float) -> float:
        self._test_ratio = ratio
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

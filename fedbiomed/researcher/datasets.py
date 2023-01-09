# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Module includes the classes that allow researcher to interact with remote datasets (federated datasets)."""

from typing import List, Dict
import uuid


class FederatedDataSet:
    """A class that allows researcher to interact with remote datasets (federated datasets).

    It contains details about remote datasets, such as client ids, data size that can be useful for aggregating
    or sampling strategies on researcher's side
    """

    def __init__(self, data: Dict):
        """Construct FederatedDataSet object.

        Args:
            data: Dictionary of datasets. Each key represents single node, keys as node ids.
        """
        self._data = data

    def data(self) -> Dict:
        """Retrieve FederatedDataset as [`dict`][dict].

        Returns:
           Federated datasets, keys as node ids
        """
        return self._data

    def node_ids(self) -> List[uuid.UUID]:
        """Retrieve Node ids from `FederatedDataSet`.

        Returns:
            List of node ids
        """
        return list(self._data.keys())

    def sample_sizes(self) -> List[int]:
        """Retrieve list of sample sizes of node's dataset.

        Returns:
            List of sample sizes in federated datasets in the same order with
                [node_ids][fedbiomed.researcher.datasets.FederatedDataSet.node_ids]
        """
        sample_sizes = []
        for (key, val) in self._data.items():
            sample_sizes.append(val[0]["shape"][0])

        return sample_sizes

    def shapes(self) -> Dict[uuid.UUID, int]:
        """Get shape of FederatedDatasets by node ids.

        Returns:
            Includes [`sample_sizes`][fedbiomed.researcher.datasets.FederatedDataSet.sample_sizes] by node_ids.
        """
        shapes_dict = {}
        for node_id, node_data_size in zip(self.node_ids(),
                                           self.sample_sizes()):
            shapes_dict[node_id] = node_data_size

        return shapes_dict

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Module includes the classes that allow researcher to interact with remote datasets (federated datasets)."""

import copy
from asyncio.log import logger
from typing import Dict, List, Optional

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


class FederatedDataset:
    """A class that allows researcher to interact with remote datasets (federated datasets).

    It contains details about remote datasets, such as client ids, data size that can be useful for aggregating
    or sampling strategies on researcher's side, and the researcher-defined nodes filter that selects the
    nodes taking part in requests.
    """

    _data: Dict

    def __init__(self, data: Optional[Dict] = None):
        """Construct FederatedDataset object.

        Args:
            data:  Dictionary of datasets. Each key is a `str` representing a node's ID. Each value is
                a `dict` (or a `list` containing exactly one `dict`). Each `dict` contains the description
                of the dataset associated to this node in the federated dataset.
        """
        # researcher-defined nodes filter, shared by all components using this dataset
        self._nodes_filter: Optional[List[str]] = None

        # check structure of data

        if data is not None:
            self.set_federated_dataset(data)
        else:
            self._data = {}

    def set_federated_dataset(self, datasets: Dict) -> None:
        """Set federated dataset.

        Args:
            datasets:  Dictionary of datasets. Each key is a `str` representing a node's ID. Each value is
                a `dict` (or a `list` containing exactly one `dict`). Each `dict` contains the description
                of the dataset associated to this node in the federated dataset.

        Raises:
            FedbiomedError: bad `data` format
        """
        # check structure of data
        # DEPRECATED: to be removed in future versions
        if isinstance(datasets, FederatedDataset):
            logger.warning(
                "DEPRECATED: Passing a `FederatedDataset` instance"
                " to the `data` parameter of `FederatedDataset` is deprecated and "
                "will not be supported in future versions. Please pass a `dict` "
                "representing the federated dataset instead."
            )
            datasets = copy.deepcopy(datasets.data())

        if isinstance(datasets, dict) is False:
            raise FedbiomedError(
                f"{ErrorNumbers.FB416.value}: bad parameter `data` must be a `dict` of "
                f"(`list` of one) `dict`."
            )

        for node_id, node_data in datasets.items():
            if not (isinstance(node_data, dict) or isinstance(node_data, list)):
                raise FedbiomedError(
                    f"{ErrorNumbers.FB416.value}: bad parameter `data` for node {node_id}. "
                    f"Must be a `dict` or a `list` containing exactly one `dict`."
                )
            if isinstance(node_data, list):
                if len(node_data) != 1 or not isinstance(node_data[0], dict):
                    raise FedbiomedError(
                        f"{ErrorNumbers.FB416.value}: bad parameter `data` for node {node_id}. "
                        f"Must be a `dict` or a `list` containing exactly one `dict`."
                    )
                else:
                    datasets[node_id] = node_data[0]

        self._data = datasets

    @staticmethod
    def _dataset_type(value) -> bool:
        """Check if argument is a dict or a list.

        Args:
            value: argument to check.

        Returns:
            True if argument matches constraint, False if it does not.
        """
        return isinstance(value, dict) or isinstance(value, list)

    def data(self) -> Dict:
        """Retrieve FederatedDataset as [`dict`][dict].

        Returns:
           Federated datasets, keys as node ids
        """
        return self._data

    def node_ids(self) -> List[str]:
        """Retrieve Node ids from `FederatedDataset`.

        Returns:
            List of node ids
        """
        return list(self._data.keys())

    def set_nodes_filter(self, nodes: Optional[List[str]]) -> None:
        """Set the nodes filter. None keeps all nodes, an empty list keeps none."""
        self._nodes_filter = nodes

    def nodes_filter(self) -> Optional[List[str]]:
        """Retrieve the nodes filter, or None if no filter is set."""
        return self._nodes_filter

    def filtered_node_ids(self) -> List[str]:
        """Retrieve the ids of the nodes that pass the nodes filter."""
        if self._nodes_filter is None:
            return self.node_ids()
        return [node for node in self._data if node in self._nodes_filter]

    def sample_sizes(self) -> List[int]:
        """Retrieve list of sample sizes of node's dataset.

        Returns:
            List of sample sizes in federated datasets in the same order with
                [node_ids][fedbiomed.researcher.datasets.FederatedDataset.node_ids]
        """
        sample_sizes = []
        for _, val in self._data.items():
            sample_sizes.append(val["shape"][0])

        return sample_sizes

    def shapes(self) -> Dict[str, int]:
        """Get shape of FederatedDatasets by node ids.

        Returns:
            Includes [`sample_sizes`][fedbiomed.researcher.datasets.FederatedDataset.sample_sizes] by node_ids.
        """
        shapes_dict = {}
        for node_id, node_data_size in zip(
            self.node_ids(), self.sample_sizes(), strict=False
        ):
            shapes_dict[node_id] = node_data_size

        return shapes_dict


FederatedDataSet = FederatedDataset  # backward compatibility

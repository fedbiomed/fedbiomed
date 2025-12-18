# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Module includes the classes that allow researcher to interact with remote datasets (federated datasets)."""

from typing import Dict, List, Optional

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError


class FederatedDataSet:
    """A class that allows researcher to interact with remote datasets (federated datasets).

    It contains details about remote datasets, such as client ids, data size that can be useful for aggregating
    or sampling strategies on researcher's side
    """

    def __init__(self, data: Optional[Dict] = None):
        """Construct FederatedDataSet object.

        Args:
            data:  Dictionary of datasets. Each key is a `str` representing a node's ID. Each value is
                a `dict` (or a `list` containing exactly one `dict`). Each `dict` contains the description
                of the dataset associated to this node in the federated dataset.

        Raises:
            FedbiomedFederatedDataSetError: bad `data` format
        """
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
            FedbiomedFederatedDataSetError: bad `data` format
        """
        # check structure of data

        if isinstance(datasets, dict) is False:
            raise FedbiomedError(
                f"{ErrorNumbers.FB416.value}: bad parameter `data` must be a `dict` of "
                f"(`list` of one) `dict`."
            )

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
        for _, val in self._data.items():
            sample_sizes.append(val["shape"][0])

        return sample_sizes

    def shapes(self) -> Dict[str, int]:
        """Get shape of FederatedDatasets by node ids.

        Returns:
            Includes [`sample_sizes`][fedbiomed.researcher.datasets.FederatedDataSet.sample_sizes] by node_ids.
        """
        shapes_dict = {}
        for node_id, node_data_size in zip(
            self.node_ids(), self.sample_sizes(), strict=False
        ):
            shapes_dict[node_id] = node_data_size

        return shapes_dict

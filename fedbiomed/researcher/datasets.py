# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Module includes the classes that allow researcher to interact with remote datasets (federated datasets)."""

from typing import List, Dict

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedFederatedDataSetError
from fedbiomed.common.logger import logger
from fedbiomed.common.validator import Validator, ValidatorError


class FederatedDataSet:
    """A class that allows researcher to interact with remote datasets (federated datasets).

    It contains details about remote datasets, such as client ids, data size that can be useful for aggregating
    or sampling strategies on researcher's side
    """

    def __init__(self, data: Dict):
        """Construct FederatedDataSet object.

        Args:
            data: Dictionary of datasets. Each key is a `str` representing a node's ID. Each value is
                a `dict` (or a `list` containing exactly one `dict`). Each `dict` contains the description
                of the dataset associated to this node in the federated dataset. 

        Raises:
            FedbiomedFederatedDataSetError: bad `data` format
        """
        # check structure of data
        self._v = Validator()
        self._v.register("list_or_dict", self._dataset_type, override=True)
        try:
            self._v.validate(data, dict)
            for node, ds in data.items():
                self._v.validate(node, str)
                self._v.validate(ds, "list_or_dict")
                if isinstance(ds, list):
                    if len(ds) == 1:
                        self._v.validate(ds[0], dict)
                        # convert list of one dict to dict
                        data[node] = ds[0]
                    else:
                        errmess = f'{ErrorNumbers.FB416.value}: {node} must have one unique dataset ' \
                            f'but has {len(ds)} datasets.'
                        logger.error(errmess)
                        raise FedbiomedFederatedDataSetError(errmess)
        except ValidatorError as e:
            errmess = f'{ErrorNumbers.FB416.value}: bad parameter `data` must be a `dict` of ' \
                f'(`list` of one) `dict`: {e}'
            logger.error(errmess)
            raise FedbiomedFederatedDataSetError(errmess)

        self._data = data

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
        for (key, val) in self._data.items():
            sample_sizes.append(val["shape"][0])

        return sample_sizes

    def shapes(self) -> Dict[str, int]:
        """Get shape of FederatedDatasets by node ids.

        Returns:
            Includes [`sample_sizes`][fedbiomed.researcher.datasets.FederatedDataSet.sample_sizes] by node_ids.
        """
        shapes_dict = {}
        for node_id, node_data_size in zip(self.node_ids(),
                                           self.sample_sizes()):
            shapes_dict[node_id] = node_data_size

        return shapes_dict

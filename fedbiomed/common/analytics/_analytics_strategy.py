# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError

"""Classes and mappings to handle arguments mapping for different dataset types 
in analytics strategies.

None means no special arguments are needed for that dataset type.
"""
DatasetArgumentsFA = {
    DatasetTypes.MEDICAL_FOLDER: {
        "modalities": {"arg_name": "data_modalities", "required": True}
    },
    DatasetTypes.TABULAR: {
        "col_names": {"arg_name": "input_columns", "required": False}
    },
    DatasetTypes.IMAGES: None,
    DatasetTypes.DEFAULT: None,
    DatasetTypes.MEDNIST: None,
    DatasetTypes.CUSTOM: None,
}


def validate_dataset_arguments_for_fa(
    dataset_args: dict,
    dataset_type: DatasetTypes,
) -> None:
    """Validate dataset arguments for federated analytics.

    Args:
    dataset_args: Dataset arguments to validate
    """

    if DatasetArgumentsFA.get(dataset_type) is None:
        if dataset_args is not None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}: Dataset type '{dataset_type}' does not accept"
                f" dataset arguments, but some were provided: {dataset_args}"
            )
        return

    for arg, arg_info in DatasetArgumentsFA[dataset_type].items():
        # Check if required arguments are present
        if arg_info["required"] and (dataset_args is None or arg not in dataset_args):
            raise FedbiomedError(
                f"Missing required dataset argument '{arg}' for dataset type '{dataset_type}'."
            )
        # Detect unexpected arguments
        if (
            dataset_args
            and dataset_args.keys() - DatasetArgumentsFA[dataset_type].keys()
        ):
            raise FedbiomedError(
                f"{ErrorNumbers.FB633.value}:"
                f"Unexpected dataset argument(s) for dataset type '{dataset_type}'. "
                f"Expected arguments: {list(DatasetArgumentsFA[dataset_type].keys())}, "
                f"but got: {list(dataset_args.keys())}."
            )


class AnalyticsStrategy:
    @abstractmethod
    def basic_stats(self, **kwargs) -> Dict:
        pass

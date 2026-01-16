# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict, Iterable, Optional

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers, Stats
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

# Define dependencies between statistics
STATS_DEPENDENCIES = {
    Stats.MEAN: {Stats.COUNT},
    Stats.STD: {Stats.MEAN, Stats.COUNT},
    Stats.VARIANCE: {Stats.MEAN, Stats.COUNT},
    Stats.SUM: {Stats.COUNT},
}


def resolve_stats(requested: Optional[Iterable[str]] = None) -> Optional[set]:
    """Resolve all dependent statistics for a given set of requested statistics.

    Args:
        requested: Set of requested statistics.

    Returns:
        Set of resolved statistics including dependencies, or None if no statistics are requested.
    """
    if requested is None:
        return None

    # Cast to set for easier handling
    try:
        resolved = set(requested)
    except TypeError as e:
        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: Requested statistics argument is not iterable."
        ) from e

    # Validate requested statistics
    valid_values = {stat.value for stat in Stats}
    if not resolved.issubset(valid_values):
        raise FedbiomedError(
            f"{ErrorNumbers.FB633.value}: One or more requested statistics are not recognized: {requested}"
        )

    # Worklist algorithm for dependency resolution
    worklist = list(resolved)

    while worklist:
        stat_value = worklist.pop()

        # Convert string back to Enum to lookup dependencies
        # This is safe because we validated against valid_values
        stat_enum = Stats(stat_value)

        dependencies = STATS_DEPENDENCIES.get(stat_enum, set())
        for dep in dependencies:
            if dep.value not in resolved:
                resolved.add(dep.value)
                worklist.append(dep.value)

    return resolved


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

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Registry for dataset controllers and their parameters
"""

import inspect
from dataclasses import asdict, dataclass, fields
from typing import Optional

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset_controller import (
    Controller,
    CustomController,
    ImageFolderController,
    MedicalFolderController,
    MedNistController,
    MnistController,
    TabularController,
)
from fedbiomed.common.exceptions import FedbiomedError

from ._custom_dataset import CustomDataset
from ._medical_folder_dataset import MedicalFolderDataset
from ._simple_dataset import (
    ImageFolderDataset,
    MedNistDataset,
    MnistDataset,
)
from ._tabular_dataset import TabularDataset

DATASET_CLASSES_PER_TYPE = {
    DatasetTypes.CUSTOM: CustomDataset,
    DatasetTypes.IMAGES: ImageFolderDataset,
    DatasetTypes.MEDICAL_FOLDER: MedicalFolderDataset,
    DatasetTypes.MEDNIST: MedNistDataset,
    DatasetTypes.DEFAULT: MnistDataset,
    DatasetTypes.TABULAR: TabularDataset,
}


@dataclass
class ControllerParametersBase:
    root: str

    def to_dict(self) -> dict:
        """Convert entry to dictionary - removes None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict):
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)


@dataclass
class MedicalFolderParameters(ControllerParametersBase):
    tabular_file: Optional[str] = None
    index_col: Optional[str] = None
    dlp: Optional[DataLoadingPlan] = None


# Registry mapping data types to corresponding controller and expected parameters
REGISTRY_CONTROLLERS = {
    DatasetTypes.TABULAR: (
        TabularController,
        ControllerParametersBase,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.TABULAR],
    ),
    DatasetTypes.MEDICAL_FOLDER: (
        MedicalFolderController,
        MedicalFolderParameters,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.MEDICAL_FOLDER],
    ),
    DatasetTypes.IMAGES: (
        ImageFolderController,
        ControllerParametersBase,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.IMAGES],
    ),
    DatasetTypes.DEFAULT: (
        MnistController,
        ControllerParametersBase,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.DEFAULT],
    ),
    DatasetTypes.MEDNIST: (
        MedNistController,
        ControllerParametersBase,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.MEDNIST],
    ),
    DatasetTypes.CUSTOM: (
        CustomController,
        ControllerParametersBase,
        DATASET_CLASSES_PER_TYPE[DatasetTypes.CUSTOM],
    ),
}


def get_controller(
    data_type: str,
    controller_parameters: dict,
) -> Controller:
    """Get controller instance based on data_type and dataset_parameters"""
    # Validate that data_type is implemented.
    data_type = DatasetTypes.get_type_by_value(data_type)
    if not data_type or data_type not in REGISTRY_CONTROLLERS:
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: "
            f"Unknown 'data_type', implemented are: {list(REGISTRY_CONTROLLERS.keys())}"
        )

    controller_class, parameters_class, _ = REGISTRY_CONTROLLERS[data_type]

    # Validate and instantiate parameters
    try:
        parameters_instance = parameters_class.from_dict(controller_parameters)
    except Exception as e:
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: Failed to parse dataset_parameters: {str(e)}"
        ) from e

    try:
        return controller_class(**parameters_instance.to_dict())
    except FedbiomedError:
        raise
    except Exception as e:
        raise FedbiomedError(
            f"{ErrorNumbers.FB632.value}: Unhandled exception occurred: {str(e)}"
        ) from e


def validate_dataset_args(data_type: str, dataset_args: dict) -> None:
    """Validate dataset_parameters for a given data_type. Dataset constructors must have
    explicit arguments for all parameters that can be passed via dataset_args.

    Args:
        data_type: Dataset type as string_
        dataset_args: Arguments to validate against the dataset class constructor

    Raises:
        FedbiomedError: If dataset type is unsupported or arguments are invalid.
    """
    dataset_type = DatasetTypes.get_type_by_value(data_type)

    if dataset_type not in DATASET_CLASSES_PER_TYPE:
        raise FedbiomedError(f"Unsupported dataset type '{data_type}' for analytics.")

    dataset_cls = DATASET_CLASSES_PER_TYPE[dataset_type]

    # Get signature of the __init__ method
    sig = inspect.signature(dataset_cls.__init__)

    # Filter parameters to identify valid and required arguments
    valid_keys = set()
    required_keys = set()

    # Skip 'self' and consider only positional or keyword parameters
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            valid_keys.add(p.name)
            if p.default == inspect.Parameter.empty:
                required_keys.add(p.name)

    given_keys = set(dataset_args.keys())

    # Check for invalid keys: arguments provided but not expected by the constructor
    invalid_keys = given_keys - valid_keys
    if invalid_keys:
        raise FedbiomedError(
            f"Invalid dataset_args {invalid_keys} for dataset type '{data_type}'."
        )

    # Check for missing keys: required arguments that were not provided
    missing_keys = required_keys - given_keys
    if missing_keys:
        raise FedbiomedError(
            f"Missing required dataset_args {missing_keys} for dataset type '{data_type}'."
        )

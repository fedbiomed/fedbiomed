# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Registry for dataset controllers and their parameters
"""

from dataclasses import asdict, dataclass, fields
from typing import Optional

from fedbiomed.common.constants import DatasetTypes, ErrorNumbers
from fedbiomed.common.dataloadingplan import DataLoadingPlan
from fedbiomed.common.dataset import DATASET_CLASSES_PER_TYPE
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

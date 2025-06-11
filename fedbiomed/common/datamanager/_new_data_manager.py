# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data Management factory class
"""


from typing import Dict, Optional, Union, Any

from fedbiomed.common.exceptions import FedbiomedDataManagerError
from fedbiomed.common.constants import ErrorNumbers, TrainingPlans
from fedbiomed.common.dataset import Dataset
from ._torch_data_manager import TorchDataManager
from ._sklearn_data_manager import SkLearnDataManager


class DataManager(object):
    """Factory class that builds different data loaders

    Data loader type is based on the framework of the training plan.

    If `dataset` is not yet a `Dataset`, it also wraps it in a `NativeDataset` object.
    """

    # step 1 : the constructor starts by formatting the received dataset to a
    # Dataset (or raises an error if incorrect data format)
    # - either receives "dataset: Dataset, target: None": case of a structured
    #   dataset, no action required
    # - or receives "dataset: Any, target Union[Any, None]": case of
    #   unstructured dataset (with optional target). Instantiates a
    #   NativeDataset (which checks if native data format is correct and can
    #   be handled, and raises an error if this is not the case).
    def __init__(
        self,
        dataset: Union[Dataset, Any],
        target: Optional[Any] = None,
        **kwargs: dict
    ) -> None:

        """Constructor of DataManager,

        Args:
            dataset: Either a `Dataset` or the data component of unformatted dataset
            target: Target component of unformatted dataset.
            **kwargs: Additional parameters that are going to be used for data loader
        """

    # Same as current implementation ?
    def extend_loader_args(self, extension: Optional[Dict]):
        """Extends the class' loader arguments

        Extends the class's `_loader_arguments` attribute with additional key-values from
        the `extension` argument. If a key already exists in the `_loader_arguments`, then
        it is not replaced.

        Args:
            extension: the mapping used to extend the loader arguments
        """

    # step 2 : instantiate a DataManager depending of the TP type
    # (more simple than current implementation, already formatted to  Dataset )
    def load(self, tp_type: TrainingPlans):
        """Loads proper DataManager based on given TrainingPlan and
        `dataset`, `target` attributes.

        Args:
            tp_type: Enumeration instance of TrainingPlans that stands for type of training plan.

        Raises:
            FedbiomedDataManagerError: If requested DataManager does not match with given arguments.

        """


    # Same as current implementation ?
    def __getattr__(self, item: str):

        """Wraps all functions/attributes of factory class members.

        Args:
             item: Requested item from class

        Raises:
            FedbiomedDataManagerError: If the attribute is not implemented

        """

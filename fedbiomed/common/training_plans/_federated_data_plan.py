# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, TypeVar, TypedDict, Callable, Any, Union, Dict

from fedbiomed.common import utils
from fedbiomed.common.constants import ProcessTypes, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTrainingPlanError
from fedbiomed.common.logger import logger
from fedbiomed.common.training_plans._federated_plan import FederatedPlan


TDataLoader = TypeVar('TDataLoader')


class PreProcessDict(TypedDict):
    """Dict structure to specify a pre-processing transform."""

    method: Callable[..., Any]
    process_type: ProcessTypes


class FederatedDataPlan(FederatedPlan, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.dataset_path: Union[str, None] = None
        self.pre_processes: Dict[str, PreProcessDict] = OrderedDict()
        self.training_data_loader: Optional[TDataLoader] = None
        self.testing_data_loader: Optional[TDataLoader] = None
        self._loader_args: Dict[str, Any] = None

    def post_init(
            self,
            training_args: Dict[str, Any],
            **kwargs
    ) -> None:
        """Process model, training and optimizer arguments.

        Args:
            training_args: Arguments that are used in training routines
                such as epoch, dry_run etc.
                Please see [`TrainingArgs`][fedbiomed.common.training_args.TrainingArgs]
        """

        # Store various arguments provided by the researcher
        self._loader_args = training_args.loader_arguments() or {}

    def set_dataset_path(self, dataset_path: str) -> None:
        """Dataset path setter for TrainingPlan

        Args:
            dataset_path: The path where data is saved on the node.
                This method is called by the node that executes the training.
        """
        self.dataset_path = dataset_path
        logger.debug(f"Dataset path has been set as {self.dataset_path}")

    def set_data_loaders(
            self,
            train_data_loader: Optional[TDataLoader],
            test_data_loader:  Optional[TDataLoader]
    ) -> None:
        """Sets data loaders

        Args:
            train_data_loader: Data loader for training routine/loop
            test_data_loader: Data loader for validation routine
        """
        self.training_data_loader = train_data_loader
        self.testing_data_loader = test_data_loader

    @abstractmethod
    def training_data(self):
        """All subclasses must provide a training_data routine the purpose of this actual code is to detect
        that it has been provided
        """

    def add_preprocess(
            self,
            method: Callable,
            process_type: ProcessTypes
    ) -> None:
        """Register a pre-processing method to be executed on training data.

        Args:
            method: Pre-processing method to be run before training.
            process_type: Type of pre-processing that will be run.
                The expected signature of `method` and the arguments
                passed to it depend on this parameter.
        """
        if not callable(method):
            msg = (
                f"{ErrorNumbers.FB605.value}: error while adding "
                "preprocess, `method` should be callable."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        if not isinstance(process_type, ProcessTypes):
            msg = (
                f"{ErrorNumbers.FB605.value}: error while adding "
                "preprocess, `process_type` should be an instance "
                "of `fedbiomed.common.constants.ProcessTypes`."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # NOTE: this may be revised into a list rather than OrderedDict
        self.pre_processes[method.__name__] = {
            'method': method,
            'process_type': process_type
        }

    def _preprocess(self) -> None:
        """Executes registered data pre-processors."""
        for name, process in self.pre_processes.items():
            method = process['method']
            process_type = process['process_type']
            if process_type == ProcessTypes.DATA_LOADER:
                self._process_data_loader(method=method)
            else:
                logger.error(
                    f"Process type `{process_type}` is not implemented."
                    f"Preprocessor '{name}' will therefore be ignored."
                )

    def _process_data_loader(
            self,
            method: Callable[..., Any]
    ) -> None:
        """Handle a data-loader pre-processing action.

        Args:
            method: Process method that is to be executed.

        Raises:
            FedbiomedTrainingPlanError: If one of the following happens:
                - the method does not have 1 positional argument (dataloader)
                - running the method fails
                - the method does not return a dataloader of the same type as
                its input
        """
        # Check that the preprocessing method has a proper signature.
        argspec = utils.get_method_spec(method)
        if len(argspec) != 1:
            msg = (
                f"{ErrorNumbers.FB605.value}: preprocess method of type "
                "`PreprocessType.DATA_LOADER` sould expect one argument: "
                "the data loader wrapping the training dataset."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)
        # Try running the preprocessor.
        try:
            data_loader = method(self.training_data_loader)
        except Exception as exc:
            msg = (
                f"{ErrorNumbers.FB605.value}: error while running "
                f"preprocess method `{method.__name__}` -> {exc}"
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg) from exc
        logger.debug(
            f"The process `{method.__name__}` has been successfully executed."
        )
        # Verify that the output is of proper type and assign it.
        if isinstance(data_loader, type(self.training_data_loader)):
            self.training_data_loader = data_loader
            logger.debug(
                "Data loader for training routine has been updated "
                f"by the process `{method.__name__}`."
            )
        else:
            msg = (
                f"{ErrorNumbers.FB605.value}: the return type of the "
                f"`{method.__name__}` preprocess method was expected "
                f"to be {type(self.training_data_loader)}, but was "
                f"{type(data_loader)}."
            )
            logger.critical(msg)
            raise FedbiomedTrainingPlanError(msg)

    @staticmethod
    def _infer_batch_size(data: Union[dict, list, tuple, 'torch.Tensor', 'np.ndarray']) -> int:
        """Utility function to guess batch size from data.

        This function is a temporary fix needed to handle the case where
        Opacus changes the batch_size dynamically, without communicating
        it in any way.

        This will be improved by issue #422.

        Returns:
            the batch size for the input data
        """
        if isinstance(data, dict):
            # case `data` is a dict (eg {'modality1': data1, 'modality2': data2}):
            # compute length of the first modality
            return FederatedDataPlan._infer_batch_size(next(iter(data.values())))
        elif isinstance(data, (list, tuple)):
            return FederatedDataPlan._infer_batch_size(data[0])
        else:
            # case `data` is a torch.Tensor or a np.ndarray
            batch_size = len(data)
            return batch_size

    def loader_args(self) -> Dict[str, Any]:
        """Retrieve loader arguments

        Returns:
            Loader arguments
        """
        return self._loader_args


# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Data manager for scikit-learn training plan
"""

import math
from typing import List, Optional, Tuple

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.dataloader import SkLearnDataLoader
from fedbiomed.common.dataset import Dataset
from fedbiomed.common.dataset_types import DataReturnFormat
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger

from ._framework_data_manager import FrameworkDataManager


class SkLearnDataManager(FrameworkDataManager):
    """Class for creating data loaders from dataset for scikit-learn training plans"""

    _dataset: Dataset
    _loader_arguments: dict

    _subset_test: Optional[_Subset] = None
    _subset_train: Optional[_Subset] = None
    _training_index: List[int] = []
    _testing_index: List[int] = []
    _test_ratio: Optional[float] = None

    def __init__(self, dataset: Dataset, **kwargs: dict):
        """Class constructor

        Args:
            dataset: dataset object
            **kwargs: arguments for data loader
        """

        # SkLearnDataManager should get `dataset` argument as an instance of torch.utils.data.Dataset
        if not isinstance(dataset, Dataset):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: The argument `dataset` should be a "
                f"`fedbiomed.common.Dataset` object"
            )
        self._dataset = dataset

        self._loader_arguments = kwargs

        # TODO: randomizing / reproducibility
        #
        # rand_seed = kwargs.get('random_seed')
        # self.rng(rand_seed)

        self._dataset.to_format = DataReturnFormat.SKLEARN

    # Nota: used only for unit tests
    def subset_test(self) -> Optional[_Subset]:
        """Gets validation subset of the dataset.

        Returns:
            Validation subset
        """

        return self._subset_test

    # Nota: used only for unit tests
    def subset_train(self) -> Optional[_Subset]:
        """Gets train subset of the dataset.

        Returns:
            Train subset
        """
        return self._subset_train

    def split(
        self,
        test_ratio: float,
        test_batch_size: Optional[int],
        is_shuffled_testing_dataset: bool = False,
    ) -> Tuple[SkLearnDataLoader, Optional[SkLearnDataLoader]]:
        """Splitting scikit-learn Dataset into train and validation.

        Args:
            test_ratio: Split ratio for validation set ratio. Rest of the samples will be used for training
            test_batch_size: Batch size to use for testing subset
            is_shuffled_testing_dataset: if True, randomly select different samples for the testing
                subset at each execution. If False, reuse previous split when possible.
        Raises:
            FedbiomedError: Arguments bad format
            FedbiomedError: Cannot get number of samples from dataset

        Returns:
            train_loader: SkLearnDataLoader for training subset. `None` if the `test_ratio` is `1`
            test_loader: SkLearnDataLoader for validation subset. `None` if the `test_ratio` is `0`
        """
        # No need to check is_shuffled_testing_dataset, any argument can be interpreted as bool

        # Check the type of argument test_batch_size
        if not isinstance(test_batch_size, int) and test_batch_size is not None:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: The argument `test_batch_size` should be "
                f"type `int` or `None` not {type(test_batch_size)}"
            )

        # Check the argument `ratio` is of type `float`
        if not isinstance(test_ratio, (float, int)):
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: The argument `test_ratio` should be "
                f"type `float` or `int` not {type(test_ratio)}"
            )

        # Check ratio is valid for splitting
        if test_ratio < 0 or test_ratio > 1:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: The argument `test_ratio` should be "
                f"equal or between 0 and 1, not {test_ratio}"
            )

        # PREPARE code merge
        framework_dataset = self._dataset

        try:
            samples = len(framework_dataset)
        except AttributeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Can not get number of samples from "
                f"{str(self._dataset)} due to undefined attribute, {str(e)}"
            ) from e
        except TypeError as e:
            raise FedbiomedError(
                f"{ErrorNumbers.FB632.value}: Can not get number of samples from "
                f"{str(self._dataset)}, {str(e)}"
            ) from e

        if self._test_ratio != test_ratio and self._test_ratio is not None:
            if not is_shuffled_testing_dataset:
                logger.info(
                    "`test_ratio` value has changed: this will change the testing dataset"
                )
            is_shuffled_testing_dataset = True

        _is_loading_failed: bool = False
        # Calculate number of samples for train and validation subsets
        test_samples = math.floor(samples * test_ratio)
        if self._testing_index and not is_shuffled_testing_dataset:
            try:
                self._load_indexes(
                    framework_dataset, self._training_index, self._testing_index
                )
            except IndexError:
                _is_loading_failed = True
        if (
            not self._testing_index or is_shuffled_testing_dataset
        ) or _is_loading_failed:
            train_samples = samples - test_samples

            self._subset_train, self._subset_test = _dataset_random_split(
                framework_dataset,
                [train_samples, test_samples],
            )

            self._testing_index = list(self._subset_test.indices)
            self._training_index = list(self._subset_train.indices)

        if not test_batch_size and self._subset_test is not None:
            test_batch_size = len(self._subset_test)

        self._test_ratio = test_ratio

        loaders = (
            self._subset_loader(self._subset_train, **self._loader_arguments),
            self._subset_loader(self._subset_test, batch_size=test_batch_size),
        )

        return loaders

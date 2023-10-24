# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Optional, TypeVar
from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedUserInputError


TTrainingIterationsAccountant = TypeVar(
    "TTrainingIterationsAccountant",
    bound="MiniBatchTrainingIterationsAccountant"
)

TBaseTrainingPlan = TypeVar(
    "TBaseTrainingPlan",
    bound="BaseTrainingPlan"
)


class MiniBatchTrainingIterationsAccountant:
    """Accounting class for keeping track of training iterations.

    This class has the following responsibilities:

        - manage iterators for epochs and batches
        - provide up-to-date values for reporting
        - handle different semantics in case the researcher asked for num_updates or epochs

    We assume that the underlying implementation for the training loop is always made in terms of epochs and batches.
    So the primary purpose of this class is to provide a way to correctly convert the number of updates into
    epochs and batches.

    For reporting purposes, in the case of num_updates then we think of the training as a single big loop, while
    in the case of epochs and batches we think of it as two nested loops. This changes the meaning of the values
    outputted by the reporting functions (see their docstrings for more details).

    Attributes:
        _training_plan: a reference to the training plan executing the training iterations
        cur_epoch: the index of the current epoch during iterations
        cur_batch: the index of the current batch during iterations
        epochs: the total number of epochs to be performed (we always perform one additional -- possibly empty -- epoch
        num_batches_per_epoch: the number of iterations per epoch
        num_batches_in_last_epoch: the number of iterations in the last epoch (can be zero)
        num_samples_observed_in_epoch: a counter for the number of samples observed in the current epoch, for reporting
        num_samples_observed_in_total: a counter for the number of samples observed total, for reporting
    """
    def __init__(self, training_plan: TBaseTrainingPlan):
        """Initialize the class.

        Arguments:
            training_plan: a reference to the training plan that is executing the training iterations
        """
        self._training_plan = training_plan
        self.cur_epoch: int = 0
        self.cur_batch: int = 0
        self.epochs: int = 0
        self.num_batches_per_epoch: int = 0
        self.num_batches_in_last_epoch: int = 0
        self.num_samples_observed_in_epoch: int = 0
        self.num_samples_observed_in_total: int = 0
        self._n_training_iterations()

    def num_batches_in_this_epoch(self) -> int:
        """Returns the number of iterations to be performed in the current epoch"""
        if self.cur_epoch == self.epochs:
            return self.num_batches_in_last_epoch
        else:
            return self.num_batches_per_epoch

    def increment_sample_counters(self, n_samples: int):
        """Increments internal counter for numbers of observed samples"""
        self.num_samples_observed_in_epoch += n_samples
        self.num_samples_observed_in_total += n_samples

    def reporting_on_num_samples(self) -> Tuple[int, int]:
        """Outputs useful reporting information about the number of observed samples

        If the researcher specified num_updates, then the number of observed samples will be the grand total, and
        similarly the maximum number of samples will be the grand total over all iterations.
        If the researcher specified epochs, then both values will be specific to the current epoch.

        Returns:
            the number of samples observed until the current iteration
            the maximum number of samples to be observed
        """
        # get batch size
        if 'batch_size' in self._training_plan.loader_args():
            batch_size = self._training_plan.loader_args()['batch_size']
        else:
            raise FedbiomedUserInputError('Missing required key `batch_size` in `loader_args`.')
        # compute number of observed samples
        if self._training_plan.training_args()['num_updates'] is not None:
            num_samples = self.num_samples_observed_in_total
            total_batches_to_be_observed = (self.epochs - 1) * self.num_batches_per_epoch + \
                self.num_batches_in_last_epoch
            total_n_samples_to_be_observed = batch_size * total_batches_to_be_observed
            num_samples_max = total_n_samples_to_be_observed
        else:
            num_samples = self.num_samples_observed_in_epoch
            num_samples_max = batch_size*self.num_batches_in_this_epoch() if \
                self.cur_batch < self.num_batches_in_this_epoch() else num_samples
        return num_samples, num_samples_max

    def reporting_on_num_iter(self) -> Tuple[int, int]:
        """Outputs useful reporting information about the number of iterations

        If the researcher specified num_updates, then the iteration number will be the cumulated total, and
        similarly the maximum number of iterations will be equal to the requested number of updates.
        If the researcher specified epochs, then the iteration number will be the batch index in the current epoch,
        while the maximum number of iterations will be computed specifically for the current epoch.

        Returns:
            the iteration number
            the maximum number of iterations to be reported
        """
        if self._training_plan.training_args()['num_updates'] is not None:
            num_iter = (self.cur_epoch - 1) * self.num_batches_per_epoch + self.cur_batch
            total_batches_to_be_observed = (self.epochs - 1) * self.num_batches_per_epoch + \
                self.num_batches_in_last_epoch
            num_iter_max = total_batches_to_be_observed
        else:
            num_iter = self.cur_batch
            num_iter_max = self.num_batches_per_epoch
        return num_iter, num_iter_max

    def reporting_on_epoch(self) -> Optional[int]:
        """Returns the optional index of the current epoch, for reporting."""
        if self._training_plan.training_args()['num_updates'] is not None:
            return None
        else:
            return self.cur_epoch

    def should_log_this_batch(self) -> bool:
        """Whether the current batch should be logged or not.

        A batch shall be logged if at least one of the following conditions is True:

            - the cumulative batch index is a multiple of the logging interval
            - the dry_run condition was specified by the researcher
            - it is the last batch of the epoch
            - it is the first batch of the epoch
        """
        current_iter = (self.cur_epoch - 1) * self.num_batches_per_epoch + self.cur_batch
        return (current_iter % self._training_plan.training_args()['log_interval'] == 0 or
                self._training_plan.training_args()['dry_run'] or
                self.cur_batch >= self.num_batches_in_this_epoch() or  # last batch
                self.cur_batch == 1)  # first batch

    def _n_training_iterations(self):
        """Computes the number of training iterations from the training arguments given by researcher.

        This function assumes that a training plan's dataloader has already been created.
        If `num_updates` is specified, both epoch and `batch_maxnum` are ignored.

        Raises:
            FedbiomedUserInputError if neither num_updates nor epochs were specified.

        Returns:
            number of epochs
            number of batches in last epoch
            number of batches per epoch
        """
        num_batches_per_epoch = len(self._training_plan.training_data_loader)
        # override number of batches per epoch if researcher specified batch_maxnum
        if self._training_plan.training_args()['batch_maxnum'] is not None and \
                self._training_plan.training_args()['batch_maxnum'] > 0:
            num_batches_per_epoch = min(num_batches_per_epoch, self._training_plan.training_args()['batch_maxnum'])
        # first scenario: researcher specified epochs
        if self._training_plan.training_args()['num_updates'] is None:
            if self._training_plan.training_args()['epochs'] is not None:
                epochs = self._training_plan.training_args()['epochs']
                num_batches_in_last_epoch = 0
            else:
                msg = f'{ErrorNumbers.FB605.value}. Must specify one of num_updates or epochs.'
                logger.critical(msg)
                raise FedbiomedUserInputError(msg)
        # second scenario: researcher specified num_updates
        else:
            if self._training_plan.training_args()['epochs'] is not None:
                logger.warning('Both epochs and num_updates specified. num_updates takes precedence.',
                               broadcast=True)
            if self._training_plan.training_args()['batch_maxnum'] is not None:
                logger.warning('Both batch_maxnum and num_updates specified. batch_maxnum will be ignored.',
                               broadcast=True)
                # revert num_batches_per_epoch to correct value, ignoring batch_maxnum
                num_batches_per_epoch = len(self._training_plan.training_data_loader)
            epochs = self._training_plan.training_args()['num_updates'] // num_batches_per_epoch
            num_batches_in_last_epoch = self._training_plan.training_args()['num_updates'] % num_batches_per_epoch

        self.epochs = epochs + 1
        self.num_batches_in_last_epoch = num_batches_in_last_epoch
        self.num_batches_per_epoch = num_batches_per_epoch

    class EpochsIter:
        """Iterator over epochs.

        Attributes:
            _accountant: an instance of the class that created this iterator
        """
        def __init__(self, accountant: TTrainingIterationsAccountant):
            self._accountant = accountant

        def __next__(self):
            """Performs next epoch iteration

            This function also resets the batch counter and other reporting attributes

            Raises:
                StopIteration: when the total number of epochs has been exhausted
            """
            self._accountant.cur_epoch += 1
            self._accountant.num_samples_observed_in_epoch = 0
            self._accountant.cur_batch = 0
            if self._accountant.cur_epoch > self._accountant.epochs:
                raise StopIteration
            return self._accountant.cur_epoch

        def __iter__(self):
            """Returns this iterator's instance."""
            self._accountant.cur_epoch = 0
            return self

    class BatchIter:
        """Iterator over batches.

        Attributes:
            _accountant: an instance of the class that created this iterator
        """
        def __init__(self, accountant: TTrainingIterationsAccountant):
            self._accountant = accountant

        def __next__(self):
            """Performs next batch iteration

            Raises:
                StopIteration: when the total number of epochs has been exhausted
            """
            self._accountant.cur_batch += 1
            if self._accountant.cur_batch > self._accountant.num_batches_in_this_epoch():
                raise StopIteration
            return self._accountant.cur_batch

        def __iter__(self):
            """Returns this iterator's instance."""
            self._accountant.cur_batch = 0
            return self

    def iterate_epochs(self):
        """Returns an instance of an epochs iterator."""
        return MiniBatchTrainingIterationsAccountant.EpochsIter(self)

    def iterate_batches(self):
        """Returns an instance of a batches iterator."""
        return MiniBatchTrainingIterationsAccountant.BatchIter(self)

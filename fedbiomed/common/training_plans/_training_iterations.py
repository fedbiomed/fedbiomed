from typing import Tuple
from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedUserInputError


class MiniBatchTrainingIterationsAccountant:
    def __init__(self, training_plan):
        self._training_plan = training_plan
        self.cur_epoch: int = 0
        self.cur_batch: int = 0
        self.epochs: int = 0
        self.num_batches_per_epoch: int = 0
        self.num_batches_in_last_epoch: int = 0
        self.num_samples_observed_in_epoch: int = 0
        self.num_samples_observed_in_total: int = 0
        self._n_training_iterations()

    def num_batches_in_this_epoch(self):
        if self.cur_epoch == self.epochs:
            return self.num_batches_in_last_epoch
        else:
            return self.num_batches_per_epoch

    def increment_sample_counters(self, n_samples):
        self.num_samples_observed_in_epoch += n_samples
        self.num_samples_observed_in_total += n_samples

    def reporting_on_num_samples(self):
        if self._training_plan.training_args()['num_updates'] is not None:
            num_samples = self.num_samples_observed_in_total
            total_batches_to_be_observed = (self.epochs - 1) * self.num_batches_per_epoch + \
                self.num_batches_in_last_epoch
            total_n_samples_to_be_observed = \
                self._training_plan.training_args()['batch_size'] * total_batches_to_be_observed
            num_samples_max = total_n_samples_to_be_observed if \
                self.cur_batch < self.num_batches_in_this_epoch() else num_samples
        else:
            num_samples = self.num_samples_observed_in_epoch
            num_samples_max = self._training_plan.training_args()['batch_size']*self.num_batches_in_this_epoch() if \
                self.cur_batch < self.num_batches_in_this_epoch() else num_samples
        return num_samples, num_samples_max

    def reporting_on_num_iter(self):
        if self._training_plan.training_args()['num_updates'] is not None:
            num_iter = (self.cur_epoch - 1) * self.num_batches_per_epoch + self.cur_batch
            total_batches_to_be_observed = self.epochs * self.num_batches_per_epoch + self.num_batches_in_last_epoch
            num_iter_max = total_batches_to_be_observed
        else:
            num_iter = self.cur_batch
            num_iter_max = self.num_batches_per_epoch
        return num_iter, num_iter_max

    def reporting_on_epoch(self):
        if self._training_plan.training_args()['num_updates'] is not None:
            return None
        else:
            return self.cur_epoch

    def should_log_this_batch(self):
        current_iter = (self.cur_epoch - 1) * self.num_batches_per_epoch + self.cur_batch
        return (current_iter % self._training_plan.training_args()['log_interval'] == 0 or
                self._training_plan.training_args()['dry_run'] or
                self.cur_batch >= self.num_batches_in_this_epoch() or  # last batch
                self.cur_batch == 1)  # first batch

    def _n_training_iterations(self) -> Tuple[int, int, int]:
        """Computes the number of training iterations from the arguments given by researcher.

        This function assumes that a training dataloader has already been created.
        If `num_updates` is specified, both epoch and `batch_maxnum` are ignored.

        Args:
            training_args: Training arguments for one round training.

        Raises:
            FedbiomedUserInputError if neither num_updates nor epochs were specified.

        Returns:
            number of epochs
            number of remainder batches
            number of batches per epoch
        """
        num_batches_per_epoch = len(self._training_plan.training_data_loader)
        # override number of batches per epoch if researcher specified batch_maxnum
        if self._training_plan.training_args()['batch_maxnum'] is not None and \
                self._training_plan.training_args()['batch_maxnum'] > 0:
            num_batches_per_epoch = self._training_plan.training_args()['batch_maxnum']
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
                logger.warning('Both epochs and num_updates specified. num_updates takes precedence.')
            if self._training_plan.training_args()['batch_maxnum'] is not None:
                logger.warning('Both batch_maxnum and num_updates specified. batch_maxnum will be ignored.')
                # revert num_batches_per_epoch to correct value, ignoring batch_maxnum
                num_batches_per_epoch = len(self._training_data_loader)
            epochs = self._training_plan.training_args()['num_updates'] // num_batches_per_epoch
            num_batches_in_last_epoch = self._training_plan.training_args()['num_updates'] % num_batches_per_epoch

        self.epochs = epochs + 1
        self.num_batches_in_last_epoch = num_batches_in_last_epoch
        self.num_batches_per_epoch = num_batches_per_epoch

    class EpochsIter:
        def __init__(self, accountant: 'MiniBatchTrainingIterationsAccountant'):
            self._accountant = accountant

        def __next__(self):
            self._accountant.cur_epoch += 1
            self._accountant.num_samples_observed_in_epoch = 0
            self._accountant.cur_batch = 0
            if self._accountant.cur_epoch > self._accountant.epochs:
                self._accountant.cur_epoch = 0
                raise StopIteration
            return self._accountant.cur_epoch

        def __iter__(self):
            return self

    class BatchIter:
        def __init__(self, accountant: 'MiniBatchTrainingIterationsAccountant'):
            self._accountant = accountant

        def __next__(self):
            self._accountant.cur_batch += 1
            if self._accountant.cur_batch > self._accountant.num_batches_in_this_epoch():
                self._accountant.cur_batch = 0
                raise StopIteration
            return self._accountant.cur_batch

        def __iter__(self):
            return self

    def iterate_epochs(self):
        return MiniBatchTrainingIterationsAccountant.EpochsIter(self)

    def iterate_batches(self):
        return MiniBatchTrainingIterationsAccountant.BatchIter(self)






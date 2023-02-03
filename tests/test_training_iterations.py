import unittest
from unittest.mock import MagicMock
import numpy as np
from fedbiomed.common.training_plans._training_iterations import MiniBatchTrainingIterationsAccountant  # noqa

reference_num_batches = np.random.randint(low=4, high=15, size=1)[0]
reference_batch_size = np.random.randint(low=1, high=5, size=1)[0]
reference_dataset_size = reference_num_batches*reference_batch_size

mock_training_plan = MagicMock()
mock_training_plan.training_data_loader.__len__.return_value = reference_num_batches
mock_training_plan.training_data_loader.dataset.__len__.return_value = reference_dataset_size


class TestMiniBatchTrainingIterationsAccountant(unittest.TestCase):
    def test_min_batch_training_iterations_accountant_01_epochs_num_updates(self):
        # Researcher asks for 1 epoch
        mock_training_plan.training_args.return_value = {
            'epochs': 1,
            'batch_maxnum': None,
            'num_updates': None,
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        self.assertEqual(iter_accountant.epochs, 2)
        self.assertEqual(iter_accountant.num_batches_in_last_epoch, 0)
        self.assertEqual(iter_accountant.num_batches_per_epoch, reference_num_batches)

        # Researcher asks for 1 epoch but batch_maxnum != 0
        mock_training_plan.training_args.return_value = {
            'epochs': 1,
            'batch_maxnum': 3,
            'num_updates': None,
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        self.assertEqual(iter_accountant.epochs, 2)
        self.assertEqual(iter_accountant.num_batches_in_last_epoch, 0)
        self.assertEqual(iter_accountant.num_batches_per_epoch, 3)

        # Researcher asks for 1 epoch but batch_maxnum == 0
        mock_training_plan.training_args.return_value = {
            'epochs': 1,
            'batch_maxnum': 0,
            'num_updates': None,
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        self.assertEqual(iter_accountant.epochs, 2)
        self.assertEqual(iter_accountant.num_batches_in_last_epoch, 0)
        self.assertEqual(iter_accountant.num_batches_per_epoch, reference_num_batches)

        # Researcher asks for len_data_loader updates
        mock_training_plan.training_args.return_value = {
            'epochs': None,
            'batch_maxnum': None,
            'num_updates': reference_num_batches,
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        self.assertEqual(iter_accountant.epochs, 2)
        self.assertEqual(iter_accountant.num_batches_in_last_epoch, 0)
        self.assertEqual(iter_accountant.num_batches_per_epoch, reference_num_batches)

        # Researcher asks for updates < len_data_loader
        mock_training_plan.training_args.return_value = {
            'epochs': None,
            'batch_maxnum': None,
            'num_updates': reference_num_batches - 3,
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        self.assertEqual(iter_accountant.epochs, 1)
        self.assertEqual(iter_accountant.num_batches_in_last_epoch, reference_num_batches - 3)
        self.assertEqual(iter_accountant.num_batches_per_epoch, reference_num_batches)

        # Researcher asks for updates > len_data_loader
        mock_training_plan.training_args.return_value = {
            'epochs': None,
            'batch_maxnum': None,
            'num_updates': 2*reference_num_batches + 3,
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        self.assertEqual(iter_accountant.epochs, 3)
        self.assertEqual(iter_accountant.num_batches_in_last_epoch, 3)
        self.assertEqual(iter_accountant.num_batches_per_epoch, reference_num_batches)

        # Precedence rules
        mock_training_plan.training_args.return_value = {
            'epochs': 42,
            'batch_maxnum': 4242,
            'num_updates': 2*reference_num_batches + 3,
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        self.assertEqual(iter_accountant.epochs, 3)
        self.assertEqual(iter_accountant.num_batches_in_last_epoch, 3)
        self.assertEqual(iter_accountant.num_batches_per_epoch, reference_num_batches)

    def test_mini_batch_training_iterations_accountaint_02_iterations(self):
        mock_training_plan.training_args.return_value = {
            'epochs': None,
            'batch_maxnum': None,
            'num_updates': 2*reference_num_batches + 3,
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        epoch_counter = 0
        tot_steps_counter = 0
        for epoch in iter_accountant.iterate_epochs():
            epoch_counter += 1
            batch_counter = 0
            self.assertEqual(epoch_counter, iter_accountant.cur_epoch)
            for batch in iter_accountant.iterate_batches():
                batch_counter += 1
                tot_steps_counter += 1
                self.assertEqual(batch_counter, iter_accountant.cur_batch)
        self.assertEqual(epoch_counter, 3)
        self.assertEqual(batch_counter, 3)
        self.assertEqual(tot_steps_counter, 2*reference_num_batches + 3)

        mock_training_plan.training_args.return_value = {
            'epochs': 2,
            'batch_maxnum': None,
            'num_updates': None,
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        epoch_counter = 0
        tot_steps_counter = 0
        for epoch in iter_accountant.iterate_epochs():
            epoch_counter += 1
            batch_counter = 0
            self.assertEqual(epoch_counter, iter_accountant.cur_epoch)
            for batch in iter_accountant.iterate_batches():
                batch_counter += 1
                tot_steps_counter += 1
                self.assertEqual(batch_counter, iter_accountant.cur_batch)
        self.assertEqual(epoch_counter, 3)
        self.assertEqual(batch_counter, 0)
        self.assertEqual(tot_steps_counter, 2*reference_num_batches)

        mock_training_plan.training_args.return_value = {
            'epochs': 4,
            'batch_maxnum': 3,
            'num_updates': None,
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        epoch_counter = 0
        tot_steps_counter = 0
        for epoch in iter_accountant.iterate_epochs():
            epoch_counter += 1
            batch_counter = 0
            self.assertEqual(epoch_counter, iter_accountant.cur_epoch)
            for batch in iter_accountant.iterate_batches():
                batch_counter += 1
                tot_steps_counter += 1
                self.assertEqual(batch_counter, iter_accountant.cur_batch)
        self.assertEqual(epoch_counter, 5)
        self.assertEqual(batch_counter, 0)
        self.assertEqual(tot_steps_counter, 3*4)

    def test_mini_batch_training_iterations_accountaint_03_reporting(self):
        mock_training_plan.training_args.return_value = {
            'epochs': None,
            'batch_maxnum': None,
            'num_updates': 2*reference_num_batches + 3,
            'log_interval': 1,
            'batch_size': reference_batch_size
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        tot_steps_counter = 0
        for epoch in iter_accountant.iterate_epochs():
            for batch in iter_accountant.iterate_batches():
                tot_steps_counter += 1
                iter_accountant.increment_sample_counters(reference_batch_size)
                self.assertTrue(iter_accountant.should_log_this_batch())
                num_samples, num_samples_max = iter_accountant.reporting_on_num_samples()
                self.assertEqual(num_samples, tot_steps_counter*reference_batch_size)
                self.assertEqual(num_samples_max, 2*reference_dataset_size + 3*reference_batch_size,
                                 f'batches per epoch: {reference_num_batches}, batch size {reference_batch_size}')
                num_iter, num_iter_max = iter_accountant.reporting_on_num_iter()
                self.assertEqual(num_iter, tot_steps_counter)
                self.assertEqual(num_iter_max, 2*reference_num_batches + 3,
                                 f'batches per epoch: {reference_num_batches}, batch size {reference_batch_size}')
                epoch_to_report = iter_accountant.reporting_on_epoch()
                self.assertIsNone(epoch_to_report)

        # testing the value of attribute (important since we need it for aggregator weights computation)
        self.assertEqual(iter_accountant.num_samples_observed_in_total, tot_steps_counter * reference_batch_size)

        mock_training_plan.training_args.return_value = {
            'epochs': 2,
            'batch_maxnum': None,
            'num_updates': None,
            'log_interval': 1,
            'batch_size': reference_batch_size
        }
        iter_accountant = MiniBatchTrainingIterationsAccountant(mock_training_plan)
        tot_steps_counter = 0
        for epoch in iter_accountant.iterate_epochs():
            for batch in iter_accountant.iterate_batches():
                tot_steps_counter += 1
                iter_accountant.increment_sample_counters(reference_batch_size)
                self.assertTrue(iter_accountant.should_log_this_batch())
                num_samples, num_samples_max = iter_accountant.reporting_on_num_samples()
                self.assertEqual(num_samples, batch*reference_batch_size)
                self.assertEqual(num_samples_max, reference_dataset_size)
                num_iter, num_iter_max = iter_accountant.reporting_on_num_iter()
                self.assertEqual(num_iter, batch)
                self.assertEqual(num_iter_max, reference_num_batches)
                epoch_to_report = iter_accountant.reporting_on_epoch()
                self.assertEqual(epoch, epoch_to_report)

        # testing the value of attribute (important since we need it for aggregator weights computation)
        self.assertEqual(iter_accountant.num_samples_observed_in_total, tot_steps_counter * reference_batch_size)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

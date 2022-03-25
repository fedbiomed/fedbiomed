import os
import shutil
import unittest
from unittest.mock import patch, MagicMock

import testsupport.mock_researcher_environ  # noqa (remove flake8 false warning)

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.monitor import Monitor


def create_file(file_name: str):
    """ Creates a simple file for testing.

    Args:
        file_name (str): [path name of the file
    """
    # with tempfile.TemporaryFile() as temp_file:
    with open(file_name, "w") as f:
        f.write("this is a test. This file should be removed")


class TestMonitor(unittest.TestCase):
    """
    Test `Monitor` class
    Args:
        unittest ([type]): [description]
    """

    # before the tests
    def setUp(self):

        self.patch_summary_writer = patch('torch.utils.tensorboard.SummaryWriter')
        self.patch_add_scalar = patch('fedbiomed.researcher.monitor.SummaryWriter.add_scalar')

        self.mock_summary_writer = self.patch_summary_writer.start()
        self.mock_add_scalar = self.patch_add_scalar.start()

        self.mock_summary_writer.return_value = MagicMock()
        self.mock_add_scalar.return_value = MagicMock()

        self.monitor = Monitor()

    # after the tests
    def tearDown(self):

        # Clean all files in tmp/runs tensorboard results directory
        for files in os.listdir(environ['TENSORBOARD_RESULTS_DIR']):
            path = os.path.join(environ['TENSORBOARD_RESULTS_DIR'], files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)

        self.patch_summary_writer.stop()
        self.patch_add_scalar.stop()

        pass

    @patch('fedbiomed.researcher.monitor.Monitor._remove_logs')
    def test_monitor_01_initialization(self,
                                       patch_remove_logs):

        tensorboard_folder = self.monitor._log_dir
        self.assertTrue(os.path.isdir(tensorboard_folder))

        # create file inside
        test_file = os.path.join(tensorboard_folder, "test_file")
        create_file(test_file)

        # 2nd call to monitor
        _ = Monitor()
        patch_remove_logs.assert_called_once()

    def test_monitor_02_remove_logs_as_file(self):

        """ Test removing log files from directory
        using _remove_logs method
        """

        test_file = os.path.join(environ['TENSORBOARD_RESULTS_DIR'], "test_file")
        create_file(test_file)
        self.monitor._remove_logs()
        self.assertFalse(os.path.isfile(test_file), "Tensorboard log file has not been removed properly")

    def test_monitor_03_remove_logs_if_directory(self):

        test_dir = os.path.join(environ['TENSORBOARD_RESULTS_DIR'], 'test')
        os.mkdir(test_dir)
        test_file = os.path.join(environ['TENSORBOARD_RESULTS_DIR'], 'test', "test_file")
        create_file(test_file)

        self.monitor._remove_logs()
        self.assertFalse(os.path.isfile(test_file), "Tensorboard log file has not been removed properly")
        self.assertFalse(os.path.isdir(test_dir), "Tensorboard log folder has not been removed properly")

    def test_monitor_05_summary_writer(self):
        """ Test writing loss value at the first step 0 iteration """

        node_id = "1234"
        self.monitor._summary_writer('Train', node_id, metric={"Loss": 12}, cum_iter=1)
        self.assertTrue('1234' in self.monitor._event_writers)
        self.mock_add_scalar.assert_called_once()

        self.mock_add_scalar.reset_mock()
        self.monitor._summary_writer('Train', node_id, metric={"Loss": 12, "Loss2": 14}, cum_iter=1)
        self.assertEqual(self.mock_add_scalar.call_count, 2)

    @patch('fedbiomed.researcher.monitor.Monitor._summary_writer')
    def test_monitor_06_on_message_handler(self, mock_summary_writer):

        """Test on_message_handler of Monitor class """

        mock_summary_writer.reset_mock()
        self.monitor.set_tensorboard(True)
        self.monitor.on_message_handler({
            'researcher_id': '123123',
            'node_id': 'asd123',
            'job_id': '1233',
            'train': False,
            'test': True,
            'test_on_local_updates': True,
            'test_on_global_updates': True,
            'metric': {'metric_1': 12, 'metric_2': 13},
            'batch_samples': 13,
            'num_batches': 1,
            'total_samples': 1000,
            'iteration': 1,
            'epoch': 1,
            'command': 'add_scalar'
        })
        mock_summary_writer.assert_called_once_with(header='TESTING ON GLOBAL PARAMETERS',
                                                    node='asd123',
                                                    metric={'metric_1': 12, 'metric_2': 13},
                                                    cum_iter=1)

        mock_summary_writer.reset_mock()
        self.monitor.set_tensorboard(False)
        self.monitor.on_message_handler({
            'researcher_id': '123123',
            'node_id': 'asd123',
            'job_id': '1233',
            'train': False,
            'test': True,
            'test_on_local_updates': True,
            'test_on_global_updates': True,
            'metric': {'metric_1': 12, 'metric_2': 13},
            'batch_samples': 13,
            'num_batches': 1,
            'total_samples': 1000,
            'iteration': 2,
            'epoch': 1,
            'command': 'add_scalar'
        })
        mock_summary_writer.assert_not_called()

        mock_summary_writer.reset_mock()
        self.monitor.set_tensorboard(True)
        self.monitor.set_tensorboard("not_a_bool")  # as a side effect, this will set tensorboard flag to false
        self.monitor.on_message_handler({
            'researcher_id': '123123',
            'node_id': 'asd123',
            'job_id': '1233',
            'train': False,
            'test': True,
            'test_on_local_updates': True,
            'test_on_global_updates': True,
            'metric': {'metric_1': 12, 'metric_2': 13},
            'batch_samples': 13,
            'num_batches': 1,
            'total_samples': 1000,
            'iteration': 2,
            'epoch': 1,
            'command': 'add_scalar'
        })
        mock_summary_writer.assert_not_called()

    @patch('fedbiomed.researcher.monitor.SummaryWriter.close')
    def test_monitor_07_close_writers(self, mock_close):
        """  Testing closing writers """

        self.monitor._summary_writer(header='TESTING ON GLOBAL PARAMETERS',
                                     node='asd123',
                                     metric={'metric_1': 12, 'metric_2': 13},
                                     cum_iter=1)
        self.monitor.close_writer()
        mock_close.assert_called_once()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

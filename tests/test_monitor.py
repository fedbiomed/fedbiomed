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

    def test_monitor_04_summary_writer_global_step_minus(self):
        """Test writing loss value as global step is epoch not iteration """

        node_id = "1234"
        self.monitor._summary_writer(node_id, "loss", global_step=-1, scalar=2, epoch=3)

        self.mock_add_scalar.assert_called_once()
        self.assertEqual(self.monitor._event_writers[node_id]['step'], 3)
        self.assertEqual(self.monitor._event_writers[node_id]['stepper'], 0)
        self.assertEqual(self.monitor._event_writers[node_id]['step_state'], 0)

    def test_monitor_05_summary_writer_global_step_zero(self):
        """ Test writing loss value at the first step 0 iteration """

        node_id = "1234"
        self.monitor._summary_writer(node_id, "loss", global_step=0, scalar=2, epoch=3)
        self.mock_add_scalar.assert_called_once()
        self.assertEqual(self.monitor._event_writers[node_id]['step'], 1)
        self.assertEqual(self.monitor._event_writers[node_id]['stepper'], 1)
        self.assertEqual(self.monitor._event_writers[node_id]['step_state'], 1)

    def test_monitor_05_summary_writer_multiple_entry(self):
        """ Test writing multiple loss values from single node with summary writer"""

        node_id = "1234"
        self.monitor._summary_writer(node_id, "loss", global_step=0, scalar=2, epoch=3)
        self.monitor._summary_writer(node_id, "loss", global_step=10, scalar=2, epoch=3)
        self.monitor._summary_writer(node_id, "loss", global_step=20, scalar=2, epoch=3)

        self.assertEqual(self.mock_add_scalar.call_count, 3)
        self.assertEqual(self.monitor._event_writers[node_id]['step'], 21)
        self.assertEqual(self.monitor._event_writers[node_id]['stepper'], 0)
        self.assertEqual(self.monitor._event_writers[node_id]['step_state'], 1)

    def test_monitor_05_summary_writer_multiple_entry_second_round(self):
        """ Test writing multiple rounds of loss values  with summary writer"""

        node_id = "1234"
        self.monitor._summary_writer(node_id, "loss", global_step=0, scalar=2, epoch=3)
        self.monitor._summary_writer(node_id, "loss", global_step=10, scalar=2, epoch=3)
        self.monitor._summary_writer(node_id, "loss", global_step=20, scalar=2, epoch=3)
        self.monitor._summary_writer(node_id, "loss", global_step=0, scalar=2, epoch=3)

        self.assertEqual(self.mock_add_scalar.call_count, 4)
        self.assertEqual(self.monitor._event_writers[node_id]['step'], 22)
        self.assertEqual(self.monitor._event_writers[node_id]['stepper'], 1)
        self.assertEqual(self.monitor._event_writers[node_id]['step_state'], 22)

    @patch('fedbiomed.researcher.monitor.Monitor._summary_writer')
    def test_monitor_06_on_message_handler(self, mock_summary_writer):

        """Test on_message_handler of Monitor class """

        self.monitor.set_tensorboard(True)
        self.monitor.on_message_handler({
            'researcher_id': '123123',
            'node_id': 'asd123',
            'job_id': '1233',
            'iteration': 2,
            'key': 'loss',
            'value': 1.23,
            'epoch': 1,
            'command': 'add_scalar'
        })
        mock_summary_writer.assert_called_once_with('asd123', 'loss', 2, 1.23, 1)

    @patch('fedbiomed.researcher.monitor.SummaryWriter.close')
    def test_monitor_07_close_writers(self, mock_close):
        """  Testing closing writers """

        self.monitor._summary_writer('1234', "loss", global_step=-1, scalar=2, epoch=3)
        self.monitor.close_writer()
        mock_close.assert_called_once()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

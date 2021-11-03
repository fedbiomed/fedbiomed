import os
from fedbiomed.researcher.monitor import Monitor
import testsupport.mock_researcher_environ
from fedbiomed.researcher.environ import TENSORBOARD_RESULTS_DIR
import unittest

from unittest.mock import patch, MagicMock


def create_file(file_name: str):
    """ Creates a simple file for testing.

    Args:
        file_name (str): [path name of the file
    """
    #with tempfile.TemporaryFile() as temp_file:
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
        pass

    # after the tests
    def tearDown(self):
        pass

    def test_monitor_initialization_1(self):
        """Test first call and 2nd call to Monitor class.
        Checks the behaviour of Monitor (should behave as a singleton)
        and should remove existing files in `TENSORBOARD_RESULTS_DIR`
        when Monitor is reinitialized.
        """
        try:
            del self.monitor
        except Exception:
            pass

        # first call to monitor
        self.monitor = Monitor()
        tensorboard_folder = self.monitor._log_dir
        self.assertTrue(os.path.isdir(tensorboard_folder))

        # create file inside
        test_file = os.path.join(tensorboard_folder, "test_file")
        create_file(test_file)
        # 2nd call to monitor
        monitor = Monitor()
        tensorboard_folder = monitor._log_dir

        # check if created file has been deleted
        self.assertFalse(os.path.isfile(test_file))

        del self.monitor


    def test_remove_logs(self):
        
        """ Test removing log files from directory 
        using _remove_logs method
        """

        monitor = Monitor()
        log_dir = monitor._log_dir = TENSORBOARD_RESULTS_DIR
        test_file = os.path.join(log_dir, "test_file")
        create_file(test_file)
        monitor._remove_logs()
        self.assertFalse(os.path.isfile(test_file))

    @patch('fedbiomed.researcher.monitor.SummaryWriter')
    @patch('fedbiomed.researcher.monitor.SummaryWriter.add_scalar')
    def test_summary_writer(self,
                            mocking_summary_writer,
                            mocking_summary_writer_add_scalar,
                            ):
        """Tests if:
        1. `SummaryWriter.add_scalar` has been called when calling
        `_summary_writer`
        2. arguments passed are stored in `_event_writers`

        """

        monitor = Monitor()
        mocking_summary_writer.return_value = MagicMock()
        mocking_summary_writer_add_scalar.return_value = MagicMock()

        node_id = "1234"
        monitor._summary_writer(node_id,
                                "loss",
                                global_step=-1,
                                scalar=2,
                                epoch=3)


        mocking_summary_writer_add_scalar.assert_called_once()
        self.assertEqual(monitor._event_writers[node_id]['step'], 3)
        self.assertEqual(monitor._event_writers[node_id]['stepper'], 3)
        self.assertEqual(monitor._event_writers[node_id]['step_state'], 0)
        
        del monitor

    @patch('fedbiomed.researcher.monitor.SummaryWriter')
    @patch('fedbiomed.researcher.monitor.SummaryWriter.add_scalar')
    def test_on_message_handler(self,
                                mocking_summary_writer,
                                mocking_summary_writer_add_scalar):

        """Test on_message_handler of Monitor class """  

        monitor = Monitor()
        mocking_summary_writer.return_value = MagicMock()
        mocking_summary_writer_add_scalar.return_value = MagicMock()

        try: 
            monitor.on_message_handler({
                                    'researcher_id' : '123123',
                                    'node_id' : 'asd123',
                                    'job_id' : '1233',
                                    'iteration': 2,
                                    'key' : 'loss',
                                    'value' : 1.23,
                                    'epoch' : 1,
                                    'command' : 'add_scalar'
                            })
            is_success = True
        except:
            is_success = False

        self.assertEqual(is_success, True)    

    @patch('fedbiomed.researcher.monitor.SummaryWriter')
    @patch('fedbiomed.researcher.monitor.SummaryWriter.add_scalar')
    def test_close_writers(self,
                     mocking_summary_writer,
                     mocking_summary_writer_add_scalar):

        monitor = Monitor()
        mocking_summary_writer.return_value = MagicMock()
        mocking_summary_writer_add_scalar.return_value = MagicMock()

        for node_id in ('123' , '321'):
            monitor._summary_writer(node_id,
                                    "loss",
                                    global_step=-1,
                                    scalar=2,
                                    epoch=3)
        
        try: 
            monitor.close_writer()
            is_success = True
        except:
            is_success = False

        self.assertEqual(is_success, True, 'Summary writers are not closed properly') 


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

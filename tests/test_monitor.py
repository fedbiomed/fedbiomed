import tempfile
import os
from fedbiomed.researcher.monitor import Monitor
from fedbiomed.common.messaging import Messaging
import unittest

from unittest.mock import patch


def create_file(file_name:str):
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
    
    @patch('fedbiomed.common.messaging.Messaging.__init__')
    @patch('fedbiomed.common.messaging.Messaging.start')
    def test_monitor_initialization_1(self, mocking_messaging_init, mocking_messaging_start):
        """Test first call and 2nd call to Monitor class
        """
        try:
            del self.monitor
        except Exception:
            pass
        mocking_messaging_init.return_value = None
        mocking_messaging_init._messaging.return_value = None
        mocking_messaging_start.return_value = None
        
        # first call to monitor
        self.monitor = Monitor(tensorboard=True)
        tensorboard_folder = self.monitor._log_dir
        print( tensorboard_folder, os.path.isdir(tensorboard_folder))
        self.assertTrue(os.path.isdir(tensorboard_folder))
    
        # # create file inside
        # test_file = os.path.join(tensorboard_folder, "test_file")
        # create_file(test_file)
        # # 2nd call to monitor
        # monitor = Monitor(tensorboard=True)
        # tensorboard_folder = monitor._log_dir
        # print(tensorboard_folder)
        # self.assertFalse(os.path.isfile(test_file))
    
        del self.monitor
        
    @patch('fedbiomed.common.messaging')
    def test_monitor_initialization_2(self, mocking_messaging_init):
        mocking_messaging_init.return_value = None
        mocking_messaging_init._messaging.return_value = None
        monitor = Monitor(tensorboard=False)
        tensorboard_file = monitor._log_dir
        print(tensorboard_file)
        self.assertFalse(os.path.isdir(tensorboard_file))
        del monitor
        
if __name__ == '__main__':  # pragma: no cover
    unittest.main()

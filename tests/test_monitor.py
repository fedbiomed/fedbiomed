import os
from fedbiomed.researcher.monitor import Monitor
from fedbiomed.common.messaging import Messaging
from fedbiomed.researcher.environ import TENSORBOARD_RESULTS_DIR
import unittest

from unittest.mock import patch, MagicMock


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
    def test_monitor_initialization_1(self,
                                      mocking_messaging_init,
                                      mocking_messaging_start):
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
        print( tensorboard_folder, os.path.isdir(tensorboard_folder), TENSORBOARD_RESULTS_DIR)
        self.assertTrue(os.path.isdir(tensorboard_folder))
    
        # create file inside
        test_file = os.path.join(tensorboard_folder, "test_file")
        create_file(test_file)
        # 2nd call to monitor
        monitor = Monitor(tensorboard=True)
        tensorboard_folder = monitor._log_dir
        
        # check if created file has been deleted
        self.assertFalse(os.path.isfile(test_file))
    
        del self.monitor
        
    @patch('fedbiomed.common.messaging')
    def test_monitor_initialization_2(self, mocking_messaging_init):
        """Tests if folder is created even if user 
        is setting `tesorboard` to False
        """
        mocking_messaging_init.return_value = None
        mocking_messaging_init._messaging.return_value = None
        monitor = Monitor(tensorboard=False)
        tensorboard_file = monitor._log_dir
        
        self.assertTrue(os.path.isdir(tensorboard_file))
        del monitor
        
    # @patch('torch.utils.tensorboard.SummaryWriter')
    # @patch('torch.utils.tensorboard.SummaryWriter.add_scalar')
    # def test_summary_writer(self,
    #                         mocking_summary_writer_init,
    #                         mocking_summary_writer_add_scalar):
        
    #     mocking_summary_writer_init.return_value = "abcd"
    #     #mocking_summary_writer_init.return_value
    #     mocking_summary_writer_add_scalar.return_value = "abcd"
    #     monitor = Monitor(tensorboard=True)
    #     client_id = "1234"
    #     monitor._summary_writer(client_id,
    #                             "loss",
    #                             global_step=-1,
    #                             scalar=2,
    #                             epoch=3)
    #     print(monitor._event_writers)
    #     self.assertEquals(monitor._event_writers[client_id]['writer'], None)
    #     del monitor
        
if __name__ == '__main__':  # pragma: no cover
    unittest.main()

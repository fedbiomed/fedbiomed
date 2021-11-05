# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Detele environ. It is necessary to rebuild environ for required component
delete_environ()
import testsupport.mock_common_environ
# Import environ for node since test will be runing for node component
from fedbiomed.node.environ import environ

from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.common.messaging import Messaging
import unittest

from unittest.mock import patch


class TestHistoryMonitor(unittest.TestCase):
    """
    Test `HistoryMonitor` class
    Args:
        unittest ([type]): [description]
    """


    # Setup HistoryMonitor with Mocking messaging
    @patch('fedbiomed.common.messaging.Messaging.__init__')
    @patch('fedbiomed.common.messaging.Messaging.start')
    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def setUp(self, mocking_messaging_send_message,
                    mocking_messaging_start,
                    mocking_messaging_init):

        mocking_messaging_init.return_value = None
        mocking_messaging_start.return_value = None
        mocking_messaging_send_message.return_value = None

        # Messaging to pass HistoryMonitor
        self._messaging = Messaging()

        try:
            self.history_monitor = HistoryMonitor( job_id='1234',
                                                   researcher_id='reasearcher-id',
                                                   client=self._messaging
                                                 )
            self._history_monitor_ok = True
        except:
            self._history_monitor_ok = False


        self.assertTrue(self._history_monitor_ok, 'History monitor intialize correctly')



    # after the tests
    def tearDown(self):
        #self.req_pathcer1.stop()
        #self.req_pathcer1.stop()
        #self.req_pathcer1.stop()
        pass

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def test_send_message(self, mocking_messaging_send_message):
        """Test history monitor can add a scalar value using
        add_scalar method
        """
        scalar = self.history_monitor.add_scalar(
                       key='loss',
                       value=123.34,
                       iteration=1,
                       epoch=1,
        )
        self.assertEqual(scalar, None)

        pass

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def test_send_message_error(self,  mocking_messaging_send_message):

        """Test send message in case of sending wrong types"""

        with self.assertRaises(ValueError):
           scalar = self.history_monitor.add_scalar(
                           key=123,
                           value='asdasd',
                           iteration='111',
                           epoch='111',
           )


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

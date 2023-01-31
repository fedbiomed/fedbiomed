import unittest
from unittest.mock import patch

#############################################################
# Import NodeTestCase before importing FedBioMed Module
from testsupport.base_case import NodeTestCase
#############################################################


from fedbiomed.common.exceptions import FedbiomedMessageError
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.common.messaging import Messaging


class TestHistoryMonitor(NodeTestCase):
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
            self.history_monitor = HistoryMonitor(job_id='1234',
                                                  researcher_id='reasearcher-id',
                                                  client=self._messaging
                                                  )
            self._history_monitor_ok = True
        except:
            self._history_monitor_ok = False


        self.assertTrue(self._history_monitor_ok, 'History monitor intialize correctly')



    def tearDown(self):
        '''
        after all tests
        '''
        pass

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def test_send_message(self, mocking_messaging_send_message):
        """Test history monitor can add a scalar value using
        add_scalar method
        """
        scalar = self.history_monitor.add_scalar(
            metric={'test': 123},
            train=True,
            test=False,
            test_on_global_updates=False,
            test_on_local_updates=False,
            total_samples=1234,
            batch_samples=12,
            num_batches=12,
            iteration=111,
            epoch=111
        )
        self.assertEqual(scalar, None)

        pass

    @patch('fedbiomed.common.messaging.Messaging.send_message')
    def test_send_message_error(self, mocking_messaging_send_message):

        """Test send message in case of sending wrong types"""

        with self.assertRaises(FedbiomedMessageError):
            _ = self.history_monitor.add_scalar(
                metric={'test': 123},
                train=True,
                test=False,
                test_on_global_updates=False,
                test_on_local_updates=False,
                total_samples=1234,
                batch_samples=12,
                num_batches=12,
                iteration='111',
                epoch='111',
            )


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

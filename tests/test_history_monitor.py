import unittest
from unittest.mock import patch, MagicMock

from fedbiomed.common.exceptions import FedbiomedMessageError
from fedbiomed.node.history_monitor import HistoryMonitor


class TestHistoryMonitor(unittest.TestCase):
    """
    Test `HistoryMonitor` class
    Args:
        unittest ([type]): [description]
    """

    def setUp(self):

        # Messaging to pass HistoryMonitor
        self.send = MagicMock()

        try:
            self.history_monitor = HistoryMonitor(
                node_id='test-node-id',
                experiment_id='1234',
                researcher_id='researcher-id',
                send=self.send
            )
            self._history_monitor_ok = True
        except:
            self._history_monitor_ok = False


        self.assertTrue(self._history_monitor_ok, 'History monitor has initialized correctly')



    def tearDown(self):
        '''
        after all tests
        '''
        pass


    def test_send_message(self):
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

    def test_send_message_error(self):

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

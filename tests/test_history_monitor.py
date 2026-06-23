import unittest
from unittest.mock import MagicMock

from fedbiomed.common.exceptions import FedbiomedMessageError
from fedbiomed.node.history_monitor import HistoryMonitor


class TestHistoryMonitor(unittest.TestCase):
    """Tests for HistoryMonitor."""

    def setUp(self):
        # Messaging to pass HistoryMonitor
        self.send = MagicMock()
        self.history_monitor = HistoryMonitor(
            node_id="test-node-id",
            node_name="test-node-name",
            experiment_id="1234",
            researcher_id="researcher-id",
            send=self.send,
        )

    def tearDown(self):
        """After all tests."""

    def test_send_message(self):
        """Test history monitor can add a scalar value using
        add_scalar method
        """
        scalar = self.history_monitor.add_scalar(
            metric={"test": 123},
            train=True,
            test=False,
            test_on_global_updates=False,
            test_on_local_updates=False,
            total_samples=1234,
            batch_samples=12,
            num_batches=12,
            iteration=111,
            epoch=111,
        )
        self.assertEqual(scalar, None)
        self.send.assert_called_once()

        feedback_message = self.send.call_args[0][0]
        self.assertEqual(feedback_message.researcher_id, "researcher-id")
        self.assertEqual(feedback_message.scalar.node_id, "test-node-id")
        self.assertEqual(feedback_message.scalar.node_name, "test-node-name")
        self.assertEqual(feedback_message.scalar.experiment_id, "1234")
        self.assertEqual(feedback_message.scalar.metric, {"test": 123})
        self.assertTrue(feedback_message.scalar.train)
        self.assertFalse(feedback_message.scalar.test)
        self.assertEqual(feedback_message.scalar.total_samples, 1234)
        self.assertEqual(feedback_message.scalar.batch_samples, 12)
        self.assertEqual(feedback_message.scalar.num_batches, 12)
        self.assertEqual(feedback_message.scalar.iteration, 111)
        self.assertEqual(feedback_message.scalar.epoch, 111)

    def test_send_message_error(self):
        """Test send message in case of sending wrong types"""

        with self.assertRaises(FedbiomedMessageError):
            _ = self.history_monitor.add_scalar(
                metric={"test": 123},
                train=True,
                test=False,
                test_on_global_updates=False,
                test_on_local_updates=False,
                total_samples=1234,
                batch_samples=12,
                num_batches=12,
                iteration="111",
                epoch="111",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

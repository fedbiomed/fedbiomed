import unittest
from unittest.mock import patch, MagicMock
import time
import threading

from fedbiomed.common.exceptions import FedbiomedSynchroError
from fedbiomed.common.synchro import EventWaitExchange, MAX_TRIGGERED_EVENT_TIMEOUT


class TestCommonSynchro(unittest.TestCase):
    """Test for common synchro module"""

    def setUp(self):
        self.mock_event = MagicMock(spec=threading.Event)
        self.patch_event = patch(
            "fedbiomed.common.synchro.threading.Event", self.mock_event
        )
        self.patcher_event = self.patch_event.start()

    def tearDown(self):
        self.patch_event.stop()

    def test_synchro_01_init_ok(self):
        """Instantiate EventWaitExchange successfully"""

        # prepare

        # action
        for remove_delivered in [True, False]:
            EventWaitExchange(remove_delivered)

        # test
        # nothing to test at this point

    def test_synchro_02_add_wait_events(self):
        """Add event(s) to the exchange, check they are correctly stored.
        Then wait, check they are correctly retrieved.
        """

        # prepare
        e1 = "event1"
        d1 = ["one", "two"]
        e2 = "event2"
        d2 = 12345
        d3 = 23456

        for remove_delivered in [True, False]:
            exchange = EventWaitExchange(remove_delivered)
            # need to access private member to avoid creating an (un-needed) getter
            events = exchange._triggered_events

            # actions + checks

            # empty
            self.assertEqual(events, {})

            # one event
            exchange.event(e1, d1)
            self.assertEqual(set(events.keys()), set([e1]))
            self.assertEqual(events[e1]["data"], d1)

            # two events
            exchange.event(e2, d2)
            self.assertEqual(set(events.keys()), set([e1, e2]))
            self.assertEqual(events[e1]["data"], d1)
            self.assertEqual(events[e2]["data"], d2)

            # two events, previous version overwritten
            exchange.event(e2, d3)
            self.assertEqual(set(events.keys()), set([e1, e2]))
            self.assertEqual(events[e1]["data"], d1)
            self.assertEqual(events[e2]["data"], d3)

            # one event waited, is it still there ?
            all_received, events_data = exchange.wait([e1], 1)
            self.assertTrue(all_received)
            self.assertEqual(events_data, [d1])
            self.assertEqual(e1 in events, not remove_delivered)
            self.assertTrue(e2 in events)

            # another event waited, is it still there ?
            all_received, events_data = exchange.wait([e2], 1)
            self.assertTrue(all_received)
            self.assertEqual(events_data, [d3])
            self.assertEqual(e2 in events, not remove_delivered)
            self.assertEqual(e1 in events, not remove_delivered)

    def test_synchro_03_wait_add_events(self):
        """Wait for events listeners, then add events, then check listeners are woken up"""

        # prepare
        e1 = "event1"
        d1 = ["one", "two"]
        e2 = "event2"
        d2 = 12345

        # terminate the wait call - need to modify the
        self.patch_event.stop()
        self.mock_event.return_value.wait.side_effect = FedbiomedSynchroError
        self.patcher_event = self.patch_event.start()

        for remove_delivered in [True, False]:
            exchange = EventWaitExchange(remove_delivered)

            # actions: add listeners in exchange
            # check; no event set
            with self.assertRaises(FedbiomedSynchroError):
                exchange.wait([e1], 1)
            with self.assertRaises(FedbiomedSynchroError):
                exchange.wait([e2], 1)
            with self.assertRaises(FedbiomedSynchroError):
                exchange.wait([e1, e2], 1)

            self.patcher_event.return_value.set.assert_not_called()

            # actions: add first event
            # check: one listener set for [e1]
            exchange.event(e1, d1)
            self.patcher_event.return_value.set.assert_called_once()
            self.patcher_event.return_value.set.reset_mock()

            # actions: add second event
            # check: three listeners set for [e1] [e2] [e1, e2]
            exchange.event(e2, d2)
            self.assertEqual(self.patcher_event.return_value.set.call_count, 3)
            self.patcher_event.return_value.set.reset_mock()

    @patch("fedbiomed.common.synchro.EventWaitExchange._all_events", return_value=True)
    def test_synchro_04_wait_arguments_ok_bad(self, event_wait_exchange):
        """Wait called with good and bad arguments"""
        # prepare
        timeouts = [-0.0, 3, 12, 0.4, 7.65, 60, MAX_TRIGGERED_EVENT_TIMEOUT]

        for remove_delivered in [True, False]:
            for timeout in timeouts:
                exchange = EventWaitExchange(remove_delivered)

                # test
                exchange.wait(["dummy_id"], timeout)

                # no specific check

        # prepare
        timeouts = [
            "another",
            [1],
            ["bad]"],
            -1,
            -2.3,
            MAX_TRIGGERED_EVENT_TIMEOUT + 0.1,
            MAX_TRIGGERED_EVENT_TIMEOUT + 1,
        ]

        for remove_delivered in [True, False]:
            for timeout in timeouts:
                exchange = EventWaitExchange(remove_delivered)

                # test + check
                with self.assertRaises(FedbiomedSynchroError):
                    exchange.wait(["dummy_id"], timeout)

    @patch("fedbiomed.common.synchro.GRACE_TRIGGERED_EVENT", 0)
    def test_synchro_05_wait_timeout(self):
        """Call to `wait()` is timeouted"""
        # prepare
        max_timeout = 0.1

        with patch("fedbiomed.common.synchro.MAX_TRIGGERED_EVENT_TIMEOUT", max_timeout):
            for remove_delivered in [True, False]:
                exchange = EventWaitExchange(remove_delivered)

                # test
                all_received, events_data = exchange.wait(["dummy_id"], max_timeout / 2)

                # check
                self.assertFalse(all_received)
                self.assertEqual(events_data, [])

    @patch("fedbiomed.common.synchro.GRACE_TRIGGERED_EVENT", 0)
    def test_synchro_06_event_timeout(self):
        """Stored event is timeouted"""
        # prepare
        max_timeout = 0.1

        e1 = "event1"
        d1 = ["one", "two"]
        e2 = "event2"
        d2 = 12345

        with patch("fedbiomed.common.synchro.MAX_TRIGGERED_EVENT_TIMEOUT", max_timeout):
            for remove_delivered in [True, False]:
                exchange = EventWaitExchange(remove_delivered)
                # need to access private member to avoid creating an (un-needed) getter
                events = exchange._triggered_events

                # test + check

                # one event
                exchange.event(e1, d1)
                self.assertEqual(set(events.keys()), set([e1]))
                self.assertEqual(events[e1]["data"], d1)

                # trigger timeout
                time.sleep(max_timeout)

                # two events, one timeouted
                exchange.event(e2, d2)
                self.assertEqual(set(events.keys()), set([e2]))


if __name__ == "__main__":
    unittest.main()

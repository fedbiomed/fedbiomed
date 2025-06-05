# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from typing import Any, List, Optional, Tuple

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedSynchroError
from fedbiomed.common.logger import logger

# Maximum delay a triggered event is kept
MAX_TRIGGERED_EVENT_TIMEOUT = 60

# Grace delay before removing a triggered event after it reaches the MAX_TRIGGERED_EVENT_TIMEOUT
GRACE_TRIGGERED_EVENT = 10


class EventWaitExchange:
    """Provides thread safe synchronized data exchange object.

    Object users can wait on one or more events registered with an `event_id` that must be
    chosen to ensure it is unique in the object.

    Object users can trigger events using an `event_id` that may already be waited (or not).
    Arbitrary data can be associated to the event and will be transmitted to the receiving
    waiter(s).

    One or more waiter can wait on the same event. If the event is removed when delivered,
    to a waiter then only one waiter receives it. If the event is not removed when delivered,
    then multiple waiters can receive it until it is cleaned on timeout.
    """

    def __init__(self, remove_delivered):
        """Constructor of the class.

        Args:
            remove_delivered: if True, remove from an event from the triggered event when they are delivered
        """
        self._remove_delivered = remove_delivered

        # table for storing the triggered event received, still waiting to be consumed
        self._triggered_events = {}
        # lock for accessing self._triggered_events
        self._triggered_events_lock = threading.Lock()

        # table for tracking pending requests of the listeners waiting for events
        # Key is an internal unique `id_counter`, value is a list of `event_id`
        self._pending_listeners = {}
        # lock for accessing self._pending_listeners
        self._pending_listeners_lock = threading.Lock()

        self._id_counter = 0

    def _all_events(self, listener_id: Optional[int] = None) -> List[int]:
        """Check if all events were received for some node to node requests.

        Args:
            listener_id: unique ID of a pending request to check. If `None`, all pending requests are checked.

        Returns:
            A list of `listener_id` whose request is complete. Empty list if no matching completed request is found.
        """

        all_events = []
        with self._pending_listeners_lock:
            with self._triggered_events_lock:

                # if no specified listener, test all listeners
                if listener_id is None:
                    listener_ids = list(self._pending_listeners.keys())
                else:
                    listener_ids = [listener_id]

                # check if all events for this listener ids are received and pending
                all_event_ids = list(self._triggered_events.keys())
                for lid in listener_ids:
                    if all(
                        reqid in all_event_ids
                        for reqid in self._pending_listeners[lid]["event_ids"]
                    ):
                        all_events += [lid]

        return all_events

    def _clean_triggered_events(self) -> None:
        """Clean out obsolete entries from the triggered event table."""

        time_current = time.time()
        # Cast to list() needed to avoid error because dict changes during iteration
        for reqid in list(self._triggered_events.keys()):
            if (
                self._triggered_events[reqid]["start_time"]
                + MAX_TRIGGERED_EVENT_TIMEOUT
                + GRACE_TRIGGERED_EVENT
                < time_current
            ):
                # this triggered event is obsolete
                del self._triggered_events[reqid]
                logger.debug(
                    f"{ErrorNumbers.FB324}: Clean obsolete entry {reqid} from triggered"
                    "event table."
                )

                # In case a pending listener waits on this event, it is blocked until its timeout
                # and then fails (not all data are delivered).

        # Note: don't clean entries from `self._pending_listeners` as this is always done at
        # the end of the `wait()`

    def event(self, event_id: str, event_data: Any) -> None:
        """Add an entry to the table of triggered event

        Args:
            event_id: unique ID of the event
            event_data: arbitrary data to transmit to the event receiver
        """

        with self._triggered_events_lock:
            # Remove obsolete triggered events. We could additionally call at other times.
            self._clean_triggered_events()

            # In case a event already exists for this ID, overwrite it with the newer one
            self._triggered_events[event_id] = {
                "start_time": time.time(),
                "data": event_data,
            }
        # check if added event completes some listeners
        completed_listeners = self._all_events()
        with self._pending_listeners_lock:
            for completed_listener in completed_listeners:
                # check: listener may have been removed since tested `_all_events`
                # as we didn't keep the lock
                if completed_listener in self._pending_listeners:
                    # wake up waiting listener
                    self._pending_listeners[completed_listener]["event"].set()

    def wait(self, event_ids: list[str], timeout: float) -> Tuple[bool, List[Any]]:
        """Wait for a registered listener to complete.

        Blocks until all events for this listener are triggered or until timeout is reached.

        Args:
            event_ids: list of the unique IDs for all events to wait
            timeout: maximum time to wait for replies in seconds

        Returns:
            A tuple consisting of
                - a bool set to `True` if all events for this listener were delivered to this
                    request, `False` if not
                - a list of the arbitrary data associated with each delivered event

        Raises:
               FedbiomedNodeToNodeError: timeout has incorrect type or value
        """

        # Check value for timeout, as bad value may cause hard to detect problems
        if (
            not isinstance(timeout, (float, int))
            or timeout < 0
            or timeout > MAX_TRIGGERED_EVENT_TIMEOUT
        ):
            raise FedbiomedSynchroError(
                f"{ErrorNumbers.FB324}: Cannot wait {timeout} seconds. "
                f"Should be int or float between 0 and {MAX_TRIGGERED_EVENT_TIMEOUT}"
            )

        time_initial = time.time()
        with self._pending_listeners_lock:
            self._id_counter += 1
            listener_id = self._id_counter
            event = threading.Event()
            self._pending_listeners[listener_id] = {
                "start_time": time_initial,
                "event": event,
                "event_ids": event_ids,
            }

        # wait until all events are triggered or timeout is reached
        while not self._all_events(listener_id) and (
            time.time() < time_initial + timeout
        ):
            # be sure not to hold any lock when waiting !
            event.wait(time_initial + timeout - time.time())
            event.clear()

        with self._pending_listeners_lock:
            with self._triggered_events_lock:
                # check if all events were received (and are still available for delivery)
                all_received = set(
                    self._pending_listeners[listener_id]["event_ids"]
                ).issubset(set(self._triggered_events.keys()))

                # create list of delivered data
                events_data = [
                    self._triggered_events[reqid]["data"]
                    for reqid in self._pending_listeners[listener_id]["event_ids"]
                    if reqid in self._triggered_events
                ]

                if self._remove_delivered:
                    # remove all events for this request from the triggered events available
                    # for delivery
                    for reqid in self._pending_listeners[listener_id]["event_ids"]:
                        if reqid in self._triggered_events:
                            self._triggered_events.pop(reqid)

                # remove expired listener
                del self._pending_listeners[listener_id]

        return all_received, events_data

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Optional
import threading
import time

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedNodeToNodeError
from fedbiomed.common.message import InnerMessage
from fedbiomed.common.logger import logger


# Maximum delay we can wait for pending request's reply
MAX_PENDING_REPLY_TIMEOUT = 60

# Grace delay for removing a pending reply after it reaches the MAX_PENDING_REPLY_TIMEOUT
GRACE_PENDING_REPLY = 10


class PendingRequests:
    """Handle the replies to request-reply node to node communications with a researcher."""

    def __init__(self):
        """Constructor of the class."""

        # table for storing the replies received to requests, still waiting to be consumed
        self._pending_replies = {}
        # lock for accessing self._pending_replies
        self._pending_replies_lock = threading.Lock()

        # table for tracking the node to node application level requests waiting
        # for some reply, and the `reply_id` of the corresponding node to node requests
        self._pending_listeners = {}
        # lock for accessing self._pending_listeners
        self._pending_listeners_lock = threading.Lock()

        self._id_counter = 0

    def _all_replies(self, listener_id: Optional[int] = None) -> List[int]:
        """Check if all replies were received for node to node requests.

        Args:
            listener_id: unique ID of a request-reply to check. If `None`, all pending requests are checked.

        Returns:
            A list of `listener_id` of the matching completed requests found, if any.
                Empty list if no matching completed request is found.
        """

        all_replies = []
        with self._pending_listeners_lock:
            with self._pending_replies_lock:

                # if no specified listener, test all listeners
                if listener_id is None:
                    listener_ids = list(self._pending_listeners.keys())
                else:
                    listener_ids = [listener_id]

                # check if all replies for this listener ids are received and pending
                all_rep_ids = list(self._pending_replies.keys())
                for lid in listener_ids:
                    if all([reqid in all_rep_ids for reqid in self._pending_listeners[lid]['requests']]):
                        all_replies += [lid]

        return all_replies

    def _clean_pending_replies(self) -> None:
        """Clean out obsolete entries from the pending replies table.
        """

        time_current = time.time()
        # Cast to list() needed to avoid error because dict changes during iteration
        for reqid in list(self._pending_replies.keys()):
            if self._pending_replies[reqid]['start_time'] + MAX_PENDING_REPLY_TIMEOUT + GRACE_PENDING_REPLY \
                    < time_current:
                # this pending reply is obsolete
                del self._pending_replies[reqid]
                logger.debug(f"Clean obsolete entry {reqid} from pending replies table.")

                # In case a pending listener waits on this reply, it is blocked until its timeout
                # and fails. Anyway this should not happen, unless the `wait()` started more
                # than GRACE_PENDING_REPLY after creating the listener.

        # Note: don't clean entries from `self._pending_listeners` as this is always done at the end of the `wait()`


    def add_reply(self, request_id: str, msg: InnerMessage) -> None:
        """Add an entry to the pending replies tables

        Args:
            request_id: unique ID of the request-reply
            msg: inner reply message received
        """

        with self._pending_replies_lock:
            # Remove obsolete pending replies. We could additionally call at other times.
            self._clean_pending_replies()

            # In case a reply already exists for this request, overwrite it with the newer one
            self._pending_replies[request_id] = {
                'start_time': time.time(),
                'message': msg,
            }

        # check if added reply completes some listeners
        # (note: there should never be more than 1 event `set()` under current assumptions)
        completed_listeners = self._all_replies()
        with self._pending_listeners_lock:
            for completed_listener in completed_listeners:
                # check: listener may have been removed since tested `_all_replies` as we don't kept the lock
                if completed_listener in self._pending_listeners:
                    # wake up waiting listener
                    self._pending_listeners[completed_listener]['event'].set()


    def add_listener(self, request_ids: list[str]) -> int:
        """Add a listener waiting for replies to multiple requests to nodes.

        Args:
            request_ids: list of the unique IDs for all requests to wait

        Returns:
            An `int`, unique listener ID for this listener
        """
        with self._pending_listeners_lock:
            self._id_counter += 1
            self._pending_listeners[self._id_counter] = {
                'start_time': time.time(),
                'event': threading.Event(),
                'requests': request_ids,
            }

        return self._id_counter


    def wait(self, listener_id: int, timeout: float) -> Tuple[bool, List[InnerMessage]]:
        """Wait for a registered listener to complete.

        Blocks until all replies for this listener are received or until timeout is reached.

        Args:
            listener_id: unique ID of this listener
            timeout: maximum time to wait for replies in seconds

        Returns:
            A tuple consisting of
                - a bool set to `True` if all replies for this listener were received, `False` if not
                - a list of the messages for the received replies

        Raises:
               FedbiomedNodeToNodeError: timeout has incorrect type or value
        """

        # Check value for timeout, as bad value may cause hard to detect problems
        if (not isinstance(timeout, float) and not (isinstance(timeout, int))) \
                or timeout < 0 or timeout > MAX_PENDING_REPLY_TIMEOUT:
            raise FedbiomedNodeToNodeError(f"{ErrorNumbers.FB324}: Cannot wait {timeout} seconds. "
                                           f"Should be float between 0 and {MAX_PENDING_REPLY_TIMEOUT}")

        time_initial = time.time()

        with self._pending_listeners_lock:
            # Check a listener exists for this id
            if listener_id not in self._pending_listeners:
                # don't raise exception, non fatal error
                logger.error(f"{ErrorNumbers.FB324}: Node tries to wait for replies "
                             f"of a non existing request-reply: {listener_id}")
                return False, []

            # Retrieve event object for this listener
            event = self._pending_listeners[listener_id]['event']

        # wait until all replies are received or timeout is reached
        while not self._all_replies(listener_id) and (time.time() < time_initial + timeout):
            # be sure not to hold any lock when waiting !
            event.wait(time_initial + timeout - time.time())
            event.clear()

        with self._pending_listeners_lock:
            with self._pending_replies_lock:
                # check if all replies were received
                all_received = set(self._pending_listeners[listener_id]['requests']).\
                    issubset(set(self._pending_replies.keys()))

                # remove all replies for this request from the pending replies,
                # and add them to the list of received replies
                messages = [self._pending_replies.pop(reqid)
                            for reqid in self._pending_listeners[listener_id]['requests']
                            if reqid in self._pending_replies]

                # remove expired listener
                del self._pending_listeners[listener_id]

        return all_received, messages

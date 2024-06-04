# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Optional
import threading
import time

from fedbiomed.common.message import InnerMessage
from fedbiomed.common.logger import logger


class PendingRequests:
    """xxx"""

    def __init__(self):
        """xxx"""
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


    def _all_replies(self, listener_id: int = None) -> Optional[int]:
        """xxx"""

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
                        return lid

                return None

    def _clean(self) -> None:
        """xxx"""

        # TODO : add timeout entries in self._pending_replies, and call this clean when needed
        # Don't remove entries from self._pending_listeners ? Or remove corresponding entries in pending replies ?
        pass


    def add_reply(self, request_id: str, msg: InnerMessage) -> None:
        """xxx"""

        with self._pending_replies_lock:
            # In case a reply already exists for this request, overwrite it with the newer one
            self._pending_replies[request_id] = {
                'start_time': time.time(),
                'message': msg,
            }
            logger.debug(f"********** ADDED PENDING REPLY {self._pending_replies}")

        # check if added reply completes a listener
        # (note: there should never be more than 1 event `set()` under current assumptions)
        completed_listener = self._all_replies()
        if completed_listener is not None:
            with self._pending_listeners_lock:
                # check: listener may have been removed since tested `_all_replies` as we don't keep the lock
                if completed_listener in self._pending_listeners:
                    # wake up waiting listener
                    self._pending_listeners[completed_listener]['event'].set()
                    logger.debug(
                        f"********** WAKED UP COMPLETED LISTENER {self._pending_listeners[completed_listener]}")


    def add_listener(self, request_ids: list[str]) -> int:
        """xxx"""
        with self._pending_listeners_lock:
            self._id_counter += 1
            self._pending_listeners[self._id_counter] = {
                'start_time': time.time(),
                'event': threading.Event(),
                'requests': request_ids,
            }
            logger.debug(f"********** ADDED LISTENER {self._pending_listeners}")

        return self._id_counter


    def wait(self, listener_id: int, timeout: float) -> Tuple[bool, List[InnerMessage]]:
        """xxx"""

        time_initial = time.time()

        with self._pending_listeners_lock:
            # Check a listener exists for this id
            if listener_id not in self._pending_listeners:
                # don't raise exception ?
                logger.error("TODO CORRECT ERROR CODE: node tries to wait for replies "
                             f"of a non existing request-reply: {listener_id}")
                return False, []

            # Retrieve event object for this listener
            event = self._pending_listeners[listener_id]['event']

        # wait until all replies are received or timeout is reached
        while (self._all_replies(listener_id) is None) and (time.time() < time_initial + timeout):
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

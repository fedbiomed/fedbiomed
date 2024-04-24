# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

""" Queue module that contains task queue class that is a wrapper to the persistqueue python library."""

import persistqueue
from typing import Optional, Any

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTaskQueueError
from fedbiomed.common.logger import logger


class TasksQueue:
    """A disk-persistent Queue object, ensuring queue will remain on disk even if program crashes.

    Relies on `persistqueue` package.

    """

    def __init__(self, messages_queue_dir: str, tmp_dir: str):
        """Construct disk-persistent Queue.

        Args:
            messages_queue_dir: directory where enqueued data should be persisted.
            tmp_dir: indicates where temporary files should be stored.
        """
        try:
            # small chunksize to limit un-needed use of disk space
            self.queue = persistqueue.Queue(messages_queue_dir, tempdir=tmp_dir, chunksize=1)
        except ValueError as e:
            msg = ErrorNumbers.FB603.value + ": cannot create queue (" + str(e) + ")"
            logger.critical(msg)
            raise FedbiomedTaskQueueError(msg)

    def add(self, task: dict):
        """Adds a task to the queue

        Args:
            task: a dict describing the task to be added
        """
        try:
            self.queue.put(task)
        except persistqueue.exceptions.Full:
            msg = ErrorNumbers.FB603.value + ": queue is full"
            logger.critical(msg)
            raise FedbiomedTaskQueueError(msg)
        # persistequeue does also raise ValueError if timeout is < 0
        # but we do not provide a timeout value

    def get(self, block: Optional[bool] = True) -> dict:
        """Get the current task in the queue

        Args:
            block: if True, block if necessary until an item is available. Defaults to True.

        Returns:
            Dictionary object stored in queue

        Raises:
            FedbiomedTaskQueueError: If queue is empty
        """
        try:
            return self.queue.get(block)
        except persistqueue.exceptions.Empty:
            msg = ErrorNumbers.FB603.value + ": queue is empty"
            logger.debug(msg)
            raise FedbiomedTaskQueueError(msg)

        # - this ignores may be ignored by the caller
        # - persist queue does also raise ValueError then timeout < 0,
        #   but we do not use timeout yet

    def qsize(self) -> int:
        """Retrieve the size of the queue

        Returns:
            size of the queue
        """
        return self.queue.qsize()

    def task_done(self) -> Any:
        """Indicate whether a formerly enqueued task is complete

        Returns:
            True if task is complete
        """
        try:
            return self.queue.task_done()
        except ValueError:
            # persistqueue raises it if task_done called too many times we can ignore it
            return

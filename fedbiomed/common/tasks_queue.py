"""
wrapper to the persistqueue python library

This will allow to change replace the persistqueue (if needed)
without changing the tasks_queue API
"""


import persistqueue
from typing import Optional

from fedbiomed.common.constants  import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTaskQueueError
from fedbiomed.common.logger     import logger


class TasksQueue:
    """
    A disk-persistant Queue object, ensuring queue will remain on
    disk even if program crashes. Relies on `persistqueue` package.

    """
    def __init__(self, messages_queue_dir: str, tmp_dir: str):
        """Instantiates a disk-persistant Queue.

        Args:
            messages_queue_dir (str): directory where enqueued data should be
            persisted.
            tmp_dir (str): indicates where temporary files should be stored.
        """
        try:
            self.queue = persistqueue.Queue(messages_queue_dir, tempdir=tmp_dir)
        except ValueError as e:
            msg = ErrorNumbers.FB603.value + ": cannot create queue (" + str(e) + ")"
            logger.critical(msg)
            raise FedbiomedTaskQueueError(msg)

    def add(self, task: dict):
        """this method adds a task to the queue

        Args:
            task (dict): a dict describing the task to be added
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
        """this method gets the current task in the queue

        Args:
            block (bool, optional): if True, block if necessary until
            an item is available. Defaults to True.

        Returns:
            [dict]: dictionary object stored in queue
        """
        try:
            return self.queue.get(block)
        except persistqueue.exceptions.Empty:
            msg = ErrorNumbers.FB603.value + ": queue is empty"
            logger.debug(msg)
            raise FedbiomedTaskQueueError(msg)
        #
        # - this ignore may be ignored by the caller
        #
        # - persistequeue does also raise ValueError then timeout < 0
        #   but we do not use timeout yet
        #

    def qsize(self):
        """this method returns the size of the queue

        Returns:
            int: size of the queue
        """
        return self.queue.qsize()

    def task_done(self):
        """Indicates whether a formerly enqueued task is complete

        Returns:
            'bool': True if task is complete
        """
        try:
            return self.queue.task_done()
        except ValueError:
            # persistqueue raises it if task_done called too many time
            # we can ignore it
            return

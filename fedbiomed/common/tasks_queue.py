from typing import Optional, Any

import persistqueue  # used for persisiting queue on disk


exceptionsEmpty = persistqueue.exceptions.Empty

class TasksQueue:
    """
    A disk-persistant Queue object, ensuring queue will remain on
    disk even if program crashes. Relies on `persistqueue` package.
    
    """
    def __init__(self, messages_queue_dir, tmp_dir):
        self.queue = persistqueue.Queue(messages_queue_dir, tempdir=tmp_dir)

    def add(self, task: dict):
        """this method adds a task to the queue

        Args:
            task (dict): a dict describing the task to be added
        """        
        self.queue.put(task)

    def get(self, block:Optional[bool]=True) -> dict:
        """this method gets the current task in the queue

        Args:
            block (bool, optional): if True, block if necessary until an item is available. Defaults to True.

        Returns:
            [dict]: dictionary object stored in queue
        """        
        return self.queue.get(block)

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
        return self.queue.task_done()

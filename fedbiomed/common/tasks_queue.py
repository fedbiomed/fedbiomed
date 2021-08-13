import persistqueue


exceptionsEmpty = persistqueue.exceptions.Empty

class TasksQueue:
    def __init__(self, messages_queue_dir, tmp_dir):
        self.queue = persistqueue.Queue(messages_queue_dir, tempdir=tmp_dir)

    def add(self, task: dict):
        """this method adds a task to the queue

        Args:
            task (dict): a dict describing the task to be added
        """        
        self.queue.put(task)

    def get(self, block=True):
        """this method gets the current task in the queue

        Args:
            block (bool, optional): if True, block if necessary until an item is available. Defaults to True.

        Returns:
            [type]: [description]
        """        
        return self.queue.get(block)

    def qsize(self):
        """this methods returns the size of the queue

        Returns:
            int: size of the queue
        """        
        return self.queue.qsize()

    def task_done(self):
        """Indicate that a formerly enqueued task is complete

        Returns:
            'bool': True if task is complete
        """        
        return self.queue.task_done()

import os.path

import tempfile
import shutil
import unittest


from fedbiomed.common.exceptions import FedbiomedTaskQueueError
from fedbiomed.common.tasks_queue import TasksQueue


class TestTasksQueue(unittest.TestCase):
    '''
    Test the TasksQueue class
    '''
    # before the tests
    def setUp(self):
        self.tempdir   = "."
        self.queuename = next(tempfile._get_candidate_names())

    # after the tests
    def tearDown(self):
        if os.path.isfile(self.queuename) or os.path.isdir(self.queuename):
            shutil.rmtree(self.queuename)


    def test_tasksqueue(self):
        q1 = TasksQueue(self.queuename,
                        self.tempdir)

        # queue is empty at creation
        self.assertEqual( q1.qsize() , 0 )

        # add a task
        json1 = '{ "toto" }'
        q1.add( json1 )
        self.assertEqual( q1.qsize() , 1 )

        # get it back
        t1 = q1.get()
        q1.task_done()
        self.assertEqual( q1.qsize() , 0 )
        self.assertEqual( t1, json1 )

        # get data from an empty queue
        try:
            t1 = q1.get( block = False)
            # the following lines cannot be reached
            self.fail( "reading from empty queue must raise an exception" )
        except FedbiomedTaskQueueError:
            self.assertEqual( q1.qsize() , 0 )
        except:
            # this must not happen
            self.fail( "exception from reading from empty queue not catched" )

        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

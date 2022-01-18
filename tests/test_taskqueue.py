# Managing NODE, RESEARCHER environ mock before running tests
from testsupport.delete_environ import delete_environ
# Delete environ. It is necessary to rebuild environ for required component
delete_environ()
# overload with fake environ for tests
import testsupport.mock_common_environ

import tempfile
import shutil
import unittest

from fedbiomed.common.tasks_queue import TasksQueue, exceptionsEmpty

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
        shutil.rmtree(self.queuename)
        pass


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
            # the follonwing lines cannot be reached
            self.fail( "reading from empty queue must raise an exception" )
        except exceptionsEmpty:
            self.assertEqual( q1.qsize() , 0 )
        except:
            # this must not happen
            self.fail( "exception from reading from empty queue not catched" )

        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

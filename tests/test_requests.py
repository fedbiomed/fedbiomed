import unittest
from unittest.mock import Mock

import os
import sys
import time

# import a fake environment for tests bafore importing other files
import testsupport.mock_researcher_environ

from fedbiomed.researcher.job import Job

class TestJob(unittest.TestCase):

    '''
    Test the Job class
    '''
    # once in test lifetime
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    # before all the tests
    def setUp(self):
        pass

    # after all the tests
    def tearDown(self):
        pass

    # tests
    def test_job(self):
        '''

        '''
        j = Job()

        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

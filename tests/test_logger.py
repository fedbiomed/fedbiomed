import unittest

import logging

from fedbiomed.common.logger import logger
from fedbiomed.common.logger import DEFAULT_LEVEL

class TestLogger(unittest.TestCase):
    '''
    Test the Logger class
    '''
    # before all tests
    def setUp(self):
        pass

    # after all tests
    def tearDown(self):
        pass


    def test_logger_internal_translator(self):
        '''

        '''

        # string -> logging.* translator
        self.assertEqual( logger._internalLevelTranslator("DEBUG"),
                          logging.DEBUG)
        self.assertEqual( logger._internalLevelTranslator("INFO"),
                          logging.INFO)
        self.assertEqual( logger._internalLevelTranslator("WARNING"),
                          logging.WARNING)
        self.assertEqual( logger._internalLevelTranslator("ERROR"),
                          logging.ERROR)
        self.assertEqual( logger._internalLevelTranslator("CRITICAL"),
                          logging.CRITICAL)

        self.assertEqual( logger._internalLevelTranslator("STUPIDO"),
                          DEFAULT_LEVEL)


    def test_logger_internal_addhandler(self):
        '''

        '''
        # handler manager test1
        handler = logging.NullHandler()

        # console handler initialized by the constructor
        self.assertEqual( len(logger._handlers) , 1 )

        # should also return len() = 1
        logger._internalAddHandler( "CONSOLE", handler)
        self.assertEqual( len(logger._handlers) , 1 )

        # add a new one
        logger._internalAddHandler( "TEST1", handler)
        self.assertEqual( len(logger._handlers) , 2)

        # handler already exists -> count does not change
        logger._internalAddHandler( "TEST1", handler)
        self.assertEqual( len(logger._handlers) , 2)


        pass


    def test_logger_getattr(self):
        '''
        test the __getattr__
        '''

        # import the original logger class
        orig_logger = logging.getLogger('root')
        orig_logger.setLevel(logging.DEBUG)

        # method from the original Logger class
        orig_level = orig_logger.getEffectiveLevel()

        # this one uses our logger and the __getattr__ method
        logger.setLevel("DEBUG")
        over_level = logger.getEffectiveLevel()


        self.assertEqual( orig_level, over_level)
        pass


    def test_logger_singleton(self):
        '''
        test singleton mechanism
        '''

        from fedbiomed.common.logger import _FedLogger

        second_logger = _FedLogger()

        self.assertEqual( logger, second_logger)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

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
        logger.setLevel(DEFAULT_LEVEL)

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
        should not use logger._handlers (this is internal data),
        but no getter for this internal data is provided yet....
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


        # try that this fails
        try:
            logger.this_method_does_not_exists()
            self.fail("calling this_method_does_not_exists()")
        except:
            self.assertTrue( True,
                             "calling this_method_does_not_exists() detected")

        pass


    def test_logger_singleton(self):
        '''
        test singleton mechanism
        '''

        from fedbiomed.common.logger import _FedLogger

        second_logger = _FedLogger()

        self.assertEqual( logger, second_logger)

        pass

    def test_logger_setlevel(self):

        # initial DEFAULT_LEVEL
        self.assertEqual( logger.getEffectiveLevel() , DEFAULT_LEVEL)

        # modify the default level of existing handlers
        logger.setLevel( "CRITICAL" )
        self.assertEqual( logger.getEffectiveLevel() , logging.CRITICAL)

        self.assertEqual( logger._handlers["CONSOLE"].level,
                          logging.CRITICAL)

        # add a new handler and verify its initial level
        handler = logging.NullHandler()

        logger._internalAddHandler( "H_1", handler)
        self.assertEqual( logger._handlers["H_1"].level,
                          DEFAULT_LEVEL)

        # change again to critical
        logger.setLevel( "CRITICAL", "H_1")
        self.assertEqual( logger._handlers["H_1"].level,
                          logging.CRITICAL)

        # change all levels to DEBUG
        logger.setLevel( "DEBUG" )
        self.assertEqual( logger._handlers["CONSOLE"].level,
                          logging.DEBUG)
        self.assertEqual( logger._handlers["H_1"].level,
                          logging.DEBUG)
        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

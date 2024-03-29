import unittest

import logging
import tempfile
import time
import uuid


from unittest.mock import MagicMock

from fedbiomed.common.logger import logger
from fedbiomed.common.logger import DEFAULT_LOG_LEVEL


class TestLogger(unittest.TestCase):
    '''
    Test the Logger class
    '''
    def setUp(self):
        '''
        before all tests: put the loglevel to a known state
        '''
        logger.setLevel(DEFAULT_LOG_LEVEL)
        pass

    def tearDown(self):
        '''
        after all test... empty for now
        '''
        pass


    def test_logger_00_internal_translator(self):
        '''
        as test name says.... string -> logging.* translation
        '''

        # string -> logging.* translator
        self.assertEqual( logger._internal_level_translator("DEBUG"),
                          logging.DEBUG)
        self.assertEqual( logger._internal_level_translator("INFO"),
                          logging.INFO)
        self.assertEqual( logger._internal_level_translator("WARNING"),
                          logging.WARNING)
        self.assertEqual( logger._internal_level_translator("ERROR"),
                          logging.ERROR)
        self.assertEqual( logger._internal_level_translator("CRITICAL"),
                          logging.CRITICAL)

        self.assertEqual( logger._internal_level_translator("STUPIDO"),
                          DEFAULT_LOG_LEVEL)
        pass

    def test_logger_01_internal_addhandler(self):
        '''
        ulgy usage of logger._handlers (this is internal data),
        but no getter for this internal data is/should_be provided

        subject to change if internal design of logger.py is changed
        '''

        # handler manager test1
        handler = logging.NullHandler()

        # console handler initialized by the constructor
        self.assertEqual( len(logger._handlers) , 1 )

        # should also return len() = 1
        logger._internal_add_handler( "CONSOLE", handler)
        self.assertEqual( len(logger._handlers) , 1 )

        # add a new one
        logger._internal_add_handler( "TEST1", handler)
        self.assertEqual( len(logger._handlers) , 2)

        # handler already exists -> count does not change
        logger._internal_add_handler( "TEST1", handler)
        self.assertEqual( len(logger._handlers) , 2)


        pass


    def test_logger_02_getattr(self):
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
        except AttributeError:
            self.assertTrue( True,
                             "calling this_method_does_not_exists() detected")

        pass


    def test_logger_03_singleton(self):
        '''
        test singleton mechanism
        '''

        from fedbiomed.common.logger import FedLogger

        second_logger = FedLogger()

        self.assertEqual( logger, second_logger)

        pass

    def test_logger_04_setlevel(self):
        '''
        as test name says.... test the setLevel() method
        '''

        # initial DEFAULT_LOG_LEVEL
        self.assertEqual( logger.getEffectiveLevel() , DEFAULT_LOG_LEVEL)

        # chech setLevel
        for lvl in [ logging.DEBUG,
                     logging.INFO,
                     logging.WARNING,
                     logging.ERROR,
                     logging.CRITICAL ] :
            logger.setLevel(lvl)
            self.assertEqual( logger.getEffectiveLevel() , lvl)

        # bounds
        logger.setLevel( 10000 )
        self.assertEqual( logger.getEffectiveLevel() , DEFAULT_LOG_LEVEL)

        logger.setLevel( -1 )
        self.assertEqual( logger.getEffectiveLevel() , DEFAULT_LOG_LEVEL)


        # modify the default level of existing handlers
        logger.setLevel( "CRITICAL" )
        self.assertEqual( logger.getEffectiveLevel() , logging.CRITICAL)

        self.assertEqual( logger._handlers["CONSOLE"].level,
                          logging.CRITICAL)

        # add a new handler and verify its initial level
        handler = logging.NullHandler()

        logger._internal_add_handler( "H_1", handler)
        self.assertEqual( logger._handlers["H_1"].level,
                          DEFAULT_LOG_LEVEL)

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

        # not initialized handler
        with self.assertLogs('fedbiomed', logging.WARNING) as captured:
            logger.setLevel("DEBUG", "NOT_INITIALIZED_HANDLER")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "NOT_INITIALIZED_HANDLER handler not initialized yet")
        pass



    def test_logger_05_logging(self):
        '''
        as test name says.... test that logging.* levels
        '''

        # test debug() - string
        with self.assertLogs('fedbiomed', logging.DEBUG) as captured:
            logger.debug("TEST_1")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_1")

        # test info() - string
        with self.assertLogs('fedbiomed', logging.INFO) as captured:
            logger.info("TEST_2")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_2")

        # test warning() - string
        with self.assertLogs('fedbiomed', logging.WARNING) as captured:
            logger.warning("TEST_3")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_3")

        # test error() - string
        with self.assertLogs('fedbiomed', logging.ERROR) as captured:
            logger.error("TEST_4")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_4")

        # test critical() - string
        with self.assertLogs('fedbiomed', logging.CRITICAL) as captured:
            logger.critical("TEST_5")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_5")

        # test log() - string
        with self.assertLogs('fedbiomed', logging.CRITICAL) as captured:
            logger.log(logging.CRITICAL, "TEST_6")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_6")

        with self.assertLogs('fedbiomed', logging.CRITICAL) as captured:
            logger.log("CRITICAL", "TEST_7")

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "TEST_7")

    # minimal on_* handlers
    def on_message(self, client, userdata, msg):
        '''
        empty on_message handler
        '''
        pass

    def on_connect(self, client, userdata, flags, rc):
        '''
        empty on_connect handler
        '''
        pass

    def on_disconnect(self, client, userdata, flags, rc):
        '''
        empty on_disconnect handler
        '''
        pass


    def test_logger_06_grpc_handler(self):
        '''
        test grpc handler
        '''

        grpc = MagicMock()

        #
        logger.add_grpc_handler(
            on_log=grpc.send,
            node_id="dummy-id"
        )

        logger.debug("console DEBUG message", researcher_id="test-id")
        logger.error("console ERROR message")

        #
        logger.setLevel("DEBUG")
        logger.critical("verify that logger still works properly")

        pass


    def test_logger_07_filehandler(self):
        '''
        test file handler
        '''

        randomfile = tempfile.NamedTemporaryFile()

        logger.add_file_handler( filename = randomfile.name)
        logger.log("ERROR", "YYY-FIND_THIS_IN_TEMPFILE-XXX")

        # give some time to the logger
        time.sleep(2)

        # verify that the log appeared
        with open( randomfile.name ) as f:
            lines = f.readlines()

        # lines[] should contain YYY-FIND_THIS_IN_TEMPFILE-XXX
        if "YYY-FIND_THIS_IN_TEMPFILE-XXX" not in lines[0]:
            self.fail("log message not detected")
        else:
            self.assertTrue(True, "log message detected")

        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

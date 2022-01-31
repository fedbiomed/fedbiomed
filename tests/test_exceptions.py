import unittest

from fedbiomed.common.exceptions import *

class TestException(unittest.TestCase):
    '''
    Test exceptions class hierarchy
    '''

    def test_exception_environ(self):

        flag = False
        try:
            raise EnvironException("test")

        except EnvironException as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for EnvironException")


        flag = False
        try:
            raise EnvironException("test")

        except FedbiomedException as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for EnvironException")


    def test_exception_logger(self):

        flag = False
        try:
            raise LoggerException("test")

        except LoggerException as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for LoggerException")


        flag = False
        try:
            raise LoggerException("test")

        except FedbiomedException as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for LoggerException")


    def test_exception_message(self):

        flag = False
        try:
            raise MessageException("test")

        except MessageException as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for MessageException")


        flag = False
        try:
            raise MessageException("test")

        except FedbiomedException as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for MessageException")


    def test_exception_strategy(self):

        flag = False
        try:
            raise StrategyException("test")

        except StrategyException as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for StrategyException")


        flag = False
        try:
            raise StrategyException("test")

        except FedbiomedException as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for StrategyException")


    def test_exception_training(self):

        flag = False
        try:
            raise TrainingException("test")

        except TrainingException as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for TrainingException")


        flag = False
        try:
            raise TrainingException("test")

        except FedbiomedException as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for TrainingException")


if __name__ == '__main__': # pragma: no cover
    unittest.main()

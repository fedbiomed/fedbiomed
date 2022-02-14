import unittest

from fedbiomed.common.exceptions import *

class TestException(unittest.TestCase):
    '''
    Test exceptions class hierarchy
    '''

    def test_exception_environ(self):

        flag = False
        try:
            raise FedbiomedEnvironError("test")

        except FedbiomedEnvironError as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for FedbiomedEnvironError")


        flag = False
        try:
            raise FedbiomedEnvironError("test")

        except FedbiomedError as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for FedbiomedEnvironError")


    def test_exception_logger(self):

        flag = False
        try:
            raise FedbiomedLoggerError("test")

        except FedbiomedLoggerError as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for FedbiomedLoggerError")


        flag = False
        try:
            raise FedbiomedLoggerError("test")

        except FedbiomedError as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for FedbiomedLoggerError")


    def test_exception_message(self):

        flag = False
        try:
            raise FedbiomedMessageError("test")

        except FedbiomedMessageError as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for FedbiomedMessageError")


        flag = False
        try:
            raise FedbiomedMessageError("test")

        except FedbiomedError as e:
            flag = True

        except Exception as e:
            flag - False
        self.assertTrue(flag, "Bad exception was caught for FedbiomedMessageError")


    def test_exception_strategy(self):

        flag = False
        try:
            raise FedbiomedStrategyError("test")

        except FedbiomedStrategyError as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for FedbiomedStrategyError")


        flag = False
        try:
            raise FedbiomedStrategyError("test")

        except FedbiomedError as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for FedbiomedStrategyError")


    def test_exception_training(self):

        flag = False
        try:
            raise FedbiomedTrainingError("test")

        except FedbiomedTrainingError as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for FedbiomedTrainingError")


        flag = False
        try:
            raise FedbiomedTrainingError("test")

        except FedbiomedError as e:
            flag = True

        except Exception as e:
            flag - False

        self.assertTrue(flag, "Bad exception was caught for FedbiomedTrainingError")


if __name__ == '__main__': # pragma: no cover
    unittest.main()

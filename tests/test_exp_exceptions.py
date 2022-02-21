import testsupport.mock_researcher_environ
from fedbiomed.researcher.environ import environ

import unittest
import fedbiomed.researcher.experiment
from unittest.mock import patch, MagicMock, PropertyMock
from fedbiomed.researcher.experiment import exp_exceptions
from fedbiomed.common.exceptions import FedbiomedSilentTerminationError, FedbiomedError


class TestExpExceptions(unittest.TestCase):
    """ Test class for expriment.exp_exception """

    class ZMQInteractiveShell:
        """ Fake ZMQInteractiveShell class to mock get_ipython function.
            Function returns this class, so the exceptions can be raised
            as they are running on IPython kernel
        """

        def __call__(self):
            pass

    class ZMQInteractiveShellNone:
        """ Fake ZMQInteractiveShellNone class to mock get_ipython function.
            It return class name as `ZMQInteractiveShellNone` so tests
            can get into condition where the functions get runs on IPython kernel
            but not in ZMQInteractiveShell
        """

        def __call__(self):
            pass

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_exp_exceptions_1_base_exception(self):
        """ Test raising BaseException """

        @exp_exceptions
        def decFunction():
            raise BaseException

        with self.assertRaises(BaseException):
            decFunction()

    def test_exp_exceptions_2_system_Exit(self):
        """ Test  raising directly system exit"""

        # System exit on python shell
        @exp_exceptions
        def decFunction():
            raise SystemExit

        with self.assertRaises(SystemExit):
            decFunction()

    def test_exp_exceptions_3_key_int(self):
        """ Test raisin  KeyboardInterrupt error """

        @exp_exceptions
        def decFunction():
            raise KeyboardInterrupt

        with self.assertRaises(SystemExit):
            decFunction()

    def test_exp_exception_4_fedbiomed_error(self):
        """ Test raising exp FedbiomedError on notebook and python shell"""

        @exp_exceptions
        def decFunction():
            raise FedbiomedError

        # on notebook
        with patch.object(fedbiomed.researcher.experiment, 'get_ipython', create=True) as m:
            m.side_effect = TestExpExceptions.ZMQInteractiveShell
            with self.assertRaises(FedbiomedSilentTerminationError):
                decFunction()
        # on python shell
        with self.assertRaises(SystemExit):
            decFunction()

    def test_exp_exception_4_fedbiomed_silent_error(self):
        """ Test raising exp FedbiomedSilentTerminationError"""

        @exp_exceptions
        def decFunction():
            raise FedbiomedSilentTerminationError

        # SystemExit on notebook
        with self.assertRaises(FedbiomedSilentTerminationError):
            decFunction()

    def test_exp_exception_5_interactive_shell_fase(self):
        """ Test if get_ipython does not return ZMQInteractiveShell"""

        @exp_exceptions
        def decFunction():
            raise FedbiomedError

        # SystemExit on python shell
        with patch.object(fedbiomed.researcher.experiment, 'get_ipython', create=True) as m:
            m.side_effect = TestExpExceptions.ZMQInteractiveShellNone
            with self.assertRaises(SystemExit):
                decFunction()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

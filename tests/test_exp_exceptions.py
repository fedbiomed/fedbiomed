import unittest
from unittest.mock import patch

#############################################################
# Import ResearcherTestCase before importing any FedBioMed Module
from testsupport.base_case import ResearcherTestCase
#############################################################

import fedbiomed.researcher.experiment
from fedbiomed.common.exceptions import FedbiomedSilentTerminationError, FedbiomedError
from fedbiomed.researcher.experiment import exp_exceptions


class TestExpExceptions(ResearcherTestCase):
    """ Test class for expriment.exp_exception """

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
        with patch.object(fedbiomed.researcher.experiment, 'is_ipython', create=True) as m:
            m.return_value = True
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

    def test_exp_exception_5_interactive_shell_false(self):
        """ Test if get_ipython does not return ZMQInteractiveShell"""

        @exp_exceptions
        def decFunction():
            raise FedbiomedError

        # SystemExit on python shell
        with patch.object(fedbiomed.researcher.experiment, 'is_ipython', create=True) as m:
            m.return_value = False
            with self.assertRaises(SystemExit):
                decFunction()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

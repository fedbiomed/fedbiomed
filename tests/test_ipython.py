import unittest
import fedbiomed.common.ipython as fb_ipython
import fedbiomed.common.ipython

from unittest.mock import patch


class TestIpython(unittest.TestCase):
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

    def setUp(self):
        pass

    def tearDown(self) -> None:
        pass

    def test_utils_01_is_ipython(self):
        """ Test the function is_ipython """

        with patch.object(fedbiomed.common.ipython, 'get_ipython', create=True) as mock_get_ipython:
            mock_get_ipython.side_effect = TestIpython.ZMQInteractiveShell
            result = fb_ipython.is_ipython()
            self.assertTrue(result, f'`is_python` has returned {result}, while the expected is `True`')

            mock_get_ipython.side_effect = TestIpython.ZMQInteractiveShell
            result = fb_ipython.is_ipython()
            self.assertTrue(result, f'`is_python` has returned {result}, while the expected is `True`')

            mock_get_ipython.side_effect = TestIpython.ZMQInteractiveShellNone
            result = fb_ipython.is_ipython()
            self.assertFalse(result, f'`is_python` has returned {result}, while the expected is `False`')

            mock_get_ipython.side_effect = NameError
            result = fb_ipython.is_ipython()
            self.assertFalse(result, f'`is_python` has returned {result}, while the expected is `False`')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

import os
import sys
import unittest
import fedbiomed.common.utils as fed_utils
import fedbiomed.common.utils

from fedbiomed.common.exceptions import FedbiomedError
from unittest.mock import patch



# Dummy Class for testing its source --------------------
class TestClass:
    def __init__(self):
        pass
# -------------------------------------------------------


class TestUtils(unittest.TestCase):
    class ZMQInteractiveShell:
        """ Fake ZMQInteractiveShell class to mock get_ipython function.
            Function returns this class, so the exceptions can be raised
            as they are running on IPython kernel
        """

        def __call__(self):
            pass

    class ZMQInteractiveShell:
        """ Fake TerminalInteractiveShell class to mock get_ipython function.
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

    @patch('fedbiomed.common.utils.is_ipython')
    @patch('fedbiomed.common.utils.get_ipython_class_file')
    @patch('inspect.linecache.getlines')
    def test_utils_01_get_class_source(self,
                                       mock_get_lines,
                                       mock_get_ipython_class_file,
                                       mock_is_ipython):
        """
        Tests getting class source
        """
        class_source = [
            'class TestClass:\n',
            '\tdef __init__(self):\n',
            '\t\tpass\n'
        ]

        # Test getting class source when is_ipython returns True
        expected_cls_source = "".join(class_source)
        mock_get_lines.return_value = class_source
        mock_get_ipython_class_file.return_value = None
        mock_is_ipython.return_value = True

        codes = fed_utils.get_class_source(TestClass)
        self.assertEqual(codes, expected_cls_source)

        # Test getting class source if is_python returns False
        mock_is_ipython.return_value = False
        codes = fed_utils.get_class_source(TestClass)
        self.assertEqual(codes, expected_cls_source)

        # Test if `cls` is not a class
        obj = TestClass()
        with self.assertRaises(FedbiomedError):
            fed_utils.get_class_source(obj)

    def test_utils_02_is_ipython(self):
        """ Test the function is_ipython """

        with patch.object(fedbiomed.common.utils._utils, 'get_ipython', create=True) as mock_get_ipython:
            mock_get_ipython.side_effect = TestUtils.ZMQInteractiveShell
            result = fed_utils.is_ipython()
            self.assertTrue(result, f'`is_python` has returned {result}, while the expected is `True`')

            mock_get_ipython.side_effect = TestUtils.ZMQInteractiveShell
            result = fed_utils.is_ipython()
            self.assertTrue(result, f'`is_python` has returned {result}, while the expected is `True`')

            mock_get_ipython.side_effect = TestUtils.ZMQInteractiveShellNone
            result = fed_utils.is_ipython()
            self.assertFalse(result, f'`is_python` has returned {result}, while the expected is `False`')

            mock_get_ipython.side_effect = NameError
            result = fed_utils.is_ipython()
            self.assertFalse(result, f'`is_python` has returned {result}, while the expected is `False`')

    def test_utils_03_get_ipython_class_file(self):
        """ Testing function that gets class source from ipython kernel"""

        # Test normal case
        result = fed_utils.get_ipython_class_file(TestClass)
        self.assertTrue(os.path.isfile(result), 'The result of class_file is not a file')

        with patch.object(fedbiomed.common.utils._utils, 'hasattr') as mock_hasattr:
            mock_hasattr.return_value = False
            with self.assertRaises(FedbiomedError):
                fed_utils.get_ipython_class_file(TestClass)

        with patch.object(sys, 'modules') as mock_sys_modules:
            mock_sys_modules.return_value = {}
            result = fed_utils.get_ipython_class_file(TestClass)
            self.assertTrue(os.path.isfile(result), 'The result of class_file is not a file')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

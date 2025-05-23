import os
import sys
import unittest
import fedbiomed.common.utils as fed_utils
import fedbiomed.common.utils

from fedbiomed.common.exceptions import FedbiomedError

from unittest.mock import patch
from unittest.mock import MagicMock


# Dummy Class for testing its source --------------------
class TestClass:
    def __init__(self):
        pass


# -------------------------------------------------------


class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self) -> None:
        pass

    @patch("fedbiomed.common.utils._utils.is_ipython")
    @patch("fedbiomed.common.utils.get_ipython_class_file")
    @patch("inspect.linecache.getlines")
    @patch("fedbiomed.common.utils._utils.importlib")
    def test_utils_01_get_class_source(
        self,
        mock_importlib,
        mock_get_lines,
        mock_get_ipython_class_file,
        mock_is_ipython,
    ):
        """
        Tests getting class source
        """
        class_source = ["class TestClass:\n", "\tdef __init__(self):\n", "\t\tpass\n"]

        # Test getting class source when is_ipython returns True
        expected_cls_source = "".join(class_source)
        mock_get_lines.return_value = class_source
        mock_get_ipython_class_file.return_value = None
        mock_is_ipython.return_value = True
        mock_importlib.import_module.return_value.extract_symbols.return_value = [
            [expected_cls_source]
        ]

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

    def test_utils_02_get_ipython_class_file(self):
        """Testing function that gets class source from ipython kernel"""

        # Test normal case
        result = fed_utils.get_ipython_class_file(TestClass)
        self.assertTrue(
            os.path.isfile(result), "The result of class_file is not a file"
        )

        with patch.object(fedbiomed.common.utils._utils, "hasattr") as mock_hasattr:
            mock_hasattr.return_value = False
            with self.assertRaises(FedbiomedError):
                fed_utils.get_ipython_class_file(TestClass)

        with patch.object(sys, "modules") as mock_sys_modules:
            mock_sys_modules.return_value = {}
            result = fed_utils.get_ipython_class_file(TestClass)
            self.assertTrue(
                os.path.isfile(result), "The result of class_file is not a file"
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

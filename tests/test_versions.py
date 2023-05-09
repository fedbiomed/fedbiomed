import unittest
import logging
import sys, io
from packaging.version import Version
import fedbiomed, fedbiomed.researcher, fedbiomed.node
from fedbiomed.common.exceptions import FedbiomedVersionError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils._versions import raise_for_version_compatibility


class TestVersions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # REDIRECT all logging output to string stream
        logger._internalAddHandler("CONSOLE", None)
        cls.logging_output = io.StringIO()
        cls.handler = logging.StreamHandler(cls.logging_output)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s - %(message)s')
        cls.handler.setFormatter(formatter)  # copy console format
        logger._logger.addHandler(cls.handler)
        # END REDIRECT

    @classmethod
    def tearDownClass(cls) -> None:
        logger._logger.removeHandler(cls.handler)
        logger.addConsoleHandler()

    def test_versions_01_version_numbers(self):
        self.assertTrue(fedbiomed.__version__ >= Version('4.3'))
        self.assertTrue(fedbiomed.researcher.__config_version__ >= Version('1.0'))
        self.assertFalse(fedbiomed.node.__config_version__ < Version('1.0'))

    def test_versions_02_check_version_compatibility(self):
        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility(Version('1.0'), Version('2.0'), 'v1 %s v2 %s')
        self.assertEqual(str(e.exception), 'v1 1.0 v2 2.0')

        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility(Version('1.0'), '4.0', 'v1 %s v2 %s')
        self.assertEqual(str(e.exception), 'v1 1.0 v2 4.0')

        self.logging_output.truncate(0)  # clear the logging buffer for simplicity
        raise_for_version_compatibility(Version('1.1'), Version('1.5'), 'v1 %s v2 %s')
        self.assertEqual(self.logging_output.getvalue()[-34:],
                         'fedbiomed WARNING - v1 1.1 v2 1.5\n')

        self.logging_output.truncate(0)  # clear the logging buffer for simplicity
        raise_for_version_compatibility('1.1', '1.5', 'v1 %s v2 %s')
        self.assertEqual(self.logging_output.getvalue()[-34:],
                         'fedbiomed WARNING - v1 1.1 v2 1.5\n')

        self.logging_output.truncate(0)  # clear the logging buffer for simplicity
        orig_level = logger.level
        logger.setLevel('INFO')
        raise_for_version_compatibility(Version('1.1.2'), Version('1.1.5'), 'v1 %s v2 %s')
        self.assertEqual(self.logging_output.getvalue()[-35:],
                         'fedbiomed INFO - v1 1.1.2 v2 1.1.5\n')
        logger.setLevel(orig_level)

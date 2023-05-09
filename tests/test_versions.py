import unittest
from unittest.mock import patch
import logging
import sys, io
from packaging.version import Version
import fedbiomed, fedbiomed.researcher, fedbiomed.node
from fedbiomed.common.exceptions import FedbiomedVersionError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils._versions import raise_for_version_compatibility


class TestVersions(unittest.TestCase):
    def setUp(self) -> None:
        self.patch_versions_log = patch.object(fedbiomed.common.utils._versions, 'logger')
        self.mock_versions_log = self.patch_versions_log.start()

    def tearDown(self) -> None:
        self.patch_versions_log.stop()

    def test_versions_01_version_numbers(self):
        self.assertTrue(fedbiomed.__version__ >= Version('4.3'))
        self.assertTrue(fedbiomed.researcher.__config_version__ >= Version('1.0'))
        self.assertFalse(fedbiomed.node.__config_version__ < Version('1.0'))

    def test_versions_02_check_version_compatibility(self):
        self.mock_versions_log.reset_mock()
        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility(Version('1.0'), Version('2.0'), 'v1 %s v2 %s')
        self.assertEqual(str(e.exception), 'v1 1.0 v2 2.0')
        self.assertEqual(self.mock_versions_log.critical.call_count, 1)
        self.assertEqual(self.mock_versions_log.critical.call_args[0][0], 'v1 1.0 v2 2.0')

        self.mock_versions_log.reset_mock()
        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility(Version('1.0'), '4.0', 'v1 %s v2 %s')
        self.assertEqual(str(e.exception), 'v1 1.0 v2 4.0')
        self.assertEqual(self.mock_versions_log.critical.call_count, 1)
        self.assertEqual(self.mock_versions_log.critical.call_args[0][0], 'v1 1.0 v2 4.0')

        self.mock_versions_log.reset_mock()
        raise_for_version_compatibility(Version('1.1'), Version('1.5'), 'v1 %s v2 %s')
        self.assertEqual(self.mock_versions_log.warning.call_count, 1)
        self.assertEqual(self.mock_versions_log.warning.call_args[0][0], 'v1 1.1 v2 1.5')

        self.mock_versions_log.reset_mock()
        raise_for_version_compatibility('1.1', '1.5', 'v1 %s v2 %s')
        self.assertEqual(self.mock_versions_log.warning.call_count, 1)
        self.assertEqual(self.mock_versions_log.warning.call_args[0][0], 'v1 1.1 v2 1.5')

        self.mock_versions_log.reset_mock()
        raise_for_version_compatibility(Version('1.1.2'), Version('1.1.5'), 'v1 %s v2 %s')
        self.assertEqual(self.mock_versions_log.info.call_count, 1)
        self.assertEqual(self.mock_versions_log.info.call_args[0][0], 'v1 1.1.2 v2 1.1.5')


if __name__ == "__main__":
    unittest.main()

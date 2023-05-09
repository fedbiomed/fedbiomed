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
        """Test runtime version numbers hardcoded in constants"""
        self.assertTrue(fedbiomed.__version__ >= Version('4.3'))
        self.assertTrue(fedbiomed.researcher.__config_version__ >= Version('1.0'))
        self.assertFalse(fedbiomed.node.__config_version__ < Version('1.0'))

    def test_versions_02_check_version_compatibility(self):
        """Test raise_for_version_compatibility"""

        # Error case: incompatible versions
        # versions specified as packaging.Version type
        self.mock_versions_log.reset_mock()
        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility(Version('1.0'), Version('2.0'), 'v1 %s v2 %s')
        self.assertEqual(str(e.exception)[:13], 'v1 1.0 v2 2.0')
        self.assertEqual(self.mock_versions_log.critical.call_count, 1)
        self.assertEqual(self.mock_versions_log.critical.call_args[0][0][:13], 'v1 1.0 v2 2.0')

        # Error case: incompatible versions
        # versions specified as str
        self.mock_versions_log.reset_mock()
        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility(Version('1.0'), '4.0', 'v1 %s v2 %s')
        self.assertEqual(str(e.exception)[:13], 'v1 1.0 v2 4.0')
        self.assertEqual(self.mock_versions_log.critical.call_count, 1)
        self.assertEqual(self.mock_versions_log.critical.call_args[0][0][:13], 'v1 1.0 v2 4.0')

        # Warning case: difference in minor version
        # versions specified as packaging.Version type
        self.mock_versions_log.reset_mock()
        raise_for_version_compatibility(Version('1.1'), Version('1.5'), 'v1 %s v2 %s')
        self.assertEqual(self.mock_versions_log.warning.call_count, 1)
        self.assertEqual(self.mock_versions_log.warning.call_args[0][0][:13], 'v1 1.1 v2 1.5')

        # Warning case: difference in minor version
        # versions specified as str
        self.mock_versions_log.reset_mock()
        raise_for_version_compatibility('1.1', '1.5', 'v1 %s v2 %s')
        self.assertEqual(self.mock_versions_log.warning.call_count, 1)
        self.assertEqual(self.mock_versions_log.warning.call_args[0][0][:13], 'v1 1.1 v2 1.5')

        # Info case: difference in micro version
        self.mock_versions_log.reset_mock()
        raise_for_version_compatibility(Version('1.1.2'), Version('1.1.5'), 'v1 %s v2 %s')
        self.assertEqual(self.mock_versions_log.info.call_count, 1)
        self.assertEqual(self.mock_versions_log.info.call_args[0][0][:17], 'v1 1.1.2 v2 1.1.5')

        # Base case: same version
        self.mock_versions_log.reset_mock()
        raise_for_version_compatibility(Version('1.1.2'), Version('1.1.2'), 'v1 %s v2 %s')
        self.assertEqual(self.mock_versions_log.info.call_count, 0)
        self.assertEqual(self.mock_versions_log.warning.call_count, 0)
        self.assertEqual(self.mock_versions_log.critical.call_count, 0)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch
from packaging.version import Version
import fedbiomed
from fedbiomed.common.exceptions import FedbiomedVersionError
from fedbiomed.common.utils._versions import (
    raise_for_version_compatibility,
    _create_msg_for_version_check,
)


class TestVersions(unittest.TestCase):
    def setUp(self) -> None:
        self.patch_versions_log = patch.object(
            fedbiomed.common.utils._versions, "logger"
        )
        self.mock_versions_log = self.patch_versions_log.start()

    def tearDown(self) -> None:
        self.patch_versions_log.stop()

    def test_versions_01_version_numbers(self):
        """Test runtime version numbers hardcoded in constants"""
        self.assertTrue(Version(fedbiomed.__version__) >= Version("4.3"))
        self.assertEqual(fedbiomed.common.utils.__default_version__, Version("0"))

    def test_versions_02_check_version_compatibility(self):
        """Test raise_for_version_compatibility"""

        # Error case: bad type for component version
        self.mock_versions_log.reset_mock()
        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility(2, Version("2.1"), "arg1 %s arg2 %s")
        self.assertEqual(
            str(e.exception),
            "FB625: Component version error: Component version has incorrect type "
            "`their_version` type=<class 'int'> value=2",
        )
        self.assertEqual(self.mock_versions_log.critical.call_count, 1)
        self.assertEqual(
            self.mock_versions_log.critical.call_args[0][0],
            "FB625: Component version error: Component version has incorrect type "
            "`their_version` type=<class 'int'> value=2",
        )

        # Error case: bad type for software version
        self.mock_versions_log.reset_mock()
        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility(Version("2.1"), 2, "arg1 %s arg2 %s")
        self.assertEqual(
            str(e.exception),
            "FB625: Component version error: Component version has incorrect type "
            "`our_version` type=<class 'int'> value=2",
        )
        self.assertEqual(self.mock_versions_log.critical.call_count, 1)
        self.assertEqual(
            self.mock_versions_log.critical.call_args[0][0],
            "FB625: Component version error: Component version has incorrect type "
            "`our_version` type=<class 'int'> value=2",
        )

        # Error case: incompatible major versions
        # versions specified as packaging.Version type
        self.mock_versions_log.reset_mock()
        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility(
                Version("1.0"), Version("2.0"), "v1 %s v2 %s"
            )
        self.assertEqual(str(e.exception)[:13], "v1 1.0 v2 2.0")
        self.assertEqual(self.mock_versions_log.critical.call_count, 1)
        self.assertEqual(
            self.mock_versions_log.critical.call_args[0][0][:13], "v1 1.0 v2 2.0"
        )

        # Error case: incompatible major versions
        # versions specified as str
        self.mock_versions_log.reset_mock()
        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility(Version("1.0"), "4.0", "v1 %s v2 %s")
        self.assertEqual(str(e.exception)[:13], "v1 1.0 v2 4.0")
        self.assertEqual(self.mock_versions_log.critical.call_count, 1)
        self.assertEqual(
            self.mock_versions_log.critical.call_args[0][0][:13], "v1 1.0 v2 4.0"
        )

        # Warning case: difference in minor version
        # versions specified as packaging.Version type
        self.mock_versions_log.reset_mock()
        raise_for_version_compatibility(
            Version("1.1.1"), Version("1.5.1"), "v1 %s v2 %s"
        )
        self.assertEqual(self.mock_versions_log.warning.call_count, 1)
        self.assertEqual(
            self.mock_versions_log.warning.call_args[0][0][:17], "Found version 1.1"
        )

        # Warning case: difference in minor version
        # versions specified as str
        self.mock_versions_log.reset_mock()
        raise_for_version_compatibility("1.1.5", "1.5.1", "v1 %s v2 %s")
        self.assertEqual(self.mock_versions_log.warning.call_count, 1)
        self.assertEqual(
            self.mock_versions_log.warning.call_args[0][0][:17], "Found version 1.1"
        )

        # Error case: the minor version in the static component file (e.g. config file) is higher
        # than that of the runtime
        self.mock_versions_log.reset_mock()
        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility("1.5.2", "1.1.1", "v1 %s v2 %s")
        self.assertEqual(str(e.exception)[:17], "v1 1.5.2 v2 1.1.1")
        self.assertEqual(self.mock_versions_log.critical.call_count, 1)
        self.assertEqual(
            self.mock_versions_log.critical.call_args[0][0][:17], "v1 1.5.2 v2 1.1.1"
        )

        # Warning case: difference in micro version
        self.mock_versions_log.reset_mock()
        raise_for_version_compatibility(
            Version("1.1.2"), Version("1.1.5"), "v1 %s v2 %s"
        )
        self.assertEqual(self.mock_versions_log.warning.call_count, 1)
        self.assertEqual(
            self.mock_versions_log.warning.call_args[0][0][:17], "Found version 1.1"
        )

        # Error case: the micro version in the static component file (e.g. config file) is higher
        # than that of the runtime
        self.mock_versions_log.reset_mock()
        with self.assertRaises(FedbiomedVersionError) as e:
            raise_for_version_compatibility("1.5.5", "1.5.1", "v1 %s v2 %s")
        self.assertEqual(str(e.exception)[:17], "v1 1.5.5 v2 1.5.1")
        self.assertEqual(self.mock_versions_log.critical.call_count, 1)
        self.assertEqual(
            self.mock_versions_log.critical.call_args[0][0][:17], "v1 1.5.5 v2 1.5.1"
        )

        # Base case: same version
        self.mock_versions_log.reset_mock()
        raise_for_version_compatibility(
            Version("1.1.2"), Version("1.1.2"), "v1 %s v2 %s"
        )
        self.assertEqual(self.mock_versions_log.info.call_count, 0)
        self.assertEqual(self.mock_versions_log.warning.call_count, 0)
        self.assertEqual(self.mock_versions_log.critical.call_count, 0)

    def test_versions_03_message_for_version_check(self):
        """Test _create_msg_for_version_check"""

        # No need for additional test for successful case

        # Error case: bad error message pattern
        bad_patterns = ["no param", "one param %s", "three params %s %s %s"]
        for bad_pattern in bad_patterns:
            message = _create_msg_for_version_check(
                bad_pattern, Version("2.2"), Version("2.2")
            )
            self.assertEqual(
                message,
                bad_pattern
                + " -> See https://fedbiomed.org/latest/user-guide/deployment/versions for more information",
            )


if __name__ == "__main__":
    unittest.main()

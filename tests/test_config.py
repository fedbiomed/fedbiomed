import shutil
import unittest
from unittest.mock import patch

from fedbiomed.common.config import Config
from fedbiomed.common.exceptions import FedbiomedVersionError
from fedbiomed.node.config import NodeConfig
from fedbiomed.researcher.config import ResearcherConfig


class BaseConfigTest(unittest.TestCase):
    def setUp(self):
        self.patch_fed_folders = patch(
            "fedbiomed.common.config.create_fedbiomed_setup_folders"
        )
        self.patch_open = patch("builtins.open")

        self.open_mock = self.patch_open.start()
        self.create_fed_folders_mock = self.patch_fed_folders.start()

    def tearDown(self):
        self.patch_open.stop()
        self.patch_fed_folders.stop()

        # Clean up any created folders
        shutil.rmtree("dummy-root", ignore_errors=True)
        shutil.rmtree("etc", ignore_errors=True)
        return super().tearDown()


class TestConfig(BaseConfigTest):
    """Tests base methods of Config class"""

    def setUp(self):
        self.patch_ = patch.multiple(
            "fedbiomed.common.config.Config", __abstractmethods__=set()
        )
        self.patch_.start()

        return super().setUp()

    def tearDown(self):
        self.patch_.stop()
        return super().tearDown()

    @patch("fedbiomed.common.config.configparser.ConfigParser")
    def test_01_read(self, config_parser):
        config = Config(root="dummy-root")
        config._CONFIG_VERSION = "0.99"

        with self.assertRaises(FedbiomedVersionError):
            config.read()

        config_parser.return_value.read.assert_called_once()

        # With autogenereate
        with patch("fedbiomed.common.config.Config.generate") as gen:
            config = Config(root="dummy-root")
            gen.assert_called_once()

    def test_03_is_config_existing(self):
        config = Config(root="test")
        r = config.is_config_existing()
        self.assertFalse(r)


class TestNodeConfig(BaseConfigTest):
    def test_01_node_config_generate(self):
        config = NodeConfig(root="tests")
        print(config._cfg["default"])

        component = config.get("default", "component")
        self.assertEqual("node", component.lower())

        r_ip = config.get("researcher", "ip")
        r_port = config.get("researcher", "port")

        self.assertTrue(r_ip)
        self.assertTrue(r_port)

    def test_02_node_config_sections(self):
        config = NodeConfig(root="test")

        sections = config.sections()

        self.assertTrue("researcher" in sections)
        self.assertTrue("default" in sections)
        self.assertTrue("security" in sections)
        self.assertTrue("certificate" in sections)


class TestResearcherConfig(BaseConfigTest):
    def test_01_researcher_config_generate(self):
        config = ResearcherConfig(root="tests")

        component = config.get("default", "component")
        self.assertEqual("researcher", component.lower())

        r_ip = config.get("server", "host")
        r_port = config.get("server", "port")
        r_pem = config.get("certificate", "private_key")
        r_key = config.get("certificate", "public_key")

        self.assertTrue(r_ip)
        self.assertTrue(r_port)
        self.assertTrue(r_pem)
        self.assertTrue(r_key)

    def test_02_researcher_config_sections(self):
        config = ResearcherConfig(root="test")
        sections = config.sections()

        self.assertTrue("server" in sections)
        self.assertTrue("default" in sections)


if __name__ == "__main__":
    unittest.main()

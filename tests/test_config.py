import unittest
import tempfile

from unittest.mock import patch

from fedbiomed.common.config import Config
from fedbiomed.researcher.config import ResearcherConfig
from fedbiomed.node.config import NodeConfig
from fedbiomed.common.exceptions import FedbiomedVersionError, FedbiomedError


class BaseConfigTest(unittest.TestCase):

    def setUp(self):
        self.patch_open = patch('builtins.open')

        self.open_mock = self.patch_open.start()

        self.tem = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.patch_open.stop()
        self.tem.cleanup()

class TestConfig(BaseConfigTest):
    """Tests base methods of Config class"""
    def setUp(self):

        self.patch_ = patch.multiple('fedbiomed.common.config.Config', __abstractmethods__=set())
        self.patch_.start()

        return  super().setUp()

    def tearDown(self):

        self.patch_.stop()

        return super().tearDown()


    @patch('fedbiomed.common.config.configparser.ConfigParser')
    def test_01_read(self, config_parser):

        config = Config(root=self.tem.name, auto_generate=False)
        config._CONFIG_VERSION = '0.99'

        with self.assertRaises(FedbiomedVersionError):
            config.read()

        config_parser.return_value.read.assert_called_once()

        # With autogenereate
        with patch('fedbiomed.common.config.Config.generate') as gen:
            config = Config(root=self.tem.name, auto_generate=True)
            gen.assert_called_once()


    def test_02_init_with_root(self):

        Config(auto_generate=False, root=self.tem.name)


    def test_03_is_config_existing(self):

        config = Config(root=self.tem.name, auto_generate=False)
        r = config.is_config_existing()
        self.assertFalse(r)


    @patch('fedbiomed.common.config.Config.generate')
    @patch('fedbiomed.common.config.Config.is_config_existing')
    @patch('fedbiomed.common.config.configparser.ConfigParser.read')
    def test_04_refresh(self, read_, is_existing, generate):

        is_existing.return_value = False
        config = Config(auto_generate=False, root=self.tem.name)
        with self.assertRaises(FedbiomedError):
            config.refresh()

        def set_(path):
            config._cfg = {"default" : {"id": "test"}}

        read_.side_effect = set_
        is_existing.return_value = True
        config.refresh()
        generate.assert_called_once()


class TestNodeConfig(BaseConfigTest):

    def test_01_node_config_generate(self):

        config = NodeConfig(root=self.tem.name)
        print(config._cfg['default'])

        component = config.get('default', 'component')
        self.assertEqual('node', component.lower())

        r_ip = config.get('researcher', 'ip')
        r_port = config.get('researcher', 'port')

        self.assertTrue(r_ip)
        self.assertTrue(r_port)


    def test_02_node_config_sections(self):

        config = NodeConfig(root=self.tem.name)

        sections = config.sections()

        self.assertTrue('researcher' in sections)
        self.assertTrue('default' in sections)
        self.assertTrue('security' in sections)
        self.assertTrue('certificate' in sections)



class TestResearcherConfig(BaseConfigTest):

    def test_01_researcher_config_generate(self):

        config = ResearcherConfig(root=self.tem.name)

        component = config.get('default', 'component')
        self.assertEqual('researcher', component.lower())

        r_ip = config.get('server', 'host')
        r_port = config.get('server', 'port')
        r_pem = config.get('certificate', 'private_key')
        r_key = config.get('certificate', 'public_key')

        self.assertTrue(r_ip)
        self.assertTrue(r_port)
        self.assertTrue(r_pem)
        self.assertTrue(r_key)

    def test_02_researcher_config_sections(self):

        config = ResearcherConfig(root=self.tem.name)

        sections = config.sections()

        self.assertTrue('server' in sections)
        self.assertTrue('default' in sections)

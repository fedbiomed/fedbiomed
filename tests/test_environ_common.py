import unittest
import os
import tempfile
import shutil
from unittest import TestCase
from unittest.mock import patch

import fedbiomed
from fedbiomed.common.exceptions import FedbiomedEnvironError, FedbiomedVersionError
from fedbiomed.common.environ import Environ


class TestEnviron(TestCase):

    @staticmethod
    def clean_singleton():
        # Clean singleton classes
        if Environ in Environ._objects:
            del Environ._objects[Environ]

    def setUp(self):
        self.config_subdir = next(tempfile._get_candidate_names())
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))

        self.config_dir = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../etc')), self.config_subdir)
        # needed if etc. does not exist (fresh installation)
        os.makedirs(self.config_dir)

        self.patcher = patch.multiple(Environ, __abstractmethods__=set())
        self.abstract_methods = self.patcher.start()

        self.patch_versions_log = patch.object(fedbiomed.common.utils._versions, 'logger')
        self.mock_versions_log = self.patch_versions_log.start()

        self.environ = Environ()

    def tearDown(self) -> None:

        # Stop abstract method patcher
        self.patcher.stop()

        # Clean singleton classes
        TestEnviron.clean_singleton()

        shutil.rmtree(self.config_dir)

    def test_environ_01_initialize_common_variables_01(self):
        """ Test initialize common variables """
        self.environ._initialize_common_variables()
        values = self.environ._values

        self.assertTrue("ROOT_DIR" in values)
        self.assertEqual(os.path.realpath(values["ROOT_DIR"]), os.path.realpath(self.base_dir))
        self.assertEqual(os.path.realpath(values["CONFIG_DIR"]), os.path.join(os.path.realpath(self.base_dir), 'etc'))
        self.assertEqual(os.path.realpath(values["VAR_DIR"]), os.path.join(os.path.realpath(self.base_dir), 'var'))
        self.assertEqual(os.path.realpath(values["TMP_DIR"]), os.path.join(os.path.realpath(self.base_dir), 'var', 'tmp'))
        self.assertEqual(os.path.realpath(values["CACHE_DIR"]), os.path.join(os.path.realpath(self.base_dir), 'var', 'cache'))
        self.assertEqual(os.path.realpath(values["PORT_INCREMENT_FILE"]), os.path.join(os.path.realpath(self.base_dir), 'etc', 'port_increment'))

    def test_environ_01_initialize_common_variables_02(self):
        """ Test initialize common variables with root dir"""

        root_dir = '/tmp'

        # Clean environ singleton before creating new object
        TestEnviron.clean_singleton()

        environ = Environ(root_dir=root_dir)
        environ._initialize_common_variables()
        values = environ._values

        self.assertTrue("ROOT_DIR" in values)
        self.assertEqual(values["ROOT_DIR"], root_dir)
        self.assertEqual(values["CONFIG_DIR"], os.path.join(root_dir, 'etc'))
        self.assertEqual(values["VAR_DIR"], os.path.join(root_dir, 'var'))
        self.assertEqual(values["TMP_DIR"], os.path.join(root_dir, 'var', 'tmp'))
        self.assertEqual(values["CACHE_DIR"], os.path.join(root_dir, 'var', 'cache'))
        self.assertEqual(values["PORT_INCREMENT_FILE"], os.path.join(root_dir, 'etc', 'port_increment'))

    @patch("fedbiomed.common.environ.Environ.default_config_file")
    def test_environ_03_set_config_file(self, default_config_file):
        """Test creating default config files"""

        config_file = os.path.join(self.config_dir, 'test_config.ini')

        # Set variables
        os.environ['CONFIG_FILE'] = str(config_file)

        self.clean_singleton()
        environ = Environ()

        environ.set_config_file()
        self.assertEqual(environ._values["CONFIG_FILE"], config_file)

        # Check whether calls abstract method
        del os.environ['CONFIG_FILE']
        environ.set_config_file()
        default_config_file.assert_called_once()

        if os.path.isfile(config_file):
            os.remove(config_file)

    def test_environ_03_write_config_file(self):
        """Test creating default config files"""

        config_file = os.path.join(self.config_dir, 'test_config.ini')

        # Set variables
        os.environ['CONFIG_FILE'] = str(config_file)

        self.clean_singleton()
        environ = Environ()

        environ.set_config_file()
        environ._write_config_file()

        self.assertTrue(os.path.isfile(config_file), "Config file is not writen as expected")

    def test_environ_04_retrieve_ip_and_port(self):
        """Tests writing increment file """

        increment_file = os.path.join(self.config_dir, "port_increment")
        ip, port = self.environ._retrieve_ip_and_port(increment_file)

        file = open(increment_file, "r")
        inc = file.read()
        file.close()
        self.assertEqual("14000", inc, "Port/IP is not writen in port increment file as expected")
        self.assertEqual(ip, "localhost")
        self.assertEqual(port, 14000)

        ip, port = self.environ._retrieve_ip_and_port(increment_file)
        self.assertEqual(ip, 'localhost')
        self.assertEqual(port, 14001)
        file = open(increment_file, "r")
        inc = file.read()
        file.close()
        self.assertEqual("14001", inc, "Port/IP is not writen in port increment file as expected")

        ip, port = self.environ._retrieve_ip_and_port(increment_file, new=True)
        self.assertEqual(ip, 'localhost')
        self.assertEqual(port, 14000)

        ip, port = self.environ._retrieve_ip_and_port(increment_file, new=True, increment=15000)
        self.assertEqual(ip, 'localhost')
        self.assertEqual(port, 15000)

    def test_environ_05_configure_secure_aggregation(self):
        """Tests methods configure secure aggregation """
        increment_file = os.path.join(self.config_dir, "port_increment")
        self.environ._values["PORT_INCREMENT_FILE"] = increment_file

        cert_folder = os.path.join(self.config_dir, "certs")
        self.environ._values["CERT_DIR"] = cert_folder
        self.environ._values["CONFIG_DIR"] = self.config_dir

        self.environ._cfg["default"] = {"id": "test_component_id"}

        self.environ._configure_secure_aggregation()

        ip = self.environ._cfg.get("mpspdz", "mpspdz_ip")
        port = self.environ._cfg.get("mpspdz", "mpspdz_port")

        self.assertEqual(ip, "localhost", "Host is not set properly in config object")
        self.assertEqual(port, '14000', "Port is not set in config object")


    def test_environ_08_set_network_variables(self):
        """Tests setting network variables """

        # Prepare config file
        broker_ip = "0.0.0.0"
        broker_port = 3434
        mpspdz_ip = "1.1.1.1"
        mpspdz_port = "1234"
        uploads_url = "http"
        public_key = "text_public_key"
        private_key = "test_private_key"
        allow_default_biprimes = True


        self.environ._cfg["mqtt"] = {'broker_ip': broker_ip, 'port': broker_port}
        self.environ._cfg["mpspdz"] = {
            'mpspdz_ip': mpspdz_ip,
            'mpspdz_port': mpspdz_port,
            'public_key': public_key,
            'private_key': private_key,
            'allow_default_biprimes': allow_default_biprimes,
        }

        self.environ._cfg["default"] = {'uploads_url': uploads_url}


        if "ALLOW_DEFAULT_BIPRIMES" in os.environ:
            del os.environ["ALLOW_DEFAULT_BIPRIMES"]

        # MPSPDZ key paths requires CONFIG_DIR is set in values
        self.environ._values["CONFIG_DIR"] = self.config_dir

        self.environ._set_network_variables()

        self.assertEqual(self.environ._values["TIMEOUT"], 5)
        self.assertEqual(self.environ._values["MPSPDZ_PORT"], mpspdz_port)
        self.assertEqual(self.environ._values["MPSPDZ_IP"], mpspdz_ip)
        self.assertEqual(self.environ._values["MPSPDZ_CERTIFICATE_KEY"], os.path.join(self.config_dir, private_key))
        self.assertEqual(self.environ._values["MPSPDZ_CERTIFICATE_PEM"], os.path.join(self.config_dir, public_key))
        self.assertEqual(self.environ._values["ALLOW_DEFAULT_BIPRIMES"], allow_default_biprimes)

    def test_environ_09_getters_and_setters(self):
        with self.assertRaises(FedbiomedEnvironError):
            self.environ["This_is_an_unknown_key"]

        with self.assertRaises(FedbiomedEnvironError):
            self.environ["This_is_an_unknown_newkey"] = None

        self.environ["This_is_a_new_key"] = 123
        self.assertEqual(self.environ["This_is_a_new_key"], 123)


    @patch("os.makedirs")
    @patch("os.path.isdir")
    def test_10_config_file_version(self,
                                    mock_is_dir,
                                    mock_mkdir):

        mock_is_dir.return_value = False
        self.environ._values["CONFIG_FILE"] = 'test/config/file.ini'

        # Test base case: the version in the config file exactly matches the version in the runtime
        with patch.object(self.environ, 'from_config', side_effect=['99.99']):
            self.environ.check_and_set_config_file_version('99.99')
            self.assertEqual(self.environ._values["CONFIG_FILE_VERSION"], '99.99')

        # Test base case 2: the version in the config file is missing
        with patch.object(self.environ, 'from_config', side_effect=FedbiomedEnvironError):
            self.environ.check_and_set_config_file_version(fedbiomed.common.utils.__default_version__)
            self.assertEqual(self.environ._values["CONFIG_FILE_VERSION"], fedbiomed.common.utils.__default_version__)

        # Test error case: when the version is not compatible
        self.mock_versions_log.reset_mock()
        with patch.object(self.environ, 'from_config', side_effect=['1.0']):
            with self.assertRaises(FedbiomedVersionError):
                self.environ.check_and_set_config_file_version('2.0')
                self.assertEqual(self.mock_versions_log.critical.call_count, 1)

        # Test warning case: when the version is not the same, but compatible
        self.mock_versions_log.reset_mock()
        with patch.object(self.environ, 'from_config', side_effect=['1.0']):
            self.environ.check_and_set_config_file_version('1.5')
            self.assertEqual(self.mock_versions_log.warning.call_count, 1)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()

import configparser

import unittest
import tempfile
import os
import shutil

from unittest.mock import patch, MagicMock
from fedbiomed.common.utils._config_utils import get_fedbiomed_root, get_component_config, \
    get_component_certificate_from_config, get_all_existing_config_files, get_all_existing_certificates, \
    get_existing_component_db_names
from fedbiomed.common.exceptions import FedbiomedError


class TestCommonConfigUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_01_common_config_utils_get_fedbiomed_root(self):
        """Test fedbiomed get root """
        root = get_fedbiomed_root()
        self.assertTrue("/fedbiomed" in root)

    @patch("fedbiomed.common.utils._config_utils.configparser")
    def test_02_common_config_utils_get_component_config(self,
                                                         mock_configparser):
        config_mock = MagicMock()
        mock_configparser.ConfigParser.return_value = config_mock

        # Get component config
        get_component_config("dummy/path")
        config_mock.read.assert_called_once_with("dummy/path")

        config_mock.read.side_effect = Exception

        with self.assertRaises(FedbiomedError):
            get_component_config("dummy/path")

    @patch("fedbiomed.common.utils._config_utils.get_component_config")
    @patch("fedbiomed.common.utils._config_utils.os.path.isfile")
    def test_03_common_config_utils_get_component_certificate_from_config(self,
                                                                          mock_is_file,
                                                                          mock_get_component_config):
        cfg = configparser.ConfigParser()

        cfg["default"] = {
            "id": "node-id",
            "component": "NODE"
        }

        cfg["mpspdz"] = {
            "mpspdz_ip": "localhost",
            "mpspdz_port": 1234,
            "public_key": "path/to/certificate"
        }

        mock_get_component_config.return_value = cfg
        mock_is_file.return_value = False

        with self.assertRaises(FedbiomedError):
            get_component_certificate_from_config("dummy/path/to/config")

        mock_is_file.return_value = True
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = Exception
            with self.assertRaises(FedbiomedError):
                get_component_certificate_from_config("dummy/path/to/config")

            mock_open.side_effect = None
            mock_open.return_value.__enter__.return_value.read.return_value = "test-certificate"
            cert = get_component_certificate_from_config("dummy/path/to/config")

            self.assertDictEqual(cert, {'certificate': 'test-certificate',
                                        'component': 'NODE',
                                        'ip': 'localhost',
                                        'party_id': 'node-id',
                                        'port': '1234'}
                                 )

    @patch("fedbiomed.common.utils._config_utils.get_fedbiomed_root")
    def test_03_common_config_utils_get_all_existing_config_files(self,
                                                                  mock_fedbiomed_root):
        test_dir = tempfile.mkdtemp()
        os.mkdir(os.path.join(test_dir, "etc"))

        file_ = os.path.join(test_dir, "etc", "test-test-config-util.ini")
        with open(file_, "w") as file:
            file.write("Hello world")
            file.close()

        mock_fedbiomed_root.return_value = test_dir
        files = get_all_existing_config_files()

        self.assertListEqual(files, [file_])

        shutil.rmtree(test_dir)

    @patch("fedbiomed.common.utils._config_utils.get_all_existing_config_files")
    @patch("fedbiomed.common.utils._config_utils.get_component_certificate_from_config")
    def test_03_common_config_utils_get_all_existing_certificates(self,
                                                                  mock_get_component_certificate,
                                                                  mock_get_all_existing_config_files):
        certificates = [
            {
                "party_id": 'node-1',
                "certificate": "test-certificate-1",
                "ip": "localhost",
                "port": 1234,
                "component": "NODE"
            },
            {
                "party_id": 'node-1',
                "certificate": "test-certificate-1",
                "ip": "localhost",
                "port": 1234,
                "component": "NODE"
            }
        ]

        mock_get_component_certificate.side_effect = certificates
        mock_get_all_existing_config_files.return_value = ["test", "test"]
        certificates_ = get_all_existing_certificates()

        self.assertListEqual(certificates_, certificates)

    @patch("fedbiomed.common.utils._config_utils.get_all_existing_config_files")
    @patch("fedbiomed.common.utils._config_utils.get_component_config")
    def test_03_common_config_utils_get_existing_component_db_names(self,
                                                                    mock_get_component_config,
                                                                    mock_get_all_existing_config_files
                                                                    ):

        cfg_1 = configparser.ConfigParser()
        cfg_2 = configparser.ConfigParser()

        cfg_1["default"] = {"id": "node-1"}
        cfg_2["default"] = {"id": "node-2"}
        mock_get_all_existing_config_files.return_value = ["test", "test"]
        mock_get_component_config.side_effect = [cfg_1, cfg_2]

        result = get_existing_component_db_names()
        self.assertDictEqual(result, {"node-1": "db_node-1", "node-2": "db_node-2"})





if __name__ == "__main__":
    unittest.main()

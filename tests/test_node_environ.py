import sys

import os
import importlib
import inspect
import unittest
import configparser
from unittest.mock import patch

from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedEnvironError
from testsupport.fake_common_environ import Environ


class TestNodeEnviron(unittest.TestCase):

    def setUp(self) -> None:
        """Setup test for each test function"""
        self.patch_environ = patch("fedbiomed.common.environ.Environ", Environ)
        self.patch_setup_environ = patch("fedbiomed.common.environ.Environ.setup_environment")

        self.mock_environ = self.patch_environ.start()
        self.mock_setup_environ = self.patch_setup_environ.start()

        environ_module_dir = os.path.join(os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            ), "..", "fedbiomed", "node", "environ.py")

        self.env = importlib.machinery.SourceFileLoader(
            "environ_for_test", environ_module_dir)\
            .load_module()

        NodeEnviron = self.env.NodeEnviron
        self.mock_setup_environ.reset_mock()

        self.environ = NodeEnviron()

        self.environ._values = {**self.environ._values,
                                "CONFIG_DIR": "dummy/config/dir",
                                "VAR_DIR": "dummy/var/dir",
                                "ROOT_DIR": "dummy/root/dir"
                                }
        self.environ._cfg = configparser.ConfigParser()

    def tearDown(self) -> None:
        self.patch_environ.stop()
        self.patch_setup_environ.stop()

        pass

    def test_01_node_environ_init(self):
        """Tests initialization of NodeEnviron"""
        self.environ.setup_environment.assert_called_once()
        self.assertEqual(self.environ._values["COMPONENT_TYPE"], ComponentType.NODE)

    def test_02_node_environ_default_config_file(self):
        """Test default config method """

        config = self.environ.default_config_file()
        self.assertEqual(config, os.path.join("dummy/config/dir", "config_node.ini"))

    @patch("os.mkdir")
    def test_03_node_environ_set_component_specific_variables(self,
                                                              mock_mkdir):
        os.environ["NODE_ID"] = "node-1"
        os.environ["ALLOW_DEFAULT_TRAINING_PLANS"] = "True"
        os.environ["ENABLE_TRAINING_PLAN_APPROVAL"] = "True"

        self.environ.from_config.side_effect = [None, None, None, "SHA256"]
        self.environ._set_component_specific_variables()

        self.assertEqual(self.environ._values["MESSAGES_QUEUE_DIR"],
                         os.path.join("dummy/var/dir", "queue_manager_node-1"))
        self.assertEqual(self.environ._values["DB_PATH"], os.path.join("dummy/var/dir", "db_node-1.json"))
        self.assertEqual(self.environ._values["TRAINING_PLANS_DIR"], os.path.join("dummy/var/dir",
                                                                                  "training_plans_node-1"))

        self.environ.from_config.side_effect = None
        self.environ.from_config.side_effect = [None, None, None, "SHA256BLABLA"]
        with self.assertRaises(FedbiomedEnvironError):
            self.environ._set_component_specific_variables()

        self.environ.from_config.side_effect = None
        self.environ.from_config.side_effect = [None, False, False, "SHA256"]
        os.environ["ALLOW_DEFAULT_TRAINING_PLANS"] = "True"
        os.environ["ENABLE_TRAINING_PLAN_APPROVAL"] = "True"
        self.environ._set_component_specific_variables()

        self.assertTrue(self.environ._values['ALLOW_DEFAULT_TRAINING_PLANS'], "os.getenv did not overwrite the value")
        self.assertTrue(self.environ._values['TRAINING_PLAN_APPROVAL'], "os.getenv did not overwrite the value")

    def test_04_node_environ_set_component_specific_config_parameters(self):
        os.environ["NODE_ID"] = "node-1"
        os.environ["ALLOW_DEFAULT_TRAINING_PLANS"] = "True"
        os.environ["ENABLE_TRAINING_PLAN_APPROVAL"] = "True"

        self.environ._get_uploads_url.return_value = "localhost"

        self.environ._set_component_specific_config_parameters()

        self.assertEqual(self.environ._cfg["default"], {
            'id': 'node-1',
            'component': "NODE",
            'uploads_url': "localhost"
        })

        self.assertEqual(self.environ._cfg["security"], {
            'hashing_algorithm': "SHA256",
            'allow_default_training_plans': "True",
            'training_plan_approval': "True"
        })

    @patch("fedbiomed.common.logger.logger.info")
    @patch("os.mkdir")
    def test_05_node_environ_info(self, mock_mkdir, mock_logger_info):
        os.environ["NODE_ID"] = "node-1"
        os.environ["ALLOW_DEFAULT_TRAINING_PLANS"] = "True"
        os.environ["ENABLE_TRAINING_PLAN_APPROVAL"] = "True"

        self.environ.from_config.side_effect = [None, None, None, "SHA256"]
        self.environ._set_component_specific_variables()

        self.environ.info()
        self.assertEqual(mock_logger_info.call_count, 3)


if __name__ == "__main__":
    unittest.main()

import os
import importlib
import unittest
import inspect
import configparser
from unittest.mock import patch

from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedEnvironError
from testsupport.fake_common_environ import Environ


class TestResearcherEnviron(unittest.TestCase):

    def setUp(self) -> None:
        """Setup test for each test function"""

        self.patch_environ = patch("fedbiomed.common.environ.Environ", Environ)
        self.patch_setup_environ = patch("fedbiomed.common.environ.Environ.setup_environment")

        self.mock_environ = self.patch_environ.start()
        self.mock_setup_environ = self.patch_setup_environ.start()

        environ_module_dir = os.path.join(os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe()))
            ), "..", "fedbiomed", "researcher", "environ.py")

        self.env = importlib.machinery.SourceFileLoader(
            "environ_for_test_node", environ_module_dir)\
            .load_module()

        ResearcherEnviron = self.env.ResearcherEnviron
        self.mock_setup_environ.reset_mock()

        self.environ = ResearcherEnviron()
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

    def test_01_researcher_environ_init(self):
        """Tests initialization of ResearcherEnviron"""
        self.environ.setup_environment.assert_called_once()
        self.assertEqual(self.environ._values["COMPONENT_TYPE"], ComponentType.RESEARCHER)

    def test_02_researcher_environ_default_config_file(self):
        """Test default config method """

        config = self.environ.default_config_file()
        self.assertEqual(config, os.path.join("dummy/config/dir", "config_researcher.ini"))

    @patch("os.makedirs")
    @patch("os.path.isdir")
    def test_03_researcher_environ_set_component_specific_variables(self,
                                                                    mock_is_dir,
                                                                    mock_mkdir):
        """Tests setting variables for researcher environ"""
        self.environ.from_config.side_effect = None
        os.environ["RESEARCHER_ID"] = "researcher-1"

        self.environ.from_config.side_effect = [None, None, None, "SHA256"]
        mock_is_dir.return_value = False

        self.environ._set_component_specific_variables()

        self.assertEqual(self.environ._values["ID"], "researcher-1")
        self.assertEqual(self.environ._values["EXPERIMENTS_DIR"],
                         os.path.join("dummy/var/dir", "experiments"))
        self.assertEqual(self.environ._values["TENSORBOARD_RESULTS_DIR"],
                         os.path.join("dummy/root/dir", "runs"))
        self.assertEqual(self.environ._values["MESSAGES_QUEUE_DIR"],
                         os.path.join("dummy/var/dir", "queue_messages"))
        self.assertEqual(self.environ._values["DB_PATH"],
                         os.path.join("dummy/var/dir", "db_researcher-1.json"))

        mock_mkdir.side_effect = [FileExistsError, OSError]
        with self.assertRaises(FedbiomedEnvironError):
            self.environ._set_component_specific_variables()

        with self.assertRaises(FedbiomedEnvironError):
            self.environ._set_component_specific_variables()

    def test_04_researcher_environ_set_component_specific_config_parameters(self):
        """Tests setting configuration file parameters"""
        os.environ["RESEARCHER_ID"] = "researcher-1"

        self.environ._get_uploads_url.return_value = "localhost"
        self.environ._set_component_specific_config_parameters()

        self.assertEqual(self.environ._cfg["default"], {
            'id': 'researcher-1',
            'component': "RESEARCHER",
            'uploads_url': "localhost"
        })

    @patch("fedbiomed.common.logger.logger.info")
    def test_05_researcher_environ_info(self, mock_logger_info):

        self.environ._values["COMPONENT_TYPE"] = ComponentType.RESEARCHER
        self.environ.info()
        self.assertEqual(mock_logger_info.call_count, 2)


if __name__ == "__main__":
    unittest.main()
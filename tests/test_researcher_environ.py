import os, io
import importlib
import unittest
import inspect
import configparser
import logging
from unittest.mock import patch

import fedbiomed
from fedbiomed.researcher import __config_version__
from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedEnvironError, FedbiomedVersionError
from testsupport.fake_common_environ import Environ


class TestResearcherEnviron(unittest.TestCase):

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

    def test_01_researcher_environ_init(self):
        """Tests initialization of ResearcherEnviron"""
        self.environ.setup_environment.assert_called_once()
        self.assertEqual(self.environ._values["COMPONENT_TYPE"], ComponentType.RESEARCHER)

    def test_02_researcher_environ_default_config_file(self):
        """Test default config method """

        config = self.environ.default_config_file()
        self.assertEqual(config, os.path.join("dummy/config/dir", "config_researcher.ini"))

    @patch("fedbiomed.common.utils.raise_for_version_compatibility")
    @patch("os.makedirs")
    @patch("os.path.isdir")
    def test_03_researcher_environ_set_component_specific_variables(self,
                                                                    mock_is_dir,
                                                                    mock_mkdir,
                                                                    mock_compatibility):
        """Tests setting variables for researcher environ"""
        os.environ["RESEARCHER_ID"] = "researcher-1"
        # Test base case: no exceptions are raised and we assert that the values are read correctly
        self.environ.from_config.side_effect = ['1.0', None]
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

        # Test case where tensorboard/experiment dir already exist
        mock_mkdir.side_effect = [FileExistsError]
        self.environ.from_config.side_effect = ['1.0', None]
        with self.assertRaises(FedbiomedEnvironError):
            self.environ._set_component_specific_variables()

        # Test general OSError while creating the directories
        mock_mkdir.side_effect = [OSError]
        self.environ.from_config.side_effect = ['1.0', None]
        with self.assertRaises(FedbiomedEnvironError):
            self.environ._set_component_specific_variables()

    def test_04_researcher_environ_set_component_specific_config_parameters(self):
        """Tests setting configuration file parameters"""
        os.environ["RESEARCHER_ID"] = "researcher-1"

        self.environ._get_uploads_url.return_value = "localhost"
        self.environ._set_component_specific_config_parameters()

        self.assertDictEqual(dict(self.environ._cfg["default"]), {
            'id': 'researcher-1',
            'component': "RESEARCHER",
            'uploads_url': "localhost",
            'version': str(__config_version__)
        })

    @patch("fedbiomed.common.logger.logger.info")
    def test_05_researcher_environ_info(self, mock_logger_info):

        self.environ._values["COMPONENT_TYPE"] = ComponentType.RESEARCHER
        self.environ.info()
        self.assertEqual(mock_logger_info.call_count, 2)

    @patch("os.makedirs")
    @patch("os.path.isdir")
    def test_06_researcher_version(self,
                                   mock_is_dir,
                                   mock_mkdir):

        os.environ["RESEARCHER_ID"] = "researcher-1"
        mock_is_dir.return_value = False

        # Test base case: the version in the config file exactly matches the version in the runtime
        self.environ.from_config.side_effect = [__config_version__, None]
        self.environ._set_component_specific_variables()
        self.assertEqual(self.environ._values["CONFIG_FILE_VERSION"], __config_version__)

        # Test base case 2: the version in the config file is missing
        # We assign the current version (i.e. __config_version__) to the default version value in order to
        # avoid raising an error when checking for version compatibility. The purpose here is to check whether
        # we correctly assign the default version value when the field is missing from the config file.
        with patch.object(fedbiomed.common.utils, '__default_version__', __config_version__):
            self.environ.from_config.side_effect = [FedbiomedEnvironError, None]
            self.environ._set_component_specific_variables()
            self.assertEqual(self.environ._values["CONFIG_FILE_VERSION"], __config_version__)

        # Test error case: when the version is not compatible
        self.logging_output.truncate(0)  # clear the logging buffer for simplicity
        self.environ.from_config.side_effect = ['0.1', None]
        with self.assertRaises(FedbiomedVersionError):
            self.environ._set_component_specific_variables()
        self.assertEqual(self.logging_output.getvalue().split('-')[2][-19:], 'fedbiomed CRITICAL ')

        # Test warning case: when the version is not the same, but compatible
        self.logging_output.truncate(0)  # clear the logging buffer for simplicity
        new_version = ".".join([str(__config_version__.major), str(__config_version__.minor + 4)])
        self.environ.from_config.side_effect = [new_version, None]
        self.environ._set_component_specific_variables()
        self.assertEqual(self.logging_output.getvalue().split('-')[2][-18:], 'fedbiomed WARNING ')








if __name__ == "__main__":
    unittest.main()
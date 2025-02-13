import os
import configparser
from fedbiomed.node.config import NodeConfig
import shutil

from fedbiomed.common.utils import ROOT_DIR

cfg = configparser.ConfigParser()


class Config(dict):


    node_config: NodeConfig

    def __init__(self):
        """
            Config class to update configuration for Flask
        """
        self.configuration = {}
        # Updates self.configuration
        self.generate_config()

    def __delitem__(self, key):
        """Deletes given key from configuration"""

        del self.configuration[key]

    def __getitem__(self, item):
        """Gets item from self.configuration """

        return self.configuration[item]

    def generate_config(self):
        """
            This methods gets ENV variable from `os` and
            generates configuration object

            returns (dict): Dict of configurati0n

        """

        # Configuration of Flask APP to be able to access Fed-BioMed node information
        self.configuration['NODE_FEDBIOMED_ROOT'] = os.getenv(
            'FBM_NODE_COMPONENT_ROOT', os.getcwd()
        )
        conf = os.path.join(self.configuration['NODE_FEDBIOMED_ROOT'], 'etc', 'config_gui.ini')
        if not os.path.isfile(conf):
            default_config = os.path.join(ROOT_DIR, 'fedbiomed_gui', 'config_gui.ini')
            shutil.copy(default_config, conf)

        # Config file that is located in ${FEDBIOMED_DIR}/gui directory
        cfg.read(conf)


        # Data path ----------------------------------------------------------------
        data_path = os.getenv('DATA_PATH', cfg.get('server', 'DATA_PATH', fallback='data'))

        if data_path.startswith('/'):
            assert os.path.isdir(
                data_path), f'Data folder path "{data_path}" does not exist or it is not a directory.'
        else:
            data_path = os.path.join(self.configuration['NODE_FEDBIOMED_ROOT'], data_path)
            assert os.path.isdir(data_path), f'{data_path} has not been found in Fed-BioMed root directory or ' \
                                             f'it is not a directory. Please make sure that the folder is exist.'

        # Data path where datafiles are stored. Since node and gui works in same machine without docker,
        # path for writing and reading will be same for saving into database
        self.configuration['DATA_PATH_RW'] = data_path
        self.configuration['DATA_PATH_SAVE'] = data_path

        self.configuration['DEFAULT_ADMIN_CREDENTIAL'] = {'email': cfg.get('init_admin', 'email'),
                                                          'password': cfg.get('init_admin', 'password')}

        # Get name of the config file default is "config_node.ini"
        self.configuration['NODE_CONFIG_FILE'] = os.getenv('NODE_CONFIG_FILE',
                                                           "config.ini")

        # Node config file -----------------------------------------------------
        self.node_config = NodeConfig(root=self.configuration["NODE_FEDBIOMED_ROOT"])
        node_id = self.node_config.get('default', 'id')
        self.configuration['ID'] = node_id

        # Set DB_PATH based on given node id
        self.configuration['NODE_DB_PATH'] = os.path.join(
            self.configuration["NODE_FEDBIOMED_ROOT"], 'etc', self.node_config.get('default', 'db')
        )
        # Set GUI_PATH based on given node id
        self.configuration['GUI_DB_PATH'] = os.path.join(
            self.configuration["NODE_FEDBIOMED_ROOT"],
            'var',
            'gui_db_' + self.configuration['ID'] + '.json'
        )

        # Enable debug mode
        self.configuration['DEBUG'] = os.getenv('FBM_DEBUG', 'True').lower() in \
                                      ('true', 1, True, 'yes')

        # Serve  configurations PORT and IP
        self.configuration['PORT'] = os.getenv(
            'FBM_GUI_PORT', cfg.get('server', 'PORT', fallback=8484)
        )
        self.configuration['HOST'] = os.getenv(
            'FBM_GUI_HOST', cfg.get('server', 'HOST', fallback='localhost')
        )

        # Log information for setting up a node connection
        print(f'INFO: Fed-BioMed Node root dir has been set as '
              f'{self.configuration["NODE_FEDBIOMED_ROOT"]} \n')

        print(f'INFO: Services are going to be configured for the node '
              f'{self.configuration["ID"]} \n')

        return self.configuration


config = Config()

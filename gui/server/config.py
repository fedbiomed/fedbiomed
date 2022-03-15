import os
import sys
from utils import get_node_id


class Config:

    def __init__(self):
        """
            Config class to update configuration for Flask
        """
        self.configuration = {}

    @property
    def config(self):
        return self.configuration

    def generate_config(self):
        """
            This methods gets ENV variable from `os` and
            generates configuration object

            returns (dict): Dict of configurati0n

        """
        
        # Configuration of Flask APP to be able to access Fed-BioMed node information
        self.configuration['NODE_FEDBIOMED_ROOT'] = os.getenv('FEDBIOMED_DIR', '/fedbiomed')
        data_path = os.getenv('DATA_PATH')
        if not data_path:
            data_path = '/data'
        else:
            if data_path.startswith('/'):
                assert os.path.isdir(
                    data_path), f'Given absolute "{data_path}" does not exist or it is not a directory.'
            else:
                data_path = os.path.join(self.configuration['NODE_FEDBIOMED_ROOT'], data_path)
                assert os.path.isdir(data_path), f'{data_path} has not been found in Fed-BioMed root directory or ' \
                                                 f'it is not a directory. Please make sure that the folder is exist.'
        # Data path where datafiles are stored. Since node and gui
        # works in same machine without docker, path for writing and reading
        # will be same for saving into database
        self.configuration['DATA_PATH_RW'] = data_path
        self.configuration['DATA_PATH_SAVE'] = data_path

        # Get name of the config file default is "config_node.ini"
        self.configuration['NODE_CONFIG_FILE'] = os.getenv('NODE_CONFIG_FILE',
                                                           "config_node.ini")

        # Exact configuration file path
        self.configuration['NODE_CONFIG_FILE_PATH'] = \
            os.path.join(self.configuration["NODE_FEDBIOMED_ROOT"],
                         'etc',
                         self.configuration['NODE_CONFIG_FILE'])

        # Append Fed-BioMed root dir as a python path
        sys.path.append(self.configuration['NODE_FEDBIOMED_ROOT'])

        # Set config file path to make `fedbiomed.common.environ` to parse
        # correct config file
        os.environ["CONFIG_FILE"] = self.configuration['NODE_CONFIG_FILE_PATH']

        node_id = get_node_id(self.configuration['NODE_CONFIG_FILE_PATH'])
        # Set node NODE_DI
        self.configuration['NODE_ID'] = node_id

        # Set DB_PATH based on given node id
        self.configuration['NODE_DB_PATH'] = \
            os.path.join(self.configuration["NODE_FEDBIOMED_ROOT"],
                         'var',
                         'db_' + self.configuration['NODE_ID'] + '.json')

        # Enable debug mode
        self.configuration['DEBUG'] = os.getenv('DEBUG', 'True').lower() in \
                                      ('true', 1, True, 'yes')

        # TODO: Let users decide which port they would like to use
        # Serve  configurations PORT and IP
        self.configuration['PORT'] = os.getenv('PORT', 8484)
        self.configuration['HOST'] = os.getenv('HOST', 'localhost')

        # Log information for setting up a node connection
        print(f'INFO: Fed-BioMed Node root dir has been set as '
              f'{self.configuration["NODE_FEDBIOMED_ROOT"]} \n')

        print(f'INFO: Fed-BioMed  Node config file is '
              f'{self.configuration["NODE_CONFIG_FILE"]} \n')

        print(f'INFO: Services are going to be configured for the node '
              f'{self.configuration["NODE_ID"]} \n')

        return self.configuration

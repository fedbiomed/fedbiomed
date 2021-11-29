import os
import configparser



def get_node_id(config_file: str):

    """ This method parse given config file and returns node_id
        specified in the node config file.

    Args: 

        config_file     (str): Path for config file of the node that 
                        GUI services will running for
    """

    cfg = configparser.ConfigParser()
    if os.path.isfile(config_file):
        cfg.read(config_file)
    else:
        raise Exception(f'Config file does not exist, con not start flask {config_file}')

    # Get node id from config file 
    node_id = cfg.get('default', 'node_id')

    return node_id
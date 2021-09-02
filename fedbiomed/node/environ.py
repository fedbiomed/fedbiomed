import configparser
import os
import uuid

from six.moves import urllib
from typing import Optional

try:
    defined_node_env
except NameError:
    defined_node_env = False

# python imports should handle this,  but avoid eventual weird cases

# FIXME: what if ` defined_node_env` is set to True ? 
if not defined_node_env:
    def init_client_config(client_id: Optional[str] = None):
        """ This method: reads the config file if exists, otherwise it creates
                it with the NODE config params



        Args:
            client_id (str, optional): client id. Defaults to None.

        Returns:
            cfg: content of the config file
        """
        cfg = configparser.ConfigParser()

        if os.path.isfile(CONFIG_FILE):
            cfg.read(CONFIG_FILE)
            return cfg

        # Create client ID
        client_id = os.getenv('CLIENT_ID', 'client_' + str(uuid.uuid4()))

        # create network csracteristics from environment or config file
        uploads_ip = os.getenv('UPLOADS_IP')
        uploads_url = "http://localhost:8844/upload/"

        if uploads_ip:
            uploads_url = "http://" + uploads_ip + ":8844/upload/"

        # is positionned UPLOADS_URL is stronger than the one deduced from
        # UPLOADS_IP
        uploads_url = os.getenv('UPLOADS_URL', uploads_url)

        cfg['default'] = {
            'client_id': client_id,
            'uploads_url': uploads_url,
        }

        mqtt_broker = os.getenv('MQTT_BROKER', 'localhost')
        mqtt_broker_port = int(os.getenv('MQTT_BROKER_PORT', 80))

        cfg['mqtt'] = {
            'broker_ip': mqtt_broker,
            'port': mqtt_broker_port,
            'keep_alive': 60
        }

        with open(CONFIG_FILE, 'w') as f:
            cfg.write(f)

        return cfg

    ROOT_DIR = os.path.abspath(
                               os.path.join(
                                            os.path.dirname(
                                                            os.path.abspath(__file__)
                                                            ),
                                            '../..')
                               )

    CONFIG_DIR = os.path.join(ROOT_DIR, 'etc')
    VAR_DIR = os.path.join(ROOT_DIR, 'var')
    CACHE_DIR = os.path.join(VAR_DIR, 'cache')
    TMP_DIR = os.path.join(VAR_DIR, 'tmp')

    for dir in CONFIG_DIR, VAR_DIR, CACHE_DIR, TMP_DIR:
        if not os.path.isdir(dir):
            try:
                os.makedirs(dir)
            except FileExistsError:
                print("[ ERROR ] path exists but is not a directory", dir)

    if os.getenv('CONFIG_FILE'):
        CONFIG_FILE = os.getenv('CONFIG_FILE')
        if not os.path.isabs(CONFIG_FILE):
            CONFIG_FILE = os.path.join(
                                       CONFIG_DIR,
                                       os.getenv('CONFIG_FILE'))
    else:
        CONFIG_FILE = os.path.join(CONFIG_DIR, 'config_node.ini')

    cfg = init_client_config()
    CLIENT_ID = os.getenv('CLIENT_ID', cfg.get('default', 'client_id'))

    MESSAGES_QUEUE_DIR = os.path.join(VAR_DIR, f'queue_manager_{CLIENT_ID}')
    DB_PATH = os.path.join(VAR_DIR, f'db_{CLIENT_ID}.json')

    MQTT_BROKER = os.getenv('MQTT_BROKER', cfg.get('mqtt', 'broker_ip'))
    MQTT_BROKER_PORT = int(os.getenv('MQTT_BROKER_PORT',
                                     cfg.get('mqtt', 'port')))

    UPLOADS_URL = cfg.get('default', 'uploads_url')
    uploads_ip = os.getenv('UPLOADS_IP')
    if uploads_ip:
        UPLOADS_URL = "http://" + uploads_ip + ":8844/upload/"
    UPLOADS_URL = os.getenv('UPLOADS_URL', UPLOADS_URL)

    # trailing slash is needed for repo url
    if not UPLOADS_URL.endswith('/'):
        UPLOADS_URL += '/'


    # ========= PATCH MNIST Bug torchvision 0.9.0 ===================
    # https://github.com/pytorch/vision/issues/1938

    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-agent', 'Python-urllib/3.7'),
        ('Accept',
         'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'),
        ('Accept-Language', 'en-US,en;q=0.9'),
        ('Accept-Encoding', 'gzip, deflate, br')
    ]
    urllib.request.install_opener(opener)

    defined_node_env = True

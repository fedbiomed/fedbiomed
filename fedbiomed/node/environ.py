import os
import uuid

from six.moves import urllib
import configparser

try:
    defined_node_env
except NameError:
    defined_node_env = False

# python imports should handle this,  but avoid eventual weird cases
if not defined_node_env:
    def init_client_config(client_id=None):
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

        cfg['default'] = {
            'client_id': client_id,
            'uploads_url': 'http://localhost:8844/upload/',
        }

        cfg['mqtt'] = {
            'broker_url': 'localhost',
            'port': 80,
            'keep_alive': 60
        }

        with open(CONFIG_FILE, 'w') as f:
            cfg.write(f)

        return cfg


    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

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


    if os.getenv('CONFIG_FILE') :
        CONFIG_FILE = os.getenv('CONFIG_FILE')
        if not CONFIG_FILE.startswith("/") :
            CONFIG_FILE = os.path.join(CONFIG_DIR,os.getenv('CONFIG_FILE'))
    else:
        CONFIG_FILE = os.path.join(CONFIG_DIR, 'config_node.ini')

    cfg = init_client_config()
    CLIENT_ID = os.getenv('CLIENT_ID', cfg.get('default', 'client_id'))

    MESSAGES_QUEUE_DIR = os.path.join(VAR_DIR, f'queue_manager_{CLIENT_ID}')
    DB_PATH = os.path.join(VAR_DIR, f'db_{CLIENT_ID}.json')

    MQTT_BROKER = os.getenv('MQTT_BROKER', cfg.get('mqtt', 'broker_url'))
    MQTT_BROKER_PORT = int(os.getenv('MQTT_BROKER_PORT', cfg.get('mqtt', 'port')))

    UPLOADS_URL = os.getenv('UPLOADS_URL', cfg.get('default', 'uploads_url'))
    if not UPLOADS_URL.endswith('/') :
        UPLOADS_URL += '/'


    # ========= PATCH MNIST Bug torchvision 0.9.0 ===================
    # https://github.com/pytorch/vision/issues/1938

    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-agent', 'Python-urllib/3.7'),
        ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'),
        ('Accept-Language', 'en-US,en;q=0.9'),
        ('Accept-Encoding', 'gzip, deflate, br')
    ]
    urllib.request.install_opener(opener)

    defined_node_env = True

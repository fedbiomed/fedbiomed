import os
import uuid
import configparser

try:
    defined_researcher_env
except NameError:
    defined_researcher_env = False

# python imports should handle this, but avoid eventual weird cases
if not defined_researcher_env:
    def init_researcher_config(researcher_id=None):
        """ This method: reads the config file if exists, otherwise it creates
                it with the researcher config params



        Args:
            researcher_id (str, optional): researcher id. Defaults to None.

        Returns:
            cfg: content of the config file
        """

        cfg = configparser.ConfigParser()

        if os.path.isfile(CONFIG_FILE):
            cfg.read(CONFIG_FILE)
            return cfg

        researcher_id = os.getenv('RESEARCHER_ID', 'researcher_' + str(uuid.uuid4()))

        cfg['default'] = {
            'uploads_url': 'http://localhost:8844/upload/',
            'researcher_id': researcher_id,
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
    TENSORBOARD_RESULTS_DIR = os.path.join(ROOT_DIR, 'runs')

    for dir in CONFIG_DIR, VAR_DIR, CACHE_DIR, TMP_DIR, TENSORBOARD_RESULTS_DIR:
        if not os.path.isdir(dir):
            try:
                os.makedirs(dir)
            except FileExistsError:
                print("[ ERROR ] path exists but is not a directory", dir)

    MESSAGES_QUEUE_DIR = os.path.join(VAR_DIR, 'queue_messages')

    if os.getenv('CONFIG_FILE') :
        CONFIG_FILE = os.getenv('CONFIG_FILE')
        if not CONFIG_FILE.startswith("/") :
            CONFIG_FILE = os.path.join(CONFIG_DIR,os.getenv('CONFIG_FILE'))
    else:
        CONFIG_FILE = os.path.join(CONFIG_DIR, 'config_researcher.ini')

    cfg = init_researcher_config()
    RESEARCHER_ID = os.getenv('RESEARCHER_ID', cfg.get('default', 'researcher_id'))

    MQTT_BROKER = os.getenv('MQTT_BROKER', cfg.get('mqtt', 'broker_url'))
    MQTT_BROKER_PORT = int(os.getenv('MQTT_BROKER_PORT', cfg.get('mqtt', 'port')))

    UPLOADS_URL = os.getenv('UPLOADS_URL', cfg.get('default', 'uploads_url'))
    if not UPLOADS_URL.endswith('/') :
        UPLOADS_URL += '/'

    TIMEOUT = 5

    defined_researcher_env = True

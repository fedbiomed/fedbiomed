from unittest.mock import Mock

import os
import sys

# use mock to impersonnate the environ.py import from a node
# must be done before the corresponding import, in this
# file or from another imported file
fake_node_env = Mock()

fake_node_env.ROOT_DIR           = "/tmp"
fake_node_env.CONFIG_DIR         = "/tmp/etc"
fake_node_env.VAR_DIR            = "/tmp/var"
fake_node_env.CACHE_DIR          = "/tmp/var/cache"
fake_node_env.TMP_DIR            = "/tmp/var/tmp"
fake_node_env.MESSAGES_QUEUE_DIR = "/tmp/var/queue_messages"
fake_node_env.CLIENT_ID          = "mock_node_XXX"
fake_node_env.DB_PATH            = '/tmp/var/db_client_mock_node_XXX.json'
fake_node_env.MQTT_BROKER        = "localhost"
fake_node_env.MQTT_BROKER_PORT   = 9999
fake_node_env.UPLOADS_URL        = "http://localhost:8888/upload/"

os.makedirs(fake_node_env.ROOT_DIR          , exist_ok=True)
os.makedirs(fake_node_env.CONFIG_DIR        , exist_ok=True)
os.makedirs(fake_node_env.VAR_DIR           , exist_ok=True)
os.makedirs(fake_node_env.CACHE_DIR         , exist_ok=True)
os.makedirs(fake_node_env.TMP_DIR           , exist_ok=True)
os.makedirs(fake_node_env.MESSAGES_QUEUE_DIR, exist_ok=True)

sys.modules['fedbiomed.node.environ'] = fake_node_env

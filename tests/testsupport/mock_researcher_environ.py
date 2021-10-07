# use mock to impersonnate the environ.py import from a researcher
# must be imported before the corresponding import, in this
# file or from another imported file

from unittest.mock import Mock

import os
import sys

fake_researcher_env = Mock()

fake_researcher_env.ROOT_DIR           = "/tmp"
fake_researcher_env.CONFIG_DIR         = "/tmp/etc"
fake_researcher_env.VAR_DIR            = "/tmp/var"
fake_researcher_env.CACHE_DIR          = "/tmp/var/cache"
fake_researcher_env.TMP_DIR            = "/tmp/var/tmp"
fake_researcher_env.MESSAGES_QUEUE_DIR = "/tmp/var/queue_messages"
fake_researcher_env.RESEARCHER_ID      = "researcher_XXX"
fake_researcher_env.MQTT_BROKER        = "localhost"
fake_researcher_env.MQTT_BROKER_PORT   = 1883
fake_researcher_env.UPLOADS_URL        = "http://localhost:8888/upload/"
fake_researcher_env.TIMEOUT            = 10

# TODO: create random directory paths like for test_taskqueue.py
os.makedirs(fake_researcher_env.ROOT_DIR          , exist_ok=True)
os.makedirs(fake_researcher_env.CONFIG_DIR        , exist_ok=True)
os.makedirs(fake_researcher_env.VAR_DIR           , exist_ok=True)
os.makedirs(fake_researcher_env.CACHE_DIR         , exist_ok=True)
os.makedirs(fake_researcher_env.TMP_DIR           , exist_ok=True)
os.makedirs(fake_researcher_env.MESSAGES_QUEUE_DIR, exist_ok=True)

sys.modules['fedbiomed.researcher.environ'] = fake_researcher_env

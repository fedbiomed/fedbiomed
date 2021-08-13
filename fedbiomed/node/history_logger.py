from typing import Union

from fedbiomed.common.message import NodeMessages
from fedbiomed.node.environ import CLIENT_ID


class HistoryLogger:
    def __init__(self, job_id, researcher_id, client):
        self.history = {}
        self.job_id = job_id
        self.researcher_id = researcher_id
        self.messaging = client

    def add_scalar(self, key: str, value: Union[int, float], iteration: int):

        try:
            self.history[key][iteration] = value
        except (KeyError, AttributeError):
            self.history[key] = {iteration: value}

        self.messaging.send_message(NodeMessages.reply_create({'client_id':CLIENT_ID, 'job_id':self.job_id,
            'researcher_id':self.researcher_id, 'key':value, 'iteration':iteration, "command": "add_scalar"}).get_dict())

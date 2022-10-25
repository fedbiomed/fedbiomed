from typing import Union

from testsupport.fake_responses import FakeResponses


class FakeRequests():
    def __init__(self):
        self.messages = []
        self.sequence = 0
        self.custom = {}

    def set_replies_custom_fields(self, custom: dict):
        """Custom value for a field of messages. Useful for testing some error cases.
        
        This does not exist in original `Requests` and is added for test purposes.
        """
        self.custom = custom

    def send_message(self, msg: dict, client: str = None, add_sequence: bool = False) -> \
            Union[int, None]:
        # always add sequence, whatever the `add_sequence`
        self.sequence += 1
        if msg['command'] in ['secagg', 'secagg-delete']:
            message = {
                'researcher_id': msg['researcher_id'],
                'secagg_id': msg['secagg_id'],
                'sequence': self.sequence,
                'success': True,
                'node_id': client,
                'msg': 'Fake request',
                'command': msg['command'],
            }
        else:
            message = {}

        for field in self.custom.keys():
            if field in message:
                message[field] = self.custom[field]
        self.messages.append(message)
        return self.sequence

    def get_responses(
            self,
            look_for_commands: list,
            timeout: float = None,
            only_successful: bool = True,
            while_responses: bool = True) -> FakeResponses:
        # return existing responses without delay, whatever the arguments
        messages = self.messages
        self.messages = []
        return FakeResponses(messages)

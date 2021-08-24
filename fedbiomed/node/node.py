
from json import decoder
from typing import Optional, Union, Dict, Any

from fedbiomed.common import json
from fedbiomed.common.tasks_queue import TasksQueue
from fedbiomed.common.messaging import Messaging,  MessagingType
from fedbiomed.common.message import NodeMessages
from fedbiomed.node.environ import CLIENT_ID, MESSAGES_QUEUE_DIR, TMP_DIR, MQTT_BROKER, MQTT_BROKER_PORT
from fedbiomed.node.history_logger import HistoryLogger
from fedbiomed.node.round import Round
from fedbiomed.node.data_manager import Data_manager

# should we keep this import (third party package)? There is a risk
# "validators" package will be no
import validators
# longer maintained in the future. Plus, there are no test coverage
# available on github page
# (here `validators` is only used for assertion errors)


class Node:

    def __init__(self, data_manager: Data_manager):

        self.tasks_queue = TasksQueue(MESSAGES_QUEUE_DIR, TMP_DIR)
        self.messaging = Messaging(self.on_message, MessagingType.NODE,
                                   CLIENT_ID, MQTT_BROKER, MQTT_BROKER_PORT)
        self.data_manager = data_manager
        self.rounds = []

    def add_task(self, task: dict):
        """This method allows to add a task to the queue

        Args:
            task (dict): is a Message object describing a training task
        """
        self.tasks_queue.add(task)

    def on_message(self, msg):
        # TODO: describe all exceptions defined in this method
        print('[CLIENT] Message received: ', msg)
        try:
            command = msg['command']
            request = NodeMessages.request_create(msg).get_dict()
            if command == 'train':
                self.add_task(request)
            elif command == 'ping':
                self.messaging.send_message(NodeMessages.reply_create({'success': True, 'client_id': CLIENT_ID,
                                                                       'researcher_id': msg['researcher_id'], 'command': 'ping'}).get_dict())
            elif command == 'search':
                # Look for databases corresponNotImplementedErrording with tags
                databases = self.data_manager.search_by_tags(msg['tags'])
                if len(databases) != 0:
                    # remove path from search to avoid privacy issues
                    for d in databases:
                        d.pop('path', None)

                    self.messaging.send_message(NodeMessages.reply_create({'success': True, "command": "search", 'client_id': CLIENT_ID, 'researcher_id': msg['researcher_id'],
                                                                           'databases': databases, 'count': len(databases)}).get_dict())
            else:
                raise NotImplementedError('Command not found')
        except decoder.JSONDecodeError:
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.messaging.send_message(NodeMessages.reply_create(
                {'success': False, 'command': "error", 'client_id': CLIENT_ID, 'researcher_id': resid, 'msg': "Not able to deserialize the message"}).get_dict())
        except NotImplementedError:
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.messaging.send_message(NodeMessages.reply_create(
                {'success': False, 'command': "error", 'client_id': CLIENT_ID, 'researcher_id': resid, 'msg': f"Command `{command}` is not implemented"}).get_dict())
        except KeyError:
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.messaging.send_message(NodeMessages.reply_create(
                {'success': False, 'command': "error", 'client_id': CLIENT_ID, 'researcher_id': resid, 'msg': "'command' property was not found"}).get_dict())
        except TypeError:  # Message was not serializable
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.messaging.send_message(NodeMessages.reply_create(
                {'success': False, 'command': "error", 'client_id': CLIENT_ID, 'researcher_id': resid, 'msg': 'Message was not serializable'}).get_dict())

    def parser_task(self, msg: Union[bytes, str, Dict[str, Any]]):
        """ This method parses a given task message to create a round instance

        Args:
            msg (Union[bytes, str, Dict[str, Any]]): serialized Message object
            to parse (or that have been parsed)
        """
        if isinstance(msg, str) or isinstance(msg, bytes):
            msg = json.deserialize_msg(msg)
        msg = NodeMessages.request_create(msg)
        logger = HistoryLogger(job_id=msg.get_param(
            'job_id'), researcher_id=msg.get_param('researcher_id'),
                               client=self.messaging)
        # Get arguments for the model and training
        model_kwargs = msg.get_param('model_args') or {}
        training_kwargs = msg.get_param('training_args') or {}
        model_url = msg.get_param('model_url')
        model_class = msg.get_param('model_class')
        params_url = msg.get_param('params_url')
        job_id = msg.get_param('job_id')
        researcher_id = msg.get_param('researcher_id')

        assert model_url is not None, 'URL for model on repository not found.'
        assert validators.url(
            model_url), 'URL for model on repository is not valid.'
        assert model_class is not None, 'classname for the model and training routine was not found in message.'
        assert isinstance(
            model_class, str), '`model_class` must be a string corresponding to the classname for the model and training routine in the repository'

        if CLIENT_ID in msg.get_param('training_data'):
            for dataset_id in msg.get_param('training_data')[CLIENT_ID]:
                alldata = self.data_manager.search_by_id(dataset_id)
                if len(alldata) != 1 or not 'path' in alldata[0].keys():
                    self.messaging.send_message(NodeMessages.reply_create({'success': False,
                                                                           'command': "error",
                                                                           'client_id': CLIENT_ID,
                                                                           'researcher_id': researcher_id,
                                                                           'msg': "Did not found proper data in local datasets"}).get_dict())
                else:
                    self.rounds = []
                    self.rounds.append(Round(model_kwargs, training_kwargs, alldata[0], model_url,
                                             model_class, params_url, job_id, researcher_id, logger))

    def task_manager(self):
        """ This method manages training tasks in the queue
        """

        while True:
            item = self.tasks_queue.get()

            try:
                print('[TASKS QUEUE] Item:', item)
                self.parser_task(item)
                # once task is out of queue, initiate training rounds
                for round in self.rounds:
                    msg = round.run_model_training()
                    self.messaging.send_message(msg)

                self.tasks_queue.task_done()
            except Exception as e:
                # send an error message back to network if something wrong occured
                self.messaging.send_message(NodeMessages.reply_create({'success': False,  "command": "error",
                                                                       'msg': str(e), 'client_id': CLIENT_ID}).get_dict())

    def start_messaging(self, block: Optional[bool] = False):
        """This method calls the start method of messaging class

        Args:
            block (bool, optional): Defaults to False.
        """
        self.messaging.start(block)

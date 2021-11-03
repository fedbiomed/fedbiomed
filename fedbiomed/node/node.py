from json import decoder
import uuid

from typing import Optional, Union, Dict, Any

from fedbiomed.common import json
from fedbiomed.common.logger import logger
from fedbiomed.common.tasks_queue import TasksQueue
from fedbiomed.common.messaging import Messaging,  MessagingType
from fedbiomed.common.message import NodeMessages
from fedbiomed.node.environ import environ
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.round import Round
from fedbiomed.node.data_manager import Data_manager

import validators


class Node:
    """ Defines the behaviour of the node, while communicating
    with researcher through Messager, and executing / parsing task
    requested by researcher stored in a queue.
    """
    def __init__(self, data_manager: Data_manager):

        self.tasks_queue = TasksQueue(environ['MESSAGES_QUEUE_DIR'], environ['TMP_DIR'])
        self.messaging = Messaging(self.on_message, MessagingType.NODE,
                                   environ['CLIENT_ID'], environ['MQTT_BROKER'], environ['MQTT_BROKER_PORT'])
        self.data_manager = data_manager
        self.rounds = []

    def add_task(self, task: dict):
        """This method adds a task to the queue

        Args:
            task (dict): is a Message object describing a training task
        """
        self.tasks_queue.add(task)

    def on_message(self, msg, topic = None):
        """Handler to be used with `Messaging` class (ie with messager).
        It is called when a  messsage arrive through the messager
        It reads and triggers instruction received by node from Researcher,
        mainly:
        - ping requests,
        - train requests (then a new task will be added on node 's task queue),
        - search requests (for searching data in node's database).

        Args:
            msg (Dict[str, Any]): incoming message from Researcher.
            Must contain key named `command`, describing the nature
            of the command (ping requests, train requests,
            or search requests).

            topic(str): topic name, decision (specially on researcher) may
            be done regarding of the topic.
        """
        # TODO: describe all exceptions defined in this method
        logger.debug('Message received: ' + str(msg))
        try:
            # get the request from the received message (from researcher)
            command = msg['command']
            request = NodeMessages.request_create(msg).get_dict()
            if command == 'train':
                # add training task to queue
                self.add_task(request)
            elif command == 'ping':
                self.messaging.send_message(
                    NodeMessages.reply_create(
                        {
                            'researcher_id': msg['researcher_id'],
                            'node_id': environ['CLIENT_ID'],
                            'success': True,
                            'sequence': msg['sequence'],
                            'command': 'pong'
                         }).get_dict())
            elif command == 'search':
                # Look for databases matching the tags
                databases = self.data_manager.search_by_tags(msg['tags'])
                if len(databases) != 0:
                    # remove path from search to avoid privacy issues
                    for d in databases:
                        d.pop('path', None)

                    self.messaging.send_message(NodeMessages.reply_create(
                        {'success': True,
                         "command": "search",
                         'node_id': environ['CLIENT_ID'],
                         'researcher_id': msg['researcher_id'],
                         'databases': databases,
                         'count': len(databases)}).get_dict())
            elif command == 'list':
                 # Get list of all datasets
                 databases = self.data_manager.list_my_data(verbose=False)
                 remove_key = ['path', 'dataset_id']
                 for d in databases:
                     for key in remove_key:
                        d.pop(key, None)

                 self.messaging.send_message(NodeMessages.reply_create(
                     {'success': True,
                      'command': 'list',
                      'node_id': environ['CLIENT_ID'],
                      'researcher_id': msg['researcher_id'],
                      'databases': databases,
                      'count' : len(databases),
                     }).get_dict())
            else:
                raise NotImplementedError('Command not found')
        except decoder.JSONDecodeError:
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.messaging.send_message(NodeMessages.reply_create(
                {'success': False,
                 'command': "error",
                 'node_id': environ['CLIENT_ID'],
                 'researcher_id': resid,
                 'msg': "Not able to deserialize the message"}).get_dict())
        except NotImplementedError:
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.messaging.send_message(NodeMessages.reply_create(
                {'success': False,
                 'command': "error",
                 'node_id': environ['CLIENT_ID'],
                 'researcher_id': resid,
                 'msg': f"Command `{command}` is not implemented"}).get_dict())
        except KeyError:
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.messaging.send_message(NodeMessages.reply_create(
                {'success': False,
                 'command': "error",
                 'node_id': environ['CLIENT_ID'],
                 'researcher_id': resid,
                 'msg': "'command' property was not found"}).get_dict())
        except TypeError:  # Message was not serializable
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.messaging.send_message(NodeMessages.reply_create(
                {'success': False,
                 'command': "error",
                 'node_id': environ['CLIENT_ID'],
                 'researcher_id': resid,
                 'msg': 'Message was not serializable'}).get_dict())

    def parser_task(self, msg: Union[bytes, str, Dict[str, Any]]):
        """ This method parses a given task message to create a round instance

        Args:
            msg (Union[bytes, str, Dict[str, Any]]): serialized Message object
            to parse (or that have been parsed)
        """
        if isinstance(msg, str) or isinstance(msg, bytes):
            msg = json.deserialize_msg(msg)
        msg = NodeMessages.request_create(msg)
        # msg becomes a TrainRequest object
        hist_monitor = HistoryMonitor(job_id=msg.get_param(
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

        self.rounds = []  # store here rounds associated to each dataset_id
        # (so it is possible to train model on several dataset per round)

        if environ['CLIENT_ID'] in msg.get_param('training_data'):
            for dataset_id in msg.get_param('training_data')[environ['CLIENT_ID']]:
                alldata = self.data_manager.search_by_id(dataset_id)
                if len(alldata) != 1 or not 'path' in alldata[0].keys():
                    # TODO: create a data structure for messaging
                    # (ie an object creating a dict with field accordingly)
                    # FIXME: 'the confdition above depends on database model
                    # if database model changes (ie `path` field removed/
                    # modified);
                    # condition above is likely to be false
                    self.messaging.send_message(NodeMessages.reply_create(
                        {'success': False,
                         'command': "error",
                         'node_id': environ['CLIENT_ID'],
                         'researcher_id': researcher_id,
                         'msg': "Did not found proper data in local datasets"}
                        ).get_dict())
                else:
                    self.rounds.append(Round(model_kwargs,
                        training_kwargs,
                        alldata[0],
                        model_url,
                        model_class,
                        params_url,
                        job_id,
                        researcher_id,
                        hist_monitor))

    def task_manager(self):
        """ This method manages training tasks in the queue
        """

        while True:
            item = self.tasks_queue.get()

            try:
                logger.debug('[TASKS QUEUE] Item:' + str(item))
                self.parser_task(item)
                # once task is out of queue, initiate training rounds
                for round in self.rounds:
                    # iterate over each dataset found
                    # in the current round (here round refers
                    # to a round to be done on a specific dataset).
                    msg = round.run_model_training()
                    self.messaging.send_message(msg)

                self.tasks_queue.task_done()
            except Exception as e:
                # send an error message back to network if something
                # wrong occured
                self.messaging.send_message(
                    NodeMessages.reply_create(
                        {
                            'success': False,
                            "command": "error",
                            'msg': str(e),
                            'node_id': environ['CLIENT_ID']
                        }
                    ).get_dict()
                )

    def start_messaging(self, block: Optional[bool] = False):
        """This method calls the start method of messaging class

        Args:
            block (bool, optional): Defaults to False.
        """
        self.messaging.start(block)

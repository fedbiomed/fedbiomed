'''
Core code of the node component.
'''

from json import decoder

from typing import Optional, Union, Dict, Any

from fedbiomed.common import json
from fedbiomed.common.constants import ComponentType, ErrorNumbers, SecaggElementTypes
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages, SecaggRequest, TrainRequest
from fedbiomed.common.messaging import Messaging
from fedbiomed.common.tasks_queue import TasksQueue

from fedbiomed.node.environ import environ
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.model_manager import ModelManager
from fedbiomed.node.round import Round
from fedbiomed.node.secagg import SecaggSetup, SecaggServkeySetup, SecaggBiprimeSetup

import validators


class Node:
    """Core code of the node component.

    Defines the behaviour of the node, while communicating
    with the researcher through the `Messaging`, parsing messages from the researcher,
    etiher treating them instantly or queuing them,
    executing tasks requested by researcher stored in the queue.
    """

    def __init__(self,
                 dataset_manager: DatasetManager,
                 model_manager: ModelManager,
                 node_args: Union[dict, None] = None):
        """Constructor of the class.

        Attributes:
            dataset_manager: `DatasetManager` object for managing the node's datasets.
            model_manager: `ModelManager` object managing the node's models.
            node_args: Command line arguments for node.
        """

        self.tasks_queue = TasksQueue(environ['MESSAGES_QUEUE_DIR'], environ['TMP_DIR'])
        self.messaging = Messaging(self.on_message, ComponentType.NODE,
                                   environ['NODE_ID'], environ['MQTT_BROKER'], environ['MQTT_BROKER_PORT'])
        self.dataset_manager = dataset_manager
        self.model_manager = model_manager
        self.rounds = []

        self.node_args = node_args

    def add_task(self, task: dict):
        """Adds a task to the pending tasks queue.

        Args:
            task: A `Message` object describing a training task
        """
        self.tasks_queue.add(task)

    def on_message(self, msg: dict, topic: str = None):
        """Handler to be used with `Messaging` class (ie the messager).

        Called when a  message arrives through the `Messaging`.
        It reads and triggers instructions received by node from Researcher,
        mainly:
        - ping requests,
        - train requests (then a new task will be added on node's task queue),
        - search requests (for searching data in node's database).

        Args:
            msg: Incoming message from Researcher.
                Must contain key named `command`, describing the nature
                of the command (ping requests, train requests,
                or search requests).
                Should be formatted as a `Message`.
            topic: Messaging topic name, decision (specially on researcher) may
                be done regarding of the topic. Currently unused.
        """
        # TODO: describe all exceptions defined in this method
        logger.debug('Message received: ' + str(msg))
        try:
            # get the request from the received message (from researcher)
            command = msg['command']
            request = NodeMessages.request_create(msg).get_dict()
            if command in ['train', 'secagg']:
                # add training task to queue
                self.add_task(request)
            elif command == 'secagg-delete':
                logger.info('Not implemented yet, PUT SECAGG DELETE PAYLOAD HERE')
                self.messaging.send_message(
                    NodeMessages.reply_create(
                        {
                            'researcher_id': msg['researcher_id'],
                            'secagg_id': msg['secagg_id'],
                            'sequence': msg['sequence'],
                            'success': True,
                            'node_id': environ['NODE_ID'],
                            'msg': '',
                            'command': 'secagg-delete'
                        }).get_dict())
            elif command == 'ping':
                self.messaging.send_message(
                    NodeMessages.reply_create(
                        {
                            'researcher_id': msg['researcher_id'],
                            'node_id': environ['NODE_ID'],
                            'success': True,
                            'sequence': msg['sequence'],
                            'command': 'pong'
                        }).get_dict())
            elif command == 'search':
                # Look for databases matching the tags
                databases = self.dataset_manager.search_by_tags(msg['tags'])
                if len(databases) != 0:
                    databases = self.dataset_manager.obfuscate_private_information(databases)
                    # FIXME: what happens if len(database) == 0
                    self.messaging.send_message(NodeMessages.reply_create(
                        {'success': True,
                         'command': 'search',
                         'node_id': environ['NODE_ID'],
                         'researcher_id': msg['researcher_id'],
                         'databases': databases,
                         'count': len(databases)}).get_dict())
            elif command == 'list':
                # Get list of all datasets
                databases = self.dataset_manager.list_my_data(verbose=False)
                databases = self.dataset_manager.obfuscate_private_information(databases)
                self.messaging.send_message(NodeMessages.reply_create(
                    {'success': True,
                     'command': 'list',
                     'node_id': environ['NODE_ID'],
                     'researcher_id': msg['researcher_id'],
                     'databases': databases,
                     'count': len(databases),
                     }).get_dict())
            elif command == 'approval':
                # Ask for model approval
                self.model_manager.reply_model_approval_request(request, self.messaging)
            elif command == 'training-plan-status':
                # Check is model approved
                self.model_manager.reply_model_status_request(request, self.messaging)

            else:
                raise NotImplementedError('Command not found')
        except decoder.JSONDecodeError:
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.send_error(ErrorNumbers.FB301,
                            extra_msg="Not able to deserialize the message",
                            researcher_id=resid)
        except NotImplementedError:
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.send_error(ErrorNumbers.FB301,
                            extra_msg=f"Command `{command}` is not implemented",
                            researcher_id=resid)
        except KeyError:
            # FIXME: this error could be raised for other missing keys (eg
            # researcher_id, ....)
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.send_error(ErrorNumbers.FB301,
                            extra_msg="'command' property was not found",
                            researcher_id=resid)
        except TypeError:  # Message was not serializable
            resid = 'researcher_id' in msg.keys(
            ) and msg['researcher_id'] or 'unknown_researcher_id'
            self.send_error(ErrorNumbers.FB301,
                            extra_msg='Message was not serializable',
                            researcher_id=resid)

    def task_secagg(self, msg: SecaggRequest) -> None:
        """Parses a given secagg setup task message and execute secagg task.

        Args:
            msg: `SecaggRequest` message object to parse
        """
        # 1. Parse message content
        researcher_id = msg.get_param('researcher_id')
        secagg_id = msg.get_param('secagg_id')
        sequence = msg.get_param('sequence')
        element = msg.get_param('element')
        parties = msg.get_param('parties')

        if element in [m.value for m in SecaggElementTypes]:
            element = SecaggElementTypes(element)
        else:
            element = None

        if not all([researcher_id, secagg_id, element, len(parties) >= 3]):
            return None

        element2class = {
            'SERVER_KEY': SecaggServkeySetup,
            'BIPRIME': SecaggBiprimeSetup
        }

        # 2. Instantiate secagg context element
        try:
            if element.name in element2class.keys():
                # instantiate a `SecaggSetup` object
                secagg = element2class[element.name](researcher_id, secagg_id, sequence, parties)
            else:
                # should not exist 
                secagg = None
            error = ''
        except Exception as e:
            # bad secagg request
            error = e
            secagg = None

        # 3. Execute
        if secagg:
            try:
                logger.info(f"Entering secagg setup phase on node {environ['NODE_ID']}")
                msg = secagg.setup()
                self.messaging.send_message(msg)
            except Exception as e:
                errmess = f'{ErrorNumbers.FB318}: error during secagg setup for type ' \
                    f'{secagg.element()}: {e}'
                logger.error(errmess)
                self.messaging.send_message(
                    NodeMessages.reply_create(
                        {
                            'researcher_id': secagg.researcher_id(),
                            'secagg_id': secagg.secagg_id(),
                            'sequence': secagg.sequence(),
                            'success': False,
                            'node_id': environ['NODE_ID'],
                            'msg': errmess,
                            'command': 'secagg'
                        }
                    ).get_dict()
                )
        else:
            # bad secagg request, cannot reply as secagg
            errmess = f'{ErrorNumbers.FB318}: bad secure aggregation request message ' \
                f"received by {environ['NODE_ID']}: {error}"
            logger.error(errmess)
            self.messaging.send_message(
                NodeMessages.reply_create(
                    {
                        'command': 'error',
                        'extra_msg': errmess,
                        'node_id': environ['NODE_ID'],
                        'researcher_id': 'NOT_SET',
                        'errnum': ErrorNumbers.FB318
                    }
                ).get_dict()
            )


    def parser_task_train(self, msg: TrainRequest):
        """Parses a given training task message to create a round instance

        Args:
            msg: `TrainRequest` message object to parse
        """
        # msg becomes a TrainRequest object
        hist_monitor = HistoryMonitor(job_id=msg.get_param('job_id'),
                                      researcher_id=msg.get_param('researcher_id'),
                                      client=self.messaging)
        # Get arguments for the model and training
        model_kwargs = msg.get_param('model_args') or {}
        training_kwargs = msg.get_param('training_args') or {}
        training_status = msg.get_param('training') or False
        training_plan_url = msg.get_param('training_plan_url')
        training_plan_class = msg.get_param('training_plan_class')
        params_url = msg.get_param('params_url')
        job_id = msg.get_param('job_id')
        researcher_id = msg.get_param('researcher_id')

        assert training_plan_url is not None, 'URL for model on repository not found.'
        assert validators.url(
            training_plan_url), 'URL for model on repository is not valid.'
        assert training_plan_class is not None, 'classname for the model and training routine was not found in message.'

        assert isinstance(
            training_plan_class,
            str), '`training_plan_class` must be a string corresponding to the classname for the model and training routine in the repository'  # noqa

        self.rounds = []  # store here rounds associated to each dataset_id
        # (so it is possible to train model on several dataset per round)

        if environ['NODE_ID'] in msg.get_param('training_data'):
            for dataset_id in msg.get_param('training_data')[environ['NODE_ID']]:
                data = self.dataset_manager.get_by_id(dataset_id)
                if data is None or 'path' not in data.keys():
                    # TODO: create a data structure for messaging
                    # (ie an object creating a dict with field accordingly)
                    # FIXME: 'the condition above depends on database model
                    # if database model changes (ie `path` field removed/
                    # modified);
                    # condition above is likely to be false
                    logger.error('Did not found proper data in local datasets ' +
                                 f'on node={environ["NODE_ID"]}')
                    self.messaging.send_message(NodeMessages.reply_create(
                        {'command': "error",
                         'node_id': environ['NODE_ID'],
                         'researcher_id': researcher_id,
                         'errnum': ErrorNumbers.FB313,
                         'extra_msg': "Did not found proper data in local datasets"}
                    ).get_dict())
                else:
                    dlp_and_loading_block_metadata = None
                    if 'dlp_id' in data:
                        dlp_and_loading_block_metadata = self.dataset_manager.get_dlp_by_id(data['dlp_id'])
                    self.rounds.append(Round(model_kwargs,
                                             training_kwargs,
                                             training_status,
                                             data,
                                             training_plan_url,
                                             training_plan_class,
                                             params_url,
                                             job_id,
                                             researcher_id,
                                             hist_monitor,
                                             self.node_args,
                                             dlp_and_loading_block_metadata=dlp_and_loading_block_metadata))

    def task_manager(self):
        """Manages training tasks in the queue.
        """

        while True:
            item = self.tasks_queue.get()
            logger.debug('[TASKS QUEUE] Item:' + str(item))

            try:
                item = NodeMessages.request_create(item)
                command = item.get_param('command')
            except Exception as e:
                # send an error message back to network if something wrong occured
                self.messaging.send_message(
                    NodeMessages.reply_create(
                        {
                            'command': 'error',
                            'extra_msg': str(e),
                            'node_id': environ['NODE_ID'],
                            'researcher_id': 'NOT_SET',
                            'errnum': ErrorNumbers.FB300
                        }
                    ).get_dict()
                )

            if command == 'train':
                try:
                    self.parser_task_train(item)
                    # once task is out of queue, initiate training rounds
                    for round in self.rounds:
                        # iterate over each dataset found
                        # in the current round (here round refers
                        # to a round to be done on a specific dataset).
                        msg = round.run_model_training()
                        self.messaging.send_message(msg)
                except Exception as e:
                    # send an error message back to network if something
                    # wrong occured
                    self.messaging.send_message(
                        NodeMessages.reply_create(
                            {
                                'command': 'error',
                                'extra_msg': str(e),
                                'node_id': environ['NODE_ID'],
                                'researcher_id': 'NOT_SET',
                                'errnum': ErrorNumbers.FB300
                            }
                        ).get_dict()
                    )
            elif command == 'secagg':
                self.task_secagg(item)
            else:
                errmess = f'{ErrorNumbers.FB319.value}: "{command}"'
                logger.error(errmess)
                self.messaging.send_message(
                    NodeMessages.reply_create(
                        {
                            'command': 'error',
                            'extra_msg': errmess,
                            'node_id': environ['NODE_ID'],
                            'researcher_id': 'NOT_SET',
                            'errnum': ErrorNumbers.FB319
                        }
                    ).get_dict()
                )

            self.tasks_queue.task_done()

    def start_messaging(self, block: Optional[bool] = False):
        """Calls the start method of messaging class.

        Args:
            block: Whether messager is blocking (or not). Defaults to False.
        """
        self.messaging.start(block)

    def send_error(self, errnum: ErrorNumbers, extra_msg: str = "", researcher_id: str = "<unknown>"):
        """Sends an error message.

        It is a wrapper of `Messaging.send_error()`.

        Args:
            errnum: Code of the error.
            extra_msg: Additional human readable error message.
            researcher_id: Destination researcher.
        """

        #
        self.messaging.send_error(errnum, extra_msg=extra_msg, researcher_id=researcher_id)

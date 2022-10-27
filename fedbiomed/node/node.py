'''
Core code of the node component.
'''

from json import JSONDecodeError
from typing import Any, Dict, List, Optional

import validators

from fedbiomed.common.constants import ComponentType, ErrorNumbers, SecaggElementTypes
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.history_monitor import HistoryMonitor
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages, SecaggRequest, TrainRequest
from fedbiomed.common.messaging import Messaging
from fedbiomed.common.tasks_queue import TasksQueue

from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.environ import environ
from fedbiomed.node.round import Round
from fedbiomed.node.secagg import SecaggBiprimeSetup, SecaggServkeySetup
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager


class Node:
    """Core code of the node component.

    Defines the behaviour of the node, while communicating
    with the researcher through the `Messaging`, parsing messages from the researcher,
    etiher treating them instantly or queuing them,
    executing tasks requested by researcher stored in the queue.
    """

    def __init__(self,
                 dataset_manager: DatasetManager,
                 tp_security_manager: TrainingPlanSecurityManager,
                 node_args: Optional[Dict[str, Any]] = None):
        """Constructor of the class.

        Attributes:
            dataset_manager: `DatasetManager` object for managing the node's datasets.
            tp_security_manager: `TrainingPlanSecurityManager` object managing the node's training plans.
            node_args: Command line arguments for node.
        """

        self.tasks_queue = TasksQueue(environ['MESSAGES_QUEUE_DIR'], environ['TMP_DIR'])
        self.messaging = Messaging(self.on_message, ComponentType.NODE,
                                   environ['NODE_ID'], environ['MQTT_BROKER'], environ['MQTT_BROKER_PORT'])
        self.dataset_manager = dataset_manager
        self.tp_security_manager = tp_security_manager
        self.node_args = node_args

    def on_message(self, msg: dict, topic: Optional[str] = None) -> None:
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
            request = NodeMessages.request_create(msg).get_dict()
            command = request["command"]
            if command in ['train', 'secagg']:
                # add training task to queue
                self.tasks_queue.add(request)
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
                # Ask for training plan approval
                self.tp_security_manager.reply_training_plan_approval_request(request, self.messaging)
            elif command == 'training-plan-status':
                # Check is training plan approved
                self.tp_security_manager.reply_training_plan_status_request(request, self.messaging)

            else:
                raise NotImplementedError('Command not found')
        except JSONDecodeError:
            self.messaging.send_error(
                ErrorNumbers.FB301,
                extra_msg="Unable to deserialize the message",
                researcher_id=msg.get("researcher_id", "unknown_researcher_id")
            )
        except FedbiomedError as exc:
            self.messaging.send_error(
                ErrorNumbers.FB301,
                extra_msg=str(exc),
                researcher_id=msg.get("researcher_id", "unknown_researcher_id")
            )
        except NotImplementedError:
            self.messaging.send_error(
                ErrorNumbers.FB301,
                extra_msg=f"Command `{command}` is not implemented",
                researcher_id=msg.get("researcher_id", "unknown_researcher_id")
            )
        except TypeError:  # Message was not serializable
            self.messaging.send_error(
                ErrorNumbers.FB301,
                extra_msg="Message was not serializable",
                researcher_id=msg.get("researcher_id", "unknown_researcher_id")
            )

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
            if element.name in element2class:
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

    def parse_train_request(self, msg: TrainRequest) -> List[Round]:
        """Parses a given training task message to create round instances.

        Args:
            msg: `TrainRequest` message instance to parse.
        """
        # msg becomes a TrainRequest object
        hist_monitor = HistoryMonitor(
            job_id=msg.get_param('job_id'),
            node_id=environ['NODE_ID'],
            researcher_id=msg.get_param('researcher_id'),
            client=self.messaging
        )
        # Get arguments for the model and training.
        round_kwargs = {
            "model_kwargs": msg.get_param('model_args') or {},
            "training_kwargs": msg.get_param('training_args') or {},
            "training_plan_url": msg.get_param('training_plan_url'),
            "params_url": msg.get_param('params_url'),
            "training": msg.get_param('training') or False,
            "job_id": msg.get_param('job_id'),
            "researcher_id": msg.get_param('researcher_id'),
            "history_monitor": hist_monitor,
            "node_args": self.node_args
        }
        if round_kwargs["training_plan_url"] is None:
            raise FedbiomedError('URL for training plan on repository not found.')
        if not validators.url(round_kwargs["training_plan_url"]):
            raise FedbiomedError('URL for training plan on repository is not valid.')

        rounds = []  # store here rounds associated to each dataset_id
        # (so it is possible to train a model on several datasets per round)

        if environ['NODE_ID'] in msg.get_param('training_data'):
            for dataset_id in msg.get_param('training_data')[environ['NODE_ID']]:
                data = self.dataset_manager.get_by_id(dataset_id)
                if data is None or 'path' not in data.keys():
                    logger.error('Did not find proper data in local datasets '
                                 f'on node={environ["NODE_ID"]}')
                    self.messaging.send_message(NodeMessages.reply_create(
                        {'command': "error",
                         'node_id': environ["NODE_ID"],
                         'researcher_id': round_kwargs["researcher_id"],
                         'errnum': ErrorNumbers.FB313,
                         'extra_msg': "Did not find proper data in local datasets"}
                    ).get_dict())
                else:
                    dlp = (
                        self.dataset_manager.get_dlp_by_id(data['dlp_id'])
                        if 'dlp_id' in data else None
                    )
                    round_ = Round(
                        dataset=data,
                        dlp_and_loading_block_metadata=dlp,
                        **round_kwargs
                    )
                    rounds.append(round_)
        # Return the list of Round instances parsed from the request.
        return rounds

    def task_manager(self) -> None:
        """Manage training tasks in the queue."""
        while True:
            item = self.tasks_queue.get()
            logger.debug('[TASKS QUEUE] Item:' + str(item))
            # Parse the received message.
            try:
                message = NodeMessages.request_create(item)
            except FedbiomedError as exc:
                self._send_task_error(exc)
                continue
            # Case when message is a TrainRequest.
            if isinstance(message, TrainRequest):
                try:
                    # Parse the request into a list of rounds to run
                    # (once per dataset matching the request).
                    rounds = self.parse_train_request(message)
                    # Run each and every round and send back train replies.
                    for round_ in rounds:
                        reply = round_.run()
                        self.messaging.send_message(reply.get_dict())
                except FedbiomedError as exc:
                    self._send_task_error(exc)
                    continue
            # Case when message is a SecaggRequest.
            elif isinstance(message, SecaggRequest):
                self.task_secagg(message)
            # Case when message is of unsupported type.
            else:
                command = message.get_param('command')
                self._send_task_error(
                    FedbiomedError(f"{ErrorNumbers.FB319.value}: {command}")
                )
            # Mark the task as done.
            self.tasks_queue.task_done()

    def _send_task_error(self, exc: Exception) -> None:
        """Send an error message due to an exception in `task_manager`."""
        params = {
            "command": "error",
            "extra_msg": str(exc),
            "node_id": environ["NODE_ID"],
            "researcher_id": "NOT_SET",
            "errnum": ErrorNumbers.FB300
        }
        message = NodeMessages.reply_create(params)
        self.messaging.send_message(message.get_dict())

    def start_messaging(self, block: bool = False) -> None:
        """Calls the start method of messaging class.

        Args:
            block: Whether messager is blocking (or not). Defaults to False.
        """
        self.messaging.start(block)

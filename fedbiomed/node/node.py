# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

'''
Core code of the node component.
'''
from typing import Optional, Union, Callable

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedMessageError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages, SecaggDeleteRequest, SecaggRequest, TrainRequest, ErrorMessage
from fedbiomed.common.tasks_queue import TasksQueue

from fedbiomed.transport.controller import GrpcController
from fedbiomed.transport.client import ResearcherCredentials

from fedbiomed.node.environ import environ
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.node.training_plan_security_manager import TrainingPlanSecurityManager
from fedbiomed.node.round import Round
from fedbiomed.node.secagg import SecaggSetup
from fedbiomed.node.secagg_manager import SecaggManager


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
                 node_args: Union[dict, None] = None):
        """Constructor of the class.

        Attributes:
            dataset_manager: `DatasetManager` object for managing the node's datasets.
            tp_security_manager: `TrainingPlanSecurityManager` object managing the node's training plans.
            node_args: Command line arguments for node.
        """

        self._tasks_queue = TasksQueue(environ['MESSAGES_QUEUE_DIR'], environ['TMP_DIR'])
        # TODO: extend for multiple researchers, currently expect only one
        res = environ["RESEARCHERS"][0]
        self._grpc_client = GrpcController(
            node_id=environ["ID"],
            researchers=[ResearcherCredentials(port=res['port'], host=res['ip'], certificate=res['certificate'])],
            on_message=self.on_message,
        )
        self.dataset_manager = dataset_manager
        self.tp_security_manager = tp_security_manager

        self.node_args = node_args

    def add_task(self, task: dict):
        """Adds a task to the pending tasks queue.

        Args:
            task: A `Message` object describing a training task
        """
        self._tasks_queue.add(task)

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
        no_print = ["aggregator_args", "aux_vars", "params", "training_plan"]
        msg_print = {key: value for key, value in msg.items() if key not in no_print}
        logger.debug('Message received: ' + str(msg_print))
        try:
            # get the request from the received message (from researcher)
            command = msg['command']
            request = NodeMessages.format_incoming_message(msg).get_dict()
            if command in ['train', 'secagg']:
                # add training task to queue
                #MANI
                self.node_args["gpu_num"] = msg["training_args"]["gpu_num"]
                request["model_args"] = {**request["model_args"], "node_args": self.node_args, "node_id": environ["ID"]}
                self.add_task(request)
            elif command == 'secagg-delete':
                self._task_secagg_delete(NodeMessages.format_incoming_message(msg))
            elif command == 'ping':
                self._grpc_client.send(
                    NodeMessages.format_outgoing_message(
                        {
                            'researcher_id': msg['researcher_id'],
                            'request_id': msg['request_id'],
                            'node_id': environ['NODE_ID'],
                            'success': True,
                            'command': 'pong'
                        }))
            elif command == 'search':
                # Look for databases matching the tags
                databases = self.dataset_manager.search_by_tags(msg['tags'])
                if len(databases) != 0:
                    databases = self.dataset_manager.obfuscate_private_information(databases)
                self._grpc_client.send(NodeMessages.format_outgoing_message(
                    {'request_id': msg['request_id'],
                     'success': True,
                     'command': 'search',
                     'node_id': environ['NODE_ID'],
                     'researcher_id': msg['researcher_id'],
                     'databases': databases,
                     'count': len(databases)}))

            elif command == 'list':
                # Get list of all datasets
                databases = self.dataset_manager.list_my_data(verbose=False)
                databases = self.dataset_manager.obfuscate_private_information(databases)
                self._grpc_client.send(NodeMessages.format_outgoing_message(
                    {'success': True,
                     'request_id': msg['request_id'],
                     'command': 'list',
                     'node_id': environ['NODE_ID'],
                     'researcher_id': msg['researcher_id'],
                     'databases': databases,
                     'count': len(databases),
                     }))
            elif command == 'approval':
                # Ask for training plan approval
                reply = self.tp_security_manager.reply_training_plan_approval_request(request)
                self._grpc_client.send(reply)
            elif command == 'training-plan-status':
                # Check is training plan approved
                reply = self.tp_security_manager.reply_training_plan_status_request(request)
                self._grpc_client.send(reply)
            else:
                raise NotImplementedError('Command not found')

        except NotImplementedError:
            resid = msg.get('researcher_id', 'unknown_researcher_id')
            self.send_error(ErrorNumbers.FB301,
                            extra_msg=f"Command `{command}` is not implemented",
                            researcher_id=resid)
        except KeyError:
            # FIXME: this error could be raised for other missing keys (eg
            # researcher_id, ....)
            resid = msg.get('researcher_id', 'unknown_researcher_id')
            self.send_error(ErrorNumbers.FB301,
                            extra_msg="'command' property was not found",
                            researcher_id=resid)
        except FedbiomedMessageError:  # Message was not properly formatted
            resid = msg.get('researcher_id', 'unknown_researcher_id')
            self.send_error(ErrorNumbers.FB301,
                            extra_msg='Message was not properly formatted',
                            researcher_id=resid)
        except TypeError:  # Message was not serializable
            resid = msg.get('researcher_id', 'unknown_researcher_id')
            self.send_error(ErrorNumbers.FB301,
                            extra_msg='Message was not serializable',
                            researcher_id=resid)

    def _task_secagg_delete(self, msg: SecaggDeleteRequest) -> None:
        """Parse a given secagg delete task message and execute secagg delete task.

        Args:
            msg: `SecaggDeleteRequest` message object to parse
        """

        secagg_id = msg.get_param('secagg_id')

        reply = {
            'researcher_id': msg.get_param('researcher_id'),
            'secagg_id': secagg_id,
            'request_id': msg.request_id,
            'command': 'secagg-delete'}

        try:
            secagg_manager = SecaggManager(element=msg.get_param('element'))()
        except Exception as e:
            message = f'{ErrorNumbers.FB321.value}: Can not instantiate SecaggManager object {e}'
            logger.error(message)
            return self.reply({"success": False, "msg": message, **reply})

        try:
            status = secagg_manager.remove(secagg_id=secagg_id,
                                           job_id=msg.get_param('job_id'))
            if status:
                message = 'Delete request is successful'
            else:
                message = f"{ErrorNumbers.FB321.value}: no such secagg context element in node database for " \
                          f"node_id={environ['NODE_ID']} secagg_id={secagg_id}"
        except Exception as e:
            message = f"{ErrorNumbers.FB321.value}: error during secagg delete on node_id={environ['NODE_ID']} " \
                      f'secagg_id={secagg_id}: {e}'
            logger.error(message)
            status = False

        return self.reply({"success": status, "msg": message, **reply})

    def _task_secagg(self, msg: SecaggRequest) -> None:
        """Parse a given secagg setup task message and execute secagg task.

        Args:
            msg: `SecaggRequest` message object to parse
        """
        setup_arguments = {key: value for (key, value) in msg.get_dict().items()}

        try:
            secagg = SecaggSetup(**setup_arguments)()
        except Exception as error_message:
            logger.error(error_message)
            return self.reply({"researcher_id": msg.get_param('researcher_id'),
                               "secagg_id": msg.get_param('secagg_id'),
                               'request_id': msg.request_id,
                               "msg": str(error_message),
                               "success": False,
                               "command": "secagg"})

        reply = secagg.setup()
        reply["request_id"] = msg.request_id

        return self.reply(reply)

    def parser_task_train(self, msg: TrainRequest) -> Union[Round, None]:
        """Parses a given training task message to create a round instance

        Args:
            msg: `TrainRequest` message object to parse

        Returns:
            a `Round` object for the training to perform, or None if no training
        """
        round_ = None
        # msg becomes a TrainRequest object
        hist_monitor = HistoryMonitor(job_id=msg.get_param('job_id'),
                                      researcher_id=msg.get_param('researcher_id'),
                                      send=self._grpc_client.send)

        dataset_id = msg.get_param('dataset_id')
        data = self.dataset_manager.get_by_id(dataset_id)

        if data is None:
            logger.error('Did not found proper data in local datasets '
                         f'on node={environ["NODE_ID"]}')
            self._grpc_client.send(NodeMessages.format_outgoing_message(
                {'command': "error",
                 'request_id': msg.request_id,
                 'node_id': environ['NODE_ID'],
                 'researcher_id': msg.get_param('researcher_id'),
                 'errnum': ErrorNumbers.FB313.name,
                 'extra_msg': "Did not found proper data in local datasets"}
            ))
        else:
            dlp_and_loading_block_metadata = None
            if 'dlp_id' in data:
                dlp_and_loading_block_metadata = self.dataset_manager.get_dlp_by_id(data['dlp_id'])

            round_ = Round(training_plan=msg.get_param('training_plan'),
                           training_plan_class=msg.get_param('training_plan_class'),
                           model_kwargs=msg.get_param('model_args') or {},
                           training_kwargs=msg.get_param('training_args') or {},
                           training=msg.get_param('training') or False,
                           dataset=data,
                           params=msg.get_param('params'),
                           job_id=msg.get_param('job_id'),
                           researcher_id=msg.get_param('researcher_id'),
                           history_monitor=hist_monitor,
                           aggregator_args=msg.get_param('aggregator_args') or None,
                           node_args=self.node_args,
                           round_number=msg.get_param('round'),
                           dlp_and_loading_block_metadata=dlp_and_loading_block_metadata,
                           aux_vars=msg.get_param('aux_vars'))

            # the round raises an error if it cannot initialize
            err_msg = round_.initialize_arguments(msg.get_param('state_id'))
            if err_msg is not None:
                self._grpc_client.send(
                    NodeMessages.format_outgoing_message(
                        {   'command': 'error',
                            'node_id': environ['NODE_ID'],
                            'errnum': ErrorNumbers.FB300,
                            'researcher_id': msg.get_param('researcher_id'),
                            'extra_msg': "Could not initialize arguments."}
                    ))

        return round_

    def task_manager(self):
        """Manages training tasks in the queue.
        """

        while True:
            item = self._tasks_queue.get()
            # don't want to treat again in case of failure
            self._tasks_queue.task_done()

            logger.info(f"[TASKS QUEUE] Task received by task manager: Command: "
                        f"{item['command']} Researcher: {item['researcher_id']} Job: {item.get('job_id')}")

            try:

                item = NodeMessages.format_incoming_message(item)
                command = item.get_param('command')
            except Exception as e:
                # send an error message back to network if something wrong occured
                self._grpc_client.send(
                    NodeMessages.format_outgoing_message(
                        {
                            'command': 'error',
                            'extra_msg': str(e),
                            'node_id': environ['NODE_ID'],
                            'researcher_id': 'NOT_SET',
                            'errnum': ErrorNumbers.FB300.name
                        }
                    )
                )
            else:
                if command == 'train':
                    try:
                        round = self.parser_task_train(item)

                        # once task is out of queue, initiate training rounds
                        if round is not None:
                            # Runs model training and send message using callback
                            msg = round.run_model_training(
                                secagg_arguments={
                                    'secagg_servkey_id': item.get_param('secagg_servkey_id'),
                                    'secagg_biprime_id': item.get_param('secagg_biprime_id'),
                                    'secagg_random': item.get_param('secagg_random'),
                                    'secagg_clipping_range': item.get_param('secagg_clipping_range')
                                }
                            )
                            msg.request_id = item.request_id
                            self._grpc_client.send(msg)
                    except Exception as e:
                        # send an error message back to network if something
                        # wrong occured
                        self._grpc_client.send(
                            NodeMessages.format_outgoing_message(
                                {
                                    'command': 'error',
                                    'extra_msg': 'Round error: ' + str(e),
                                    'node_id': environ['NODE_ID'],
                                    'researcher_id': item.get_param('researcher_id'),
                                    'errnum': ErrorNumbers.FB300.name
                                }
                            )
                        )
                        logger.debug(f"{ErrorNumbers.FB300.value}: {e}")
                elif command == 'secagg':
                    self._task_secagg(item)
                else:
                    errmess = f'{ErrorNumbers.FB319.value}: "{command}"'
                    logger.error(errmess)
                    self.send_error(errnum=ErrorNumbers.FB319, extra_msg=errmess)

    def start_messaging(self, on_finish: Optional[Callable] = None):
        """Calls the start method of messaging class.

        Args:
            on_finish: Called when the tasks for handling all known researchers have finished.
                Callable has no argument.
        """
        self._grpc_client.start(on_finish)

    def is_connected(self) -> bool:
        """Checks if node is ready for communication with researcher

        Returns:
            True if node is ready, False if node is not ready
        """
        return self._grpc_client.is_connected()

    def reply(self, msg: dict):
        """Send reply to researcher

        Args:
            msg:

        """

        try:
            reply = NodeMessages.format_outgoing_message(
                {'node_id': environ['ID'],
                 **msg}
            )
        except FedbiomedMessageError as e:
            logger.error(f"{ErrorNumbers.FB601.value}: {e}")
            self.send_error(errnum=ErrorNumbers.FB601, extra_msg=f"{ErrorNumbers.FB601.value}: Can not reply "
                                                                 f"due to incorrect message type {e}.")
        except Exception as e:
            logger.error(f"{ErrorNumbers.FB601.value} Unexpected error while creating node reply message {e}")
            self.send_error(errnum=ErrorNumbers.FB601, extra_msg=f"{ErrorNumbers.FB601.value}: "
                                                                 f"Unexpected error occurred")

        else:
            self._grpc_client.send(reply)



    def send_error(
            self,
            errnum: ErrorNumbers,
            extra_msg: str = "",
            researcher_id: str = "<unknown>",
            broadcast: bool = False
    ):
        """Sends an error message.

        It is a wrapper of `Messaging.send_error()`.

        Args:
            errnum: Code of the error.
            extra_msg: Additional human readable error message.
            researcher_id: Destination researcher.
        """

        #
        try:
            self._grpc_client.send(
                ErrorMessage(
                    command='error',
                    errnum=errnum.name,
                    node_id=environ['NODE_ID'],
                    extra_msg=extra_msg,
                    researcher_id=researcher_id
                ),
                broadcast=broadcast
            )
        except Exception as e:
            # TODO: Need to keep message local, cannot send error
            logger.error(f"{ErrorNumbers.FB601.value}: Cannot send error message: {e}")

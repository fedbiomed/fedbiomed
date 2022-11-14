"""
Implements the message exchanges from researcher to nodes
"""

import inspect
import json
import os
import tabulate
import uuid

from python_minifier import minify
from time import sleep
from typing import Any, Dict, Callable

from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedTaskQueueError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ResearcherMessages
from fedbiomed.common.messaging import Messaging
from fedbiomed.common.repository import Repository
from fedbiomed.common.singleton import SingletonMeta
from fedbiomed.common.tasks_queue import TasksQueue

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.responses import Responses


class Requests(metaclass=SingletonMeta):
    """
    Represents the requests addressed from Researcher to nodes. It creates a task queue storing reply to each
    incoming message. Starts a message queue and reconfigures  message to be sent into a `Messaging` object.
    """

    def __init__(self, mess: Any = None):
        """
        Constructor of the class

        Args:
            mess: message to be sent by default.
        """
        # Need to ensure unique per researcher instance message queue to avoid conflicts
        # in case several instances of researcher (with same researcher_id ?) are active,
        # eg: a notebook not quitted and launching a script
        self.queue = TasksQueue(environ['MESSAGES_QUEUE_DIR'] + '_' + str(uuid.uuid4()), environ['TMP_DIR'])

        if mess is None or type(mess) is not Messaging:
            self.messaging = Messaging(self.on_message,
                                       ComponentType.RESEARCHER,
                                       environ['RESEARCHER_ID'],
                                       environ['MQTT_BROKER'],
                                       environ['MQTT_BROKER_PORT'])
            self.messaging.start(block=False)
        else:
            self.messaging = mess

        # defines the sequence used for ping protocol
        self._sequence = 0

        self._monitor_message_callback = None

    def get_messaging(self) -> Messaging:
        """Retrieves Messaging object

        Returns:
            Messaging object
        """
        return self.messaging

    def on_message(self, msg: Dict[str, Any], topic: str):
        """ Handler called by the [`Messaging`][fedbiomed.common.messaging] class,  when a message is received on
        researcher side.

        It is run in the communication process and must ba as quick as possible:
        - it deals with quick messages (eg: ping/pong)
        - it stores the replies of the nodes to the task queue, the message will bee
        treated by the main (computing) thread.

        Args:
            msg: de-serialized msg
            topic: topic to publish message (MQTT channel)
        """

        if topic == "general/logger":
            #
            # forward the treatment to node_log_handling() (same thread)
            self.print_node_log_message(ResearcherMessages.reply_create(msg).get_dict())
        elif topic == "general/researcher":
            #
            # *Reply messages (SearchReply, TrainReply) added to the TaskQueue
            self.queue.add(ResearcherMessages.reply_create(msg).get_dict())

            # we may trap FedbiomedTaskQueueError here then queue full
            # but what can we do except of quitting ?

        elif topic == "general/monitoring":
            if self._monitor_message_callback is not None:
                # Pass message to Monitor's on message handler
                self._monitor_message_callback(ResearcherMessages.reply_create(msg).get_dict())
        else:
            logger.error("message received on wrong topic (" + topic + ") - IGNORING")

    @staticmethod
    def print_node_log_message(log: Dict[str, Any]):
        """Prints logger messages coming from the node

        It is run on the communication process and must be as quick as possible:
        - all logs (coming from the nodes) are forwarded to the researcher logger
        (immediate display on console/file/whatever)
        """

        # log contains the original message sent by the node
        # FIXME: we should use `fedbiomed.common.json.deserialize` method
        # instead of the json method when extracting json message
        original_msg = json.loads(log["msg"])

        # Loging fancy feedback for training
        logger.info("\033[1m{}\033[0m\n"
                    "\t\t\t\t\t\033[1m NODE\033[0m {}\n"
                    "\t\t\t\t\t\033[1m MESSAGE:\033[0m {}\033[0m\n"
                    "{}".format(log["level"],
                                log["node_id"],
                                original_msg["message"],
                                5 * "-------------"))

    def send_message(self, msg: dict, client: str = None):
        """
        Ask the messaging class to send a new message (receivers are
        deduced from the message content)

        Args:
            msg: the message to send to nodes
            client: defines the channel to which the message will be sent. Defaults to None (all nodes)
        """
        logger.debug(str(environ['RESEARCHER_ID']))
        self.messaging.send_message(msg, client=client)

    def get_messages(self, commands: list = [], time: float = .0) -> Responses:
        """Goes through the queue and gets messages with the specific command

        Args:
            commands: Checks if message is containing the expecting command (the message  is discarded if it doesn't).
                Defaults to None (no command message checking, meaning all incoming messages are considered).
            time: Time to sleep in seconds before considering incoming messages. Defaults to .0.

        Returns:
            Contains the corresponding answers
        """
        sleep(time)

        answers = []
        for _ in range(self.queue.qsize()):
            try:
                item = self.queue.get(block=False)
                self.queue.task_done()

                if not commands or \
                        ('command' in item.keys() and item['command'] in commands):
                    answers.append(item)
                else:
                    # currently trash all other messages
                    pass

            except FedbiomedTaskQueueError:
                # may happend on self.queue.get()
                # if queue is empty -> we ignore it
                pass

        return Responses(answers)

    def get_responses(self,
                      look_for_commands: list,
                      timeout: float = None,
                      only_successful: bool = True) -> Responses:
        """Waits for all nodes' answers, regarding a specific command returns the list of all nodes answers

        Args:
            look_for_commands: instruction that has been sent to node. Can be either ping, search or train.
            timeout: wait for a specific duration before collecting nodes messages. Defaults to None. If set to None;
                uses value in global variable TIMEOUT instead.
            only_successful: deal only with messages that have been tagged as successful (ie with field `success=True`).
                Defaults to True.
        """
        timeout = timeout or environ['TIMEOUT']
        responses = []

        while True:
            sleep(timeout)
            new_responses = []
            for resp in self.get_messages(commands=look_for_commands, time=0):
                try:
                    if not only_successful:
                        new_responses.append(resp)
                    elif resp['success']:
                        # TODO: test if 'success'key exists
                        # what do we do if not ?
                        new_responses.append(resp)
                except Exception as e:
                    logger.error(f"Incorrect message received: {resp} - error: {e}")
                    pass

            if len(new_responses) == 0:
                "Timeout finished"
                break
            responses += new_responses
        return Responses(responses)

    def ping_nodes(self) -> list:
        """ Pings online nodes

        Returns:
            List ids of up and running nodes
        """
        self.messaging.send_message(ResearcherMessages.request_create(
            {'researcher_id': environ['RESEARCHER_ID'],
             'sequence': self._sequence,
             'command': 'ping'}).get_dict())
        self._sequence += 1

        # TODO: (below, above) handle exceptions
        nodes_online = [resp['node_id'] for resp in self.get_responses(look_for_commands=['ping'])]
        return nodes_online

    def search(self, tags: tuple, nodes: list = None) -> dict:
        """ Searches available data by tags

        Args:
            tags: Tuple containing tags associated to the data researcher is looking for.
            nodes: optionally filter nodes with this list. Default is no filtering, consider all nodes

        Returns:
            A dict with node_id as keys, and list of dicts describing available data as values
        """

        # Search datasets based on node specifications
        if nodes:
            logger.info(f'Searching dataset with data tags: {tags} on specified nodes: {nodes}')
            for node in nodes:
                self.messaging.send_message(
                    ResearcherMessages.request_create({'tags': tags,
                                                       'researcher_id': environ['RESEARCHER_ID'],
                                                       "command": "search"}
                                                      ).get_dict(),
                    client=node)
        else:
            logger.info(f'Searching dataset with data tags: {tags} for all nodes')
            self.messaging.send_message(
                ResearcherMessages.request_create({'tags': tags,
                                                   'researcher_id': environ['RESEARCHER_ID'],
                                                   "command": "search"}
                                                  ).get_dict())

        data_found = {}
        for resp in self.get_responses(look_for_commands=['search']):
            if not nodes:
                data_found[resp.get('node_id')] = resp.get('databases')
            elif resp.get('node_id') in nodes:
                data_found[resp.get('node_id')] = resp.get('databases')

            logger.info('Node selected for training -> {}'.format(resp.get('node_id')))

        if not data_found:
            logger.info("No available dataset has found in nodes with tags: {}".format(tags))

        return data_found

    def list(self, nodes: list = None, verbose: bool = False) -> dict:
        """Lists available data in each node

        Args:
            nodes: Listings datasets by given node ids. Default is None.
            verbose: If it is true it prints datasets in readable format
        """

        # If nodes list is provided
        if nodes:
            for node in nodes:
                self.messaging.send_message(
                    ResearcherMessages.request_create({'researcher_id': environ['RESEARCHER_ID'],
                                                       "command": "list"}
                                                      ).get_dict(),
                    client=node)
            logger.info(f'Listing datasets of given list of nodes : {nodes}')
        else:
            self.messaging.send_message(
                ResearcherMessages.request_create({'researcher_id': environ['RESEARCHER_ID'],
                                                   "command": "list"}).get_dict())
            logger.info('Listing available datasets in all nodes... ')

        # Get datasets from node responses
        data_found = {}
        for resp in self.get_responses(look_for_commands=['list']):
            if not nodes:
                data_found[resp.get('node_id')] = resp.get('databases')
            elif resp.get('node_id') in nodes:
                data_found[resp.get('node_id')] = resp.get('databases')

        # Print dataset tables usong data_found object
        if verbose:
            for node in data_found:
                if len(data_found[node]) > 0:
                    rows = [row.values() for row in data_found[node]]
                    headers = data_found[node][0].keys()
                    info = '\n Node: {} | Number of Datasets: {} \n'.format(node, len(data_found[node]))
                    logger.info(info + tabulate.tabulate(rows, headers, tablefmt="grid") + '\n')
                else:
                    logger.info('\n Node: {} | Number of Datasets: {}'.format(node, len(data_found[node])) +
                                " No data has been set up for this node.")

        return data_found

    def training_plan_approve(self,
                              training_plan: 'BaseTrainingPlan',
                              description: str = "no description provided",
                              nodes: list = [],
                              timeout: int = 5) -> dict:
        """Send a model and a ApprovalRequest message to node(s).

        If a list of node id(s) is provided, the message will be individually sent
        to all nodes of the list.
        If the node id(s) list is None (default), the message is broadcast to all nodes.


        Args:
            training_plan: the training plan to upload and send to the nodes for approval.
                   It can be:
                   - a path_name (str)
                   - a model (class)
                   - an instance of a model (TrainingPlan instance)
            nodes: list of nodes (specified by their UUID)
            description: Description of training plan approval request
            timeout: maximum waiting time for the answers

        Returns:
            a dictionary of pairs (node_id: status), where status indicates to the researcher
            that the model has been correctly downloaded on the node side.
            Warning: status does not mean that the model is approved, only that it has been added
            to the "approval queue" on the node side.
        """

        # first verify all arguments
        if not isinstance(nodes, list):
            logger.error("bad nodes argument, model not sent")
            return {}

        # verify the model and save it to a local file name if necessary
        if isinstance(training_plan, str):
            # model is provided as a file
            # TODO: verify that this file a a proper TrainingPlan
            if os.path.isfile(training_plan) and os.access(training_plan, os.R_OK):
                training_plan_file = training_plan
            else:
                logger.error(f"cannot access to the file ({training_plan})")
                return {}
        else:
            # we need a model instance in other cases
            if inspect.isclass(training_plan):
                # case if `model` is a class
                try:
                    training_plan_instance = training_plan()
                    deps = training_plan_instance.init_dependencies()
                    training_plan_instance.add_dependency(deps)
                except Exception as e:  # TODO: be more specific
                    logger.error(f"cannot instantiate the given model ({e})")
                    return {}
            else:
                # also handle case where model is already an instance of a class
                training_plan_instance = training_plan

            # then save this instance to a file
            training_plan_file = os.path.join(environ['TMP_DIR'],
                                              "training_plan_" + str(uuid.uuid4()) + ".py")

            try:
                training_plan_instance.save_code(training_plan_file)
            except Exception as e:  # TODO: be more specific
                logger.error(f"Cannot save the model to a file ({e})")
                logger.error(f"Are you sure that {training_plan} is a TrainingPlan ?")
                return {}

        # verify that the file can be minified before sending
        #
        # TODO: enforce a stronger check here (user story #179)
        try:
            with open(training_plan_file, "r") as f:
                content = f.read()
            minify(content,
                   remove_annotations=False,
                   combine_imports=False,
                   remove_pass=False,
                   hoist_literals=False,
                   remove_object_base=True,
                   rename_locals=False)
        except Exception as e:
            # minify does not provide any specific exception
            logger.error(f"This file is not a python file ({e})")
            return {}

        # create a repository instance and upload the model file
        repository = Repository(environ['UPLOADS_URL'],
                                environ['TMP_DIR'],
                                environ['CACHE_DIR'])

        upload_status = repository.upload_file(training_plan_file)

        logger.debug(f"training_plan_approve: upload_status = {upload_status}")

        # send message to node(s)
        sequence = self._sequence  # store the sequence for reply filtering
        self._sequence += 1
        message = ResearcherMessages.request_create(
            {'researcher_id': environ['RESEARCHER_ID'],
             'description': str(description),
             'sequence': sequence,
             'training_plan_url': upload_status['file'],
             'command': 'approval'}).get_dict()

        if nodes:
            # send message to each node
            for n in nodes:
                self.messaging.send_message(message, client=n)
        else:
            # broadcast message
            self.messaging.send_message(message)

        # wait for answers for a certain timeout
        result = {}
        for resp in self.get_responses(look_for_commands=['approval'],
                                       timeout=timeout):
            if sequence != resp['sequence']:
                logger.error("received an approval_reply with wrong sequence, ignoring it")
                continue

            n = resp['node_id']
            s = resp['success']
            result[n] = s

            if s:
                logger.info(f"node ({n}) has correctly downloaded the model")
            else:
                logger.info(f"node ({n}) has not correctly downloaded the model")

        # print info to the user regarding the result
        if not result or not any(result.values()):
            logger.info("no nodes have acknowledged correct model reception before the timeout")

        # eventually complete the result with expected results
        # (if the message was sent to specific nodes)
        for n in nodes:
            if n not in result:
                result[n] = False
                logger.info(f"node ({n}) has not acknowledge model reception before the timeout")

        # return the result
        return result

    def add_monitor_callback(self, callback: Callable[[Dict], None]):
        """ Adds callback function for monitor messages

        Args:
            callback: Callback function for handling monitor messages that come due 'general/monitoring' channel
        """

        self._monitor_message_callback = callback

    def remove_monitor_callback(self):
        """ Removes callback function for Monitor class. """

        self._monitor_message_callback = None

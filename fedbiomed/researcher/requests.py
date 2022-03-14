"""
Implements the message exchanges from researcher to nodes
"""


import json
import tabulate
from time import sleep
from typing import Any, Dict, Callable
import uuid

from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedTaskQueueError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ResearcherMessages
from fedbiomed.common.messaging import Messaging
from fedbiomed.common.singleton import SingletonMeta
from fedbiomed.common.tasks_queue import TasksQueue

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.responses import Responses


class Requests(metaclass=SingletonMeta):
    """
    This class represents the requests addressed from Researcher to nodes.
    It creates a task queue storing reply to each incoming message.
    """
    def __init__(self, mess: Any = None):
        """
        Starts a message queue and reconfigures  message to be sent
        into a `Messaging` object.

        Args:
            mess (Any, optional): message to be sent. Defaults to None.
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
        """
        returns the messaging object
        """
        return(self.messaging)

    def on_message(self, msg: Dict[str, Any] , topic: str):
        """
        This handler is called by the Messaging class (Messager),
        when a message is received on researcher side.

        It is run in the communication process and must ba as quick as possible:
        - it deals with quick messages (eg: ping/pong)
        - it store the replies of the nodes to the task queue, the message will bee
        treated by the main (computing) thread.

        Args:
            msg (Dict[str, Any]): de-serialized msg
            topic (str)         : topic name (eg MQTT channel)
        """

        if topic == "general/logger":
            #
            # forward the teatment to node_log_handling() (same thread)
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


    def print_node_log_message(self, log: Dict[str, Any]):
        """
        print logger messages coming from the node

        It is run on the communication process and must be as quick as possible:
        - all logs (coming from the nodes) are forwarded to the researcher logger
        (immediate display on console/file/whatever)
        """

        # log contains the original message sent by the node
        # FIXME: we should use `fedbioed.common.json.deserialize` method
        # instead of the json method when extracting json message
        original_msg = json.loads(log["msg"])

        logger.info("log from: " +
                    log["node_id"] +
                    " / " +
                    log["level"] +
                    " - " +
                    original_msg["message"])

    def send_message(self, msg: dict, client=None):
        """
        Ask the messaging class to send a new message (receivers are
        deduced from the message content)

        Args:
            msg (dict): the message to send to nodes
            node ([str], optional): defines the channel to which the
                                message will be sent.
                                Defaults to None(all nodes)
        """
        logger.debug(str(environ['RESEARCHER_ID']))
        self.messaging.send_message(msg, client=client)

    def get_messages(self, commands: list = [], time: float = .0) -> Responses:
        """
        This method goes through the queue and gets messages with the
        specific command

        Args:
            command (list of str, optional): checks if message is containing the
            expecting command (the message  is discarded if it doesnot).
            Defaults to None (no command message checking, meaning all
            incoming messages are considered).
            time (float, optional): time to sleep in seconds before considering
            incoming messages. Defaults to .0.

        returns Reponses : `Responses` object containing the corresponding
        answers

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
        """
        waits for all nodes' answers, regarding a specific command
        returns the list of all nodes answers

        Args:
            look_for_commands (list): instruction that has been sent to
            node. Can be either ping, search or train.
            timeout (float, optional): wait for a specific duration
                before collecting nodes messages. Defaults to None.
                If set to None; uses value in global variable TIMEOUT
                instead.
            only_successful (bool, optional): deal only with messages
                that have been tagged as successful (ie with field
                `success=True`). Defaults to True.
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
                except Exception:
                    logger.error('Incorrect message received:' + str(resp))
                    pass

            if len(new_responses) == 0:
                "Timeout finished"
                break
            responses += new_responses
        return Responses(responses)

    def ping_nodes(self) -> list:
        """
        Pings online nodes
        :return: list of node_id
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
        """
        Searches available data by tags
        :param tags: Tuple containing tags associated to the data researcher
        is looking for.
        :nodes: optionally filter nodes with this list.
        Default : no filter, consider all nodes
        :return: a dict with node_id as keys, and list of dicts describing
        available data as values
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
        """
        Lists available data in each node

        Args:
            nodes (str): Listings datasets by given node ids
                            Default is none.
            verbose (bool): If it is true it prints datasets in readable format
        """

        # If nodes list is provided
        if nodes:
            for node in nodes:
                self.messaging.send_message(
                    ResearcherMessages.request_create({'researcher_id': environ['RESEARCHER_ID'],
                                                       "command": "list"}
                                                      ).get_dict() ,
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
                if len(data_found[node]) > 0 :
                    rows = [row.values() for row in data_found[node]]
                    headers = data_found[node][0].keys()
                    info = '\n Node: {} | Number of Datasets: {} \n'.format( node, len(data_found[node]))
                    logger.info(info + tabulate.tabulate(rows, headers, tablefmt="grid") + '\n')
                else:
                    logger.info('\n Node: {} | Number of Datasets: {}'.format( node, len(data_found[node])) +
                                " No data has been set up for this node.")

        return data_found

    def add_monitor_callback(self, callback: Callable[[Dict], None]):
        """
        Add callback function for monitor messages

        Args:
            callback (Callable): Callback function for handling monitor messages
                                 that comes through 'general/monitoring' channel
        """

        self._monitor_message_callback = callback

    def remove_monitor_callback(self):
        """
        Remove callback function for Monitor class. This method is called
        for canceling monitoring.  Currently it is used in Experiment when the
        tensorboard state is `False`. Since the request class is singleton there
        might be callback function already registered before (while running
        experiment on Notebook).
        """

        self._monitor_message_callback = None

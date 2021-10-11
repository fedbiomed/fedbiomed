from datetime import datetime
import json
import os
import signal
import sys
import threading
from time import sleep
from typing import Any, Dict
import uuid

from fedbiomed.common.logger import logger
from fedbiomed.common.message import ResearcherMessages
from fedbiomed.common.tasks_queue import TasksQueue, exceptionsEmpty
from fedbiomed.common.messaging import Messaging, MessagingType
from fedbiomed.researcher.environ import TIMEOUT, MESSAGES_QUEUE_DIR, RESEARCHER_ID, TMP_DIR, MQTT_BROKER, MQTT_BROKER_PORT
from fedbiomed.researcher.responses import Responses


class RequestMeta(type):
    """ This class is a thread safe singleton for Requests, a common design pattern
    for ensuring only one instance of each class using this metaclass
    is created in the process
    """

    _objects = {}
    _lock_instantiation = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """ Replace default class creation for classes using this metaclass,
        executed before the constructor
        """
        with cls._lock_instantiation:
            if cls not in cls._objects:
                object = super().__call__(*args, **kwargs)
                cls._objects[cls] = object
        return cls._objects[cls]


class Requests(metaclass=RequestMeta):
    """This class represents the requests addressed from Researcher to nodes.
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
        self.queue = TasksQueue(MESSAGES_QUEUE_DIR + '_' + str(uuid.uuid4()), TMP_DIR)

        if mess is None or type(mess) is not Messaging:
            self.messaging = Messaging(self.on_message,
                                       MessagingType.RESEARCHER,
                                       RESEARCHER_ID,
                                       MQTT_BROKER,
                                       MQTT_BROKER_PORT)
            self.messaging.start(block=False)
        else:
            self.messaging = mess

        # defines the sequence used for ping protocol
        self._sequence = 0

    def get_messaging(self) -> Messaging:
        """returns the messaging object
        """
        return(self.messaging)

    def on_message(self, msg: Dict[str, Any] , topic: str):
        """
        This handler is called by the Messaging class (Messager),
        when a message is received on researcher side.
        Adds to queue this incoming message.
        Args:
            msg (Dict[str, Any]): de-serialized msg
            topic (str)         : topic name (eg MQTT channel)
        """

        if topic == "general/logger":
            self.node_log_handling(ResearcherMessages.reply_create(msg).get_dict())
        elif topic == "general/server":
            self.queue.add(ResearcherMessages.reply_create(msg).get_dict())
        else:
            log.error("message received on wrong topic ("+ topic +") - IGNORING")


    def node_log_handling(self, log: Dict[str, Any]):
        """
        manage log/error handling
        """

        # log contains the original message sent by the node
        original_msg = json.loads(log["msg"])

        logger.info("log from: " +
                    log["client_id"] +
                    " - " +
                    log["level"] +
                    " " +
                    original_msg["message"])

        # deal with error/critical messages from a node
        node_msg_level = original_msg["level"]

        if node_msg_level == "ERROR" or node_msg_level == "CRITICAL":
            # first error  implementation: stop the researcher
            logger.critical("researcher stopped after receiving error/critical log from node: " + log["client_id"])
            os.kill(os.getpid(), signal.SIGTERM)


    def send_message(self, msg: dict, client=None):
        """
        Ask the messaging class to send a new message (receivers are
        deduced from the message content)

        Args:
            msg (dict): the message to send to nodes
            client ([str], optional): defines the channel to which the
                                message will be sent.
                                Defaults to None(all clients)
        """
        logger.debug(str(RESEARCHER_ID))
        self.messaging.send_message(msg, client=client)

    def get_messages(self, command: str = None, time: float = .0) -> Responses:
        """ This method goes through the queue and gets messages with the
        specific command

        Args:
            command (str, optional): checks if message is containing the
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

                if command is None or \
                        ('command' in item.keys() and item['command'] == command):
                    answers.append(item)
                else:
                    # currently trash all other messages
                    pass
                    #self.queue.add(item)
            except exceptionsEmpty:
                pass

        return Responses(answers)

    def get_responses(self,
                      look_for_command: str,
                      timeout: float = None,
                      only_successful: bool = True) -> Responses:
        """
        waits for all clients' answers, regarding a specific command
        returns the list of all clients answers

        Args:
            look_for_command (str): instruction that has been sent to
            node. Can be either ping, search or train.
            timeout (float, optional): wait for a specific duration
                before collecting nodes messages. Defaults to None.
                If set to None; uses value in global variable TIMEOUT
                instead.
            only_successful (bool, optional): deal only with messages
                that have been tagged as successful (ie with field
                `success=True`). Defaults to True.
        """
        timeout = timeout or TIMEOUT
        responses = []

        while True:
            sleep(timeout)
            new_responses = []
            for resp in self.get_messages(command=look_for_command, time=0):
                try:
                    if not only_successful:
                        new_responses.append(resp)
                    elif resp['success']:
                        new_responses.append(resp)
                except Exception:
                    logger.error('Incorrect message received:' + str(resp))
                    pass

            if len(new_responses) == 0:
                "Timeout finished"
                break
            responses += new_responses
        return Responses(responses)

    def ping_clients(self) -> list:
        """
        Pings online nodes
        :return: list of client_id
        """
        self.messaging.send_message(ResearcherMessages.request_create(
            {'researcher_id': RESEARCHER_ID,
             'sequence': self._sequence,
             'command':'ping'}).get_dict())
        self._sequence += 1

        # TODO: (below, above) handle exceptions
        clients_online = [resp['client_id'] for resp in self.get_responses(look_for_command='ping')]
        return clients_online

    def search(self, tags: tuple, clients: list = None) -> dict:
        """
        Searches available data by tags
        :param tags: Tuple containing tags associated to the data researcher
        is looking for.
        :clients: optionally filter clients with this list.
        Default : no filter, consider all clients
        :return: a dict with client_id as keys, and list of dicts describing
        available data as values
        """
        self.messaging.send_message(ResearcherMessages.request_create({'tags':tags, 'researcher_id':RESEARCHER_ID, "command": "search"}).get_dict())

        logger.info(f'Searching for clients with data tags: {tags}')
        data_found = {}
        for resp in self.get_responses(look_for_command='search'):
            # TODO: (below) handle KeyError exception or use `.get()` method
            if not clients or resp['client_id'] in clients:
                data_found[resp['client_id']] = resp['databases']
        return data_found

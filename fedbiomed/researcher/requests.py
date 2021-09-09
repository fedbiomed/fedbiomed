from time import sleep
from datetime import datetime
from threading import Lock
from queue import Queue, Empty
import threading

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
    _lock_instantiation = Lock()

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
    """This class represents the protocol-independent messaging layer for the researcher
    """    
    def __init__(self, mess: Messaging =None):
        """Constructor of the class.

        Args:
            mess (Messaging, optional): Messaging object to be used by the class. Defaults to None,
                Messaging will then be dynamically created
        """  
        # incoming message queue : used by underlying Messaging to report incoming messages 
        # - permanent across runs of the class
        # - shared between instances of the class
        self.queue = TasksQueue(MESSAGES_QUEUE_DIR + '_' + RESEARCHER_ID, TMP_DIR)

        # event queue : used by the underlying Messaging to report running events
        # - transient to a run of the class
        # - specific to an instance of the class
        self.event_queue = Queue()

        if mess is None or type(mess) is not Messaging:
            self.messaging = Messaging(self.on_message, self.on_event, MessagingType.RESEARCHER, \
                RESEARCHER_ID, MQTT_BROKER, MQTT_BROKER_PORT)
            self.messaging.start(block=False)
        else:
            self.messaging = mess

    def get_messaging(self):
        """returns the messaging object
        """        
        return(self.messaging)

    def on_message(self, msg: dict):
        """
        This handler is called by the Messaging class when a message is received
        Args: 
            msg(dict): de-serialized msg
        """
        print(datetime.now(), '[ RESEARCHER ] message received.', msg)
        print("---DEBUG---- on message", threading.current_thread().name, threading.current_thread().ident)
        self.queue.add(ResearcherMessages.reply_create(msg).get_dict())
        
    def on_event(self, event: dict):
        """
        This handler is called by the Messaging class when an event occurs in the messaging
        Args:
            event(dict): description of event
        """
        print(datetime.now(), '[ RESEARCHER ] event received.', event)
        print("---DEBUG---- on event", threading.current_thread().name, threading.current_thread().ident)
        self.event_queue.put(event)

    def send_message(self, msg: dict, client=None):      
        """
        ask the messaging class to send a new message (receivers are deduced from the message content)
        """
        self.messaging.send_message(msg, client=client)


    def get_messages(self, command=None, time=0):
        """ This method go through the queue and gets messages with the specific command

        returns Reponses : Dataframe containing the corresponding answers

        """
        sleep(time)

        print("---DEBUG---- get_messages", threading.current_thread().name, threading.current_thread().ident)
        answers = []

        # handle exceptional events if any
        for _ in range(self.event_queue.qsize()):
            try:
                event = self.event_queue.get(block=False)
            except Empty:
                print("[ERROR] Unexpected empty event queue in researcher")
                raise
            print('[ERROR] Requests handler received an event from Messaging ', event)
            raise SystemExit

        # handle received messages
        for _ in range(self.queue.qsize()):
            try:
                item = self.queue.get(block=False)
            except exceptionsEmpty:
                print("[ERROR] Unexpected empty message queue in researcher")
                raise
            self.queue.task_done()

            if command is None or \
                    ('command' in item.keys() and item['command'] == command):
                answers.append(item)
            else:
                # currently trash all other messages
                print("[INFO] Ignoring message ", item)
                pass

        return Responses(answers)


    def get_responses(self, look_for_command, timeout=None, only_successful=True):
        """
        wait for answers for all clients, regarding a specific command
        returns the list of all clients answers
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
                    print(datetime.now(),'[ RESEARCHER ] Incorrect message received.', resp)
                    pass

            if len(new_responses) == 0:
                "Timeout finished"
                break
            responses += new_responses
        return Responses(responses)


    def ping_clients(self):
        """
        Pings online nodes
        :return: list of client_id
        """
        self.messaging.send_message(ResearcherMessages.request_create({'researcher_id':RESEARCHER_ID, 'command':'ping'}).get_dict())
        clients_online = [resp['client_id'] for resp in self.get_responses(look_for_command='ping')]
        return clients_online


    def search(self, tags: tuple, clients: list=None):
        """
        Searches available data by tags
        :param tags: Tuple containing tags associated to the data researchir is looking for.
        :clients: optionally filter clients with this list. Default : no filter, consider all clients
        :return: a dict with client_id as keys, and list of dicts describing available data as values
        """
        self.messaging.send_message(ResearcherMessages.request_create({'tags':tags, 'researcher_id':RESEARCHER_ID, "command": "search"}).get_dict())

        print(f'Searching for clients with data tags: {tags} ...')
        data_found = {}
        for resp in self.get_responses(look_for_command='search'):
            if not clients or resp['client_id'] in clients:
                data_found[resp['client_id']] = resp['databases']
        return data_found

from time import sleep
import uuid
from datetime import datetime
from threading import Lock

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
    def __init__(self, mess=None):
        """Constructor of the class.

        Args:
            mess ([type], optional): [description]. Defaults to None.
        """
        # Need to ensure unique per researcher instance message queue to avoid conflicts 
        # in case several instances of researcher (with same researcher_id ?) are active,
        # eg: a notebook not quitted and launching a script
        self.queue = TasksQueue(MESSAGES_QUEUE_DIR + '_' + str(uuid.uuid4()), TMP_DIR)

        if mess is None or type(mess) is not Messaging:
            self.messaging = Messaging(self.on_message, MessagingType.RESEARCHER, \
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
        This handler is called by the Messaging class, then a message is received
        Args: 
            msg (dict): de-serialized msg
        """
        print(datetime.now(), '[ RESEARCHER ] message received.', msg)
        self.queue.add(ResearcherMessages.reply_create(msg).get_dict())

        

    def send_message(self, msg: dict, client=None):      
        """
        Ask the messaging class to send a new message (receivers are deduced from the message content)

        Args:
            msg (dict): message to be sent
            client (str) : the client id of the receiver
        """
        self.messaging.send_message(msg, client=client)


    def get_messages(self, command=None, time=0):
        """ This method go through the queue and gets messages with the specific command

        returns Reponses : Dataframe containing the corresponding answers

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


    def search(self, tags: tuple, clients: list = None):
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

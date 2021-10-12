import os
import shutil
from threading import Lock
from fedbiomed.researcher.environ import TENSORBOARD_RESULTS_DIR, MQTT_BROKER, MQTT_BROKER_PORT
from fedbiomed.common.messaging import Messaging, MessagingType
from fedbiomed.common.message import MonitorMessages 
from fedbiomed.common.logger import logger
from torch.utils.tensorboard import SummaryWriter

from typing import Dict, Any


class MonitorMeta(type):
    """ This class is a thread safe singleton for Monitor, a common design pattern
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
            else:
                # Change the tensorboard state with given new state if the singleton 
                # class has been already constructed 
                cls._objects[cls].reconstruct(kwargs['tensorboard'])

        return cls._objects[cls]


class Monitor(metaclass=MonitorMeta):

    """ This is the class that subscribes monitor channel and logs scalar values 
    using `logger`. It also writes scalar values to tensorboard log files. 
    """
    
    def __init__(self, tensorboard: bool = False):

        """ Constructor of the class.


        Args:
            tensorboard (bool):  Default is False. If it is true it will write scalar
                                 values recevied from node during traning to tensorboard
                                 log file. 
        """ 


        self._messaging = Messaging(self._on_message, MessagingType.MONITOR,
                                   'NodeTrainingFeedbackClient', MQTT_BROKER, MQTT_BROKER_PORT)
        
        # Start subscriber
        self._messaging.start(block=False)
        self._log_dir = TENSORBOARD_RESULTS_DIR
        self.tensorboard = tensorboard
        self.round = 0
        self._event_writers = {}

        if self.tensorboard:
            if not os.listdir(self._log_dir):
                logger.info('Removing tensorboard logs from previous experiment')
                # Clear logs directory from the files from other experiments.
                shutil.rmtree(self._log_dir)
                
    def _on_message(self, msg: Dict[str, Any], topic: str):

        """Handler to be used with `Messaging` class (ie with messager).
        It is called when a  messsage arrive through the messager
        It reads and triggers instruction received by Monitor from Node,
        - Monitoring requests that comes from node during training

        Args:
            msg (Dict[str, Any]): incoming message from Node.
            Must contain key named `command`, describing the nature
            of the command (currently the command is only add_scalar).
            topic (str): topic name (eg MQTT channel)
        """

        # Check command whether is scalar

        scalar = MonitorMessages.reply_create(msg).get_dict()

        if scalar['command'] == 'add_scalar':
            if self.tensorboard:
                self._summary_writer(msg['client_id'], 
                                     msg['key'],   
                                     msg['iteration'],   
                                     msg['value'],
                                     msg['epoch'] 
                )
    
    def _summary_writer(self, client: str, key: str, global_step: int, scalar: float, epoch: int ):

        """ This method is for writing scalar values using torch SummaryWriter
        It create new summary path for each node
        
        Args:
            client (str): node id that sends 
            key (str): Name of the scalar value it can be e.g. loss, accuracy
            global_step (int): The index of the current batch proccess during epoch.
                               Batch is all samples if it is -1.  
            scalar (float): The value that be writen into tensorboard logs: loss, accuracy etc.
            epoch (int): Epoch during training routine
        """

        # Initilize event SummaryWriters
        if client not in self._event_writers:
            self._event_writers[client] = { 
                                    'writer' : SummaryWriter(log_dir=self._log_dir + '/NODE-' + client), 
                                    'stepper': 0,
                                    'step_state': 0,
                                    'step': 0
                                    }
            
        # Means that batch is equal to all samples use epoch as global step 
        if global_step == -1:
            global_step = epoch

        # Operations for finding log interval for the training 
        if global_step != 0 and self._event_writers[client]['stepper'] == 0:
            self._event_writers[client]['stepper'] = global_step 

        if global_step == 0:
            self._event_writers[client]['step_state'] = self._event_writers[client]['step'] + \
                                                        self._event_writers[client]['stepper']

        self._event_writers[client]['step'] = self._event_writers[client]['step_state'] + global_step 

        self._event_writers[client]['writer'].add_scalar('Metric[{}]'.format( 
                                                            key ), 
                                                            scalar,  
                                                            self._event_writers[client]['step'])


    def reconstruct(self, tensorboard: bool):
        
        """This method is used for changing tensorboard in case of rebuilding Singleton class. 
        It will update tensorboard state and remove tensorboard log files from 
        previous experiment. It is necessary for runing building eperiment in notebook 
        multiple times. 
        """

        self.tensorboard = tensorboard
        self._event_writers = {}
        # Remove tensorboard files from previous experiment
        if os.path.exists(self._log_dir):
           logger.info('Removing tensorboard logs from previous experiment')      
           shutil.rmtree(self._log_dir)


    def close_writer(self):
        
        """Stops `SummaryWriter` for each node of the experiment"""

        # Bring back the round to ist default value
        self.round = 0

        # Close each open SummaryWriter
        for node in self._event_writers:
            self._event_writers[node]['writer'].close() 


    def increase_round(self):
        
        """ This method increase the round based on the rounds of the experiment
            It is called after each round loop. 
        """
        self.round += 1

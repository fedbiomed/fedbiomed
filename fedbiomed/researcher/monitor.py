import os
import shutil
from threading import Lock
from fedbiomed.node.environ import  TMP_DIR, MQTT_BROKER, MQTT_BROKER_PORT
from fedbiomed.common.messaging import Messaging, MessagingType
from fedbiomed.common.message import MonitorMessages 
from fedbiomed.common.logger import logger
from torch.utils.tensorboard import SummaryWriter


class MonitorMeta(type):
    """ This class is a thread safe singleton for Feedback, a common design pattern
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

        self._messaging = Messaging(self._on_message, MessagingType.MONITOR,
                                   'NodeTrainingFeedbackClient', MQTT_BROKER, MQTT_BROKER_PORT)
        
        # Start subscriber
        self._messaging.start(block=False)
        self._log_dir = TMP_DIR + '/tensorboard'
        self.tensorboard = tensorboard
        self.round = 0
        self._event_writers = {}
        self._step = 0
        self._step_state = 0
        self._stepper = 0


        # Clear logs directory from the files from other experiments.
        shutil.rmtree(self._log_dir)

        if self.tensorboard and not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
 
    def _on_message(self, msg):

        """Handler to be used with `Messaging` class (ie with messager).
        It is called when a  messsage arrive through the messager
        It reads and triggers instruction received by Monitor from Node,
        - Monitoring requests that comes from node during training

        Args:
            msg (Dict[str, Any]): incoming message from Node.
            Must contain key named `command`, describing the nature
            of the command (currently the command is only add_scalar).
        """

        # Check command whether is scalar

        scalar = MonitorMessages.reply_create(msg).get_dict()

        if scalar['command'] == 'add_scalar':

            # Print out scalar values    
            self._log_to_console(msg)

            if self.tensorboard:
                self._summary_writer(msg['client_id'], 
                                     msg['key'],   
                                     msg['iteration'],   
                                     msg['value'],
                                     msg['epoch'] 
                )
    

    def _log_to_console(self, msg):

        """ This method is for loging scalar values into console by using
        logger.
        """

        # Logging training feedback
        logger.info('Round: {} Node: {} - Train Epoch: {} [Batch {} ]\t{}: {:.6f}'.format(
                        str(self.round),
                        msg['client_id'],
                        msg['epoch'],
                        msg['iteration'],
                        msg['key'],
                        msg['value']))

    
    def _summary_writer(self, client: str, key: str, global_step: int, scalar: float, epoch: int ):

        """ This method is for writing scalar values using torch SummaryWriter
        It create new summary path for each round of traning
        """
        

        if client not in self._event_writers:
             self._event_writers[client] = SummaryWriter(log_dir=self._log_dir + '/NODE-' + client)
        

        # Operations for finding log interval 
        if  global_step != 0 and self._stepper == 0:
            self._stepper = global_step 

        if global_step == 0:
            self._step_state = self._step + self._stepper 

        self._step = self._step_state + global_step 

        self._event_writers[client].add_scalar('Metric[{}]'.format( 
                                                            key ), 
                                                            scalar,  
                                                            self._step)

        #self._event_writers[client].close()

    def reconstruct(self, tensorboard: bool):
        
        """This method is used for changing tensorboard in case of rebuilding Singleton class. 
        It will update tensorboard state and remove tensorboard log files from 
        previous experiment. 
        """
        self.tensorboard = tensorboard

        # Remove tensorboard files from previous experiment
        shutil.rmtree(self._log_dir)


    def close_writer(self):
        
        """Stops `SummaryWriter` if the experiment started with 
        option `tensorbroad=True`  
        """

        # Bring back the round and round state at the begening
        self.round = 0

        # Close each open event writer
        for writer in self._event_writers:
            self._event_writers[writer].close() 


    def increase_round(self):
        
        """ This method increase the round based on the rounds of the experiment
            It is called after each round loop. 
        """
        self.round += 1

import os
import shutil
from threading import Lock
from typing import Any, Callable, Union
from fedbiomed.node.environ import CLIENT_ID, TMP_DIR, MQTT_BROKER, MQTT_BROKER_PORT
from fedbiomed.common.messaging import Messaging, MessagingType 
from fedbiomed.common.logger import logger
from torch.utils.tensorboard import SummaryWriter


class FeedbackMeta(type):
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
                # If there is a class already constructed change tensorboard and verbose with given new state
                cls._objects[cls].reconstruct(kwargs['tensorboard'])

        return cls._objects[cls]


class Feedback(metaclass=FeedbackMeta):

    """ This is the class that subscribes feedback channel and logs scalar values 
    using `logger`. It also writes scalar values to tensorboard log files. 
    """
    
    def __init__(self, tensorboard: bool = False):

        self._messaging = Messaging(self.on_message, MessagingType.FEEDBACK,
                                   'NodeTrainingFeedbackClient', MQTT_BROKER, MQTT_BROKER_PORT)
        
        # Start subscriber
        self._messaging.start(block=False)
        self._log_dir = TMP_DIR + '/tensorboard'
        self.tensorboard = tensorboard
        self.round = 0
        self._round_state = 0

        # Clear logs directory from the files from other experiments.
        shutil.rmtree(self._log_dir)

        if self.tensorboard:
            self._summary = SummaryWriter(self._log_dir + '/Round-0')



    def on_message(self, msg):

        """Handler to be used with `Messaging` class (ie with messager).
        It is called when a  messsage arrive through the messager
        It reads and triggers instruction received by node from Researcher,
        - feddback requests that comes from node during traning

        Args:
            msg (Dict[str, Any]): incoming message from Node.
            Must contain key named `command`, describing the nature
            of the command (currently the command is only add_scalar).
        """

        node = msg['client_id']
        res = msg['res']

        if msg['command'] == 'add_scalar':
            # Logging training feedback
            logger.info('Round: {} Node: {} - Train Epoch: {} [{}/{}]\t{}: {:.6f}'.format(
                            str(self.round),
                            msg['client_id'],
                            msg['res']['epoch'],
                            msg['res']['iteration']*msg['res']['num_batch'],
                            msg['res']['len_data'],
                            msg['res']['key'],
                            msg['res']['value']))

            if self.tensorboard:
                self.summary_writer(msg)
    
    
    def summary_writer(self, msg):

        """ This method is for writing scalar values using torch SummaryWriter
        It create new summary path for each round of traning
        """

        # Control if round has been changed
        if self._round_state != self.round:
            self._summary.close()
            self._round_state = self.round
            self._summary = SummaryWriter(self._log_dir + '/Round-' + str(self.round))

        # Add scalar to tensorboard
        self._summary.add_scalar( 'Node-{}/Metric[{}]'.format(
                            msg['client_id'], 
                            msg['res']['key'] ), 
                            msg['res']['value'], 
                            msg['res']['iteration']*msg['res']['epoch'],
                            )
    


    def reconstruct(self, tensorboard: bool):
        
        """This method is used for changing tensorboard and verbose 
        state in case of rebuilding Singleton class. It will update only 
        tensorboard and verbose states.   
        """
        self.tensorboard = tensorboard

        # Remove tensorboard files from previous experiment
        shutil.rmtree(self._log_dir)

        self._summary = SummaryWriter(self._log_dir + '/Round-0')

    def close_writer(self):
        
        """Stops `SummaryWriter` if the experiment started with 
        option `tensorbroad=True`  
        """

        # Bring back the round and round state at the begening
        self.round = 0
        self._round_state = 0

        if self._summary:
            self._summary.close()
        else: 
            logger.warning('TensorBoard has not been activated for the experiment')

    def flush_summary(self):

        # Flush summary files
        if self._summary:
            self._summary = SummaryWriter(self._log_dir)
        else:
            logger.warning('TensorBoard has not been activated for the experiment')

    def increase_round(self):
        
        """ This method increase the round based on the rounds of the experiment
            It is called after each round loop. 
        """
        self.round += 1

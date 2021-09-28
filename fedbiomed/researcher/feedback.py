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
                cls._objects[cls]._switcher(kwargs['tensorboard'], kwargs['verbose'])

        return cls._objects[cls]


class Feedback(metaclass=FeedbackMeta):

    """ This is the class that subscribes feedback channel and logs scalar values 
    using `logger`. It also writes scalar values to tensorboard log files. 
    """
    
    def __init__(self, 
                 tensorboard: bool = False,
                 verbose: bool = False,
                 job: str = None):

        self.messaging = Messaging(self.on_message, MessagingType.FEEDBACK,
                                   'NodeTrainingFeedbackClient', MQTT_BROKER, MQTT_BROKER_PORT)
        
        # Start subscriber
        self.messaging.start(block=False)

        self._tensorboard = tensorboard
        self._verbose = verbose
        self._round = 0
        self._job = job  

        if self._tensorboard:
            self.summary = SummaryWriter(TMP_DIR + '/tensorboard/job-' + self._job)


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
            if self._verbose: 
                logger.info('Round: {} Node: {} - Train Epoch: {} [{}/{}]\t{}: {:.6f}'.format(
                                str(self._round),
                                node,
                                res['epoch'],
                                res['iteration']*res['num_batch'],
                                res['len_data'],
                                res['key'],
                                res['value']))

            if self._tensorboard:
                self.summary.add_scalar( 'Node-{}/Round-{}/{}'.format(
                                            node, 
                                            str(self._round), 
                                            res['key'] ), 
                                            res['value'], 
                                            res['iteration']*
                                            res['epoch'])
    
  
    def _switcher(self, tensorboard: bool, verbose: bool):
        
        """This method is used for changing tensorboard and verbose 
        state in case of rebuilding Singleton class. It will update only 
        tensorboard and verbose states.   
        """
        
        self._tensorboard = tensorboard
        self._verbose = verbose



    def close_writer(self):
        
        """Stops `SummaryWriter` if the experiment started with 
        option `tensorbroad=True`  
        """

        # Bring back the round at the begening
        self._round = 0

        if self.summary:
            self.summary.close()
        else: 
            logger.warning('TensorBoard has not been activated for the experiment')

    def flush_summary(self):

        # Flush summary files
        if self.summary:
            self.summary.flush()
        else:
            logger.warning('TensorBoard has not been activated for the experiment')

    def increase_round(self):
        
        """ This method increase the round based on the rounds of the experiment
            It is called after each round loop. 
        """
        self._round += 1

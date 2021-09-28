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

        self._messaging = Messaging(self.on_message, MessagingType.FEEDBACK,
                                   'NodeTrainingFeedbackClient', MQTT_BROKER, MQTT_BROKER_PORT)
        
        # Start subscriber
        self._messaging.start(block=False)

        self.tensorboard = tensorboard
        self.verbose = verbose
        self.round = 0
        self._job_id = job  

        if self.tensorboard:
            self._summary = SummaryWriter(TMP_DIR + '/tensorboard/job-' + self._job_id)


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
            if self.verbose: 
                logger.info('Round: {} Node: {} - Train Epoch: {} [{}/{}]\t{}: {:.6f}'.format(
                                str(self.round),
                                node,
                                res['epoch'],
                                res['iteration']*res['num_batch'],
                                res['len_data'],
                                res['key'],
                                res['value']))

            if self.tensorboard:
                self._summary.add_scalar( 'Node-{}/Round-{}/{}'.format(
                                            node, 
                                            str(self.round), 
                                            res['key'] ), 
                                            res['value'], 
                                            res['iteration']*
                                            res['epoch'])
    
  
    def _switcher(self, tensorboard: bool, verbose: bool):
        
        """This method is used for changing tensorboard and verbose 
        state in case of rebuilding Singleton class. It will update only 
        tensorboard and verbose states.   
        """

        self.tensorboard = tensorboard
        self.verbose = verbose



    def close_writer(self):
        
        """Stops `SummaryWriter` if the experiment started with 
        option `tensorbroad=True`  
        """

        # Bring back the round at the begening
        self.round = 0

        if self._summary:
            self._summary.close()
        else: 
            logger.warning('TensorBoard has not been activated for the experiment')

    def flush_summary(self):

        # Flush summary files
        if self._summary:
            self._summary.flush()
        else:
            logger.warning('TensorBoard has not been activated for the experiment')

    def increase_round(self):
        
        """ This method increase the round based on the rounds of the experiment
            It is called after each round loop. 
        """
        self.round += 1

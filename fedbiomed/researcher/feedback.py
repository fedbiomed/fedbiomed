import os
import shutil
import time
from threading import Lock
from fedbiomed.node.environ import  TMP_DIR, MQTT_BROKER, MQTT_BROKER_PORT
from fedbiomed.common.messaging import Messaging, MessagingType
from fedbiomed.common.message import FeedbackMessages 
from fedbiomed.common.logger import logger
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


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
                # Change the tensorboard state with given new state if the class has been already constructed 
                cls._objects[cls].reconstruct(kwargs['tensorboard'])

        return cls._objects[cls]


class Feedback(metaclass=FeedbackMeta):

    """ This is the class that subscribes feedback channel and logs scalar values 
    using `logger`. It also writes scalar values to tensorboard log files. 
    """
    
    def __init__(self, tensorboard: bool = False):

        self._messaging = Messaging(self._on_message, MessagingType.FEEDBACK,
                                   'NodeTrainingFeedbackClient', MQTT_BROKER, MQTT_BROKER_PORT)
        
        # Start subscriber
        self._messaging.start(block=False)
        self._log_dir = TMP_DIR + '/tensorboard'
        self.tensorboard = tensorboard
        self.round = 0
        self._event_writers = {}
        self._global_step = 0

        # Clear logs directory from the files from other experiments.
        shutil.rmtree(self._log_dir)

        if self.tensorboard and not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
 
    def _on_message(self, msg):

        """Handler to be used with `Messaging` class (ie with messager).
        It is called when a  messsage arrive through the messager
        It reads and triggers instruction received by node from Researcher,
        - feddback requests that comes from node during training

        Args:
            msg (Dict[str, Any]): incoming message from Node.
            Must contain key named `command`, describing the nature
            of the command (currently the command is only add_scalar).
        """

        # Check command whether is scalar

        scalar = FeedbackMessages.reply_create(msg).get_dict()

        if scalar['command'] == 'add_scalar':

            # Print out scalar values    
            self._log_to_console(msg)

            if self.tensorboard:
                self._summary_writer(msg)
    

    def _log_to_console(self, msg):

        """ This method is for loging scalar values into console by usign
        logger.
        """

        # Logging training feedback
        logger.info('Round: {} Node: {} - Train Epoch: {} [{}/{}]\t{}: {:.6f}'.format(
                        str(self.round),
                        msg['client_id'],
                        msg['res']['epoch'],
                        msg['res']['iteration']*msg['res']['num_batch'],
                        msg['res']['len_data'],
                        msg['res']['key'],
                        msg['res']['value']))

    
    def _summary_writer(self, msg):

        """ This method is for writing scalar values using torch SummaryWriter
        It create new summary path for each round of traning
        """
        if msg['client_id'] not in self._event_writers:
            self._event_writers[msg['client_id']] = self._SummaryWriter(self._log_dir + '/NODE-' + msg['client_id'])
    

        self._event_writers[msg['client_id']].add_scalar('Metric[{}]'.format( 
                                                            msg['res']['key'] ), 
                                                            msg['res']['value'],  
                                                            msg['res']['iteration'],
                                                            msg['res']['epoch'],
                                                            self.round
                                                            )

    def reconstruct(self, tensorboard: bool):
        
        """This method is used for changing tensorboard in case of rebuilding Singleton class. 
        It will update only tensorboard state, remove tensorboard log files from 
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


    class _SummaryWriter():

        """ SummaryWriter for scalar values that comes from each client"""

        def __init__(self, log_dir, flush_secs=120,  filename_suffix=''):
            self._event_writer = EventFileWriter(log_dir, 
                                                 flush_secs=flush_secs, 
                                                 filename_suffix= filename_suffix) 
            self._step = 0
            self._step_state = 0
            self._stepper = None

        def add_scalar(self, tag: str, scalar: float, global_step: int,  epoch: int, round: int, walltime: time = None):

            """ Add scalar to summary using event writer"""
            if global_step == 0:
                self._step_state = self._step 

            self._step = self._step_state + global_step 
                
                 
            summary = Summary(value=[Summary.Value(tag=tag, simple_value=scalar)])
            event = event_pb2.Event(summary=summary)
            # Time that scalar is writen
            event.wall_time = time.time() if walltime is None else walltime
            
            if global_step is not None:
                event.step = int(self._step)

            self._event_writer.add_event(event)

        def close(self):

            """Method for closing event writer"""   
            self._event_writer.close()


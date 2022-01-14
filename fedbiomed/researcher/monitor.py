import os
import shutil
from fedbiomed.researcher.environ import environ
from fedbiomed.common.logger import logger
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any


class Monitor():

    """ This is the class that subscribes monitor channel and logs scalar values
    using `logger`. It also writes scalar values to tensorboard log files.
    """

    def __init__(self):

        """ Constructor of the class. Intialize empty event writers object and
        logs directory. Removes tensorboard logs from previous experiments.
        """

        self._log_dir = environ['TENSORBOARD_RESULTS_DIR']
        self._event_writers = {}

        if os.listdir(self._log_dir):
            logger.info('Removing tensorboard logs from previous experiment')
            # Clear logs directory from the files from other experiments.
            self._remove_logs()

    def on_message_handler(self, msg: Dict[str, Any]):

        """Handler for messages received through general/monitoring channel.
        This method is used as callback function in Requests class

        Args:
            msg (Dict[str, Any]): incoming message from Node.
            Must contain key named `command`, describing the nature
            of the command (currently the command is only add_scalar).
        """

        # For now monitor can only handle add_scalar messages
        if msg['command'] == 'add_scalar':
                self._summary_writer(msg['node_id'],
                                        msg['key'],
                                        msg['iteration'],
                                        msg['value'],
                                        msg['epoch'] )


    def _summary_writer(self, node: str, key: str, global_step: int, scalar: float, epoch: int ):

        """ This method is for writing scalar values using torch SummaryWriter
        It creates new summary file for each node.

        Args:
            node (str): node id that sends
            key (str): Name of the scalar value it can be e.g. loss, accuracy
            global_step (int): The index of the current batch proccess during epoch.
                               Batch is all samples if it is -1.
            scalar (float): The value that be writen into tensorboard logs: loss, accuracy etc.
            epoch (int): Epoch during training routine
        """

        # Initialize event SummaryWriters
        if node not in self._event_writers:
            self._event_writers[node] = {
                                    'writer' : SummaryWriter(
                                        log_dir = os.path.join(self._log_dir, node)
                                    ),
                                    'stepper': 1,
                                    'step_state': 0,
                                    'step': 0
                                    }

        # Means that batch is equal to all samples, use epoch as global step
        if global_step == -1:
            global_step = epoch

        # Operations for finding iteration log interval for the training
        if global_step != 0 and self._event_writers[node]['stepper'] <= 1:
            self._event_writers[node]['stepper'] = global_step

        # In every epoch first iteration (global step) will be zero so
        # we need to update step_state to not to overwrite steps of
        # the previous  epochs
        if global_step == 0:
            self._event_writers[node]['step_state'] = self._event_writers[node]['step'] + \
                                                        self._event_writers[node]['stepper']

        # Increase step by adding global_step to step_state
        self._event_writers[node]['step'] = self._event_writers[node]['step_state'] + global_step

        self._event_writers[node]['writer'].add_scalar('Metric[{}]'.format(
                                                            key ),
                                                            scalar,
                                                            self._event_writers[node]['step'])

    def close_writer(self):

        """Closes `SummaryWriter` for each node """

        # Close each open SummaryWriter
        for node in self._event_writers:
            self._event_writers[node]['writer'].close()

    def _remove_logs(self):

        """ This is private method for removing logs files from
        tensorboard logs dir.
        """

        for file in os.listdir(self._log_dir):
            rf = os.path.join(self._log_dir, file)
            if os.path.isdir(rf):
                shutil.rmtree(rf)
            elif os.path.isfile(rf):
                os.remove(rf)

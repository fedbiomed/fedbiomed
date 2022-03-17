'''
monitor class to trap information sent during training and
sned it to tensordboard
'''


import os
import shutil
from typing import Dict, Any

from torch.utils.tensorboard import SummaryWriter

from fedbiomed.common.logger import logger
from fedbiomed.researcher.environ import environ


class Monitor():
    """
    This is the class that subscribes monitor channel and logs scalar values
    using `logger`. It also writes scalar values to tensorboard log files.
    """

    def __init__(self):
        """
        Constructor of the class. Intialize empty event writers object and
        logs directory. Removes tensorboard logs from previous experiments.
        """

        self._log_dir = environ['TENSORBOARD_RESULTS_DIR']
        self._event_writers = {}
        self._round_state = 0
        self._tensorboard = False

        if os.listdir(self._log_dir):
            logger.info('Removing tensorboard logs from previous experiment')
            # Clear logs directory from the files from other experiments.
            self._remove_logs()

    def on_message_handler(self, msg: Dict[str, Any]):
        """
        Handler for messages received through general/monitoring channel.
        This method is used as callback function in Requests class

        Args:
            msg (Dict[str, Any]): incoming message from Node.
            Must contain key named `command`, describing the nature
            of the command (currently the command is only add_scalar).
        """

        # For now monitor can only handle add_scalar messages
        if msg['command'] == 'add_scalar':

            if self._tensorboard:
                # transfert data to tensorboard
                self._summary_writer(msg['node_id'],
                                     msg['key'],
                                     msg['iteration'],
                                     msg['value'],
                                     msg['epoch'] )

            else:
                # log on console
                msg = "Monitor: node_id=" + \
                    msg['node_id'] + \
                    " epoch=" + \
                    str(msg['epoch']) + \
                    " iteration=" + \
                    str(msg['iteration']) + \
                    " " + \
                    msg['key'] + \
                    ":" + \
                    str(msg['value'])
                logger.info(msg)


    def set_tensorboard(self, tensorboard: bool):
        """
        setter for the tensorboard flag, which is used to decide the behavior
        of the monitor callback

        Args:
            tensorboard: if True, data contained in AddScalarReply message
                         will be passed to tensorboard
                         if False, fata will only be logged on the console
        """
        if isinstance(tensorboard, bool):
            self._tensorboard = tensorboard
        else:
            logger.error("tensorboard should be a boolean")
            self._tensorboard = False


    def _summary_writer(self, node: str, key: str, global_step: int, scalar: float, epoch: int ):
        """
        This method is for writing scalar values using torch SummaryWriter
        It creates new summary file for each node.

        Args:
            node (str): node id that sends
            key (str): Name of the scalar value it can be e.g. loss, accuracy
            global_step (int): The index of the current batch proccess during epoch.
                               Batch is all samples if it is -1.
            scalar (float): The value that be writen into tensorboard logs: loss, accuracy etc.
            epoch (int): Number of epoch achieved during training routine
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

        if global_step == -1:
            # Means that batch is equal to all samples, use epoch as global step
            global_step = epoch

        # Operations for finding iteration log interval for the training
        #
        # stepper    : interval for each batch of points between 2 rounds or epochs
        #            : (should be equals to 1 when passing from one epoch to another
        #               or a round to another, otherwise 0 for batch of points
        #               within the same epoch)
        # global step: index of the batch size/batch_size (for the current round)
        # step_state : number of previous steps that compose the previous rounds
        #            : Ex:
        # step       : index for loss / metric values that will be used when displaying on
        #              tensorboard. It represents number of batches processed for
        #              training model
        if global_step != 0 and self._event_writers[node]['stepper'] < 2:
            self._event_writers[node]['stepper'] = 0

        elif global_step == 0:
            self._event_writers[node]['stepper'] = 1

        # In every epoch first iteration (global step) will be zero so
        # we need to update step_state so we do not overwrite steps of
        # the previous  epochs
        if global_step == 0:
            self._event_writers[node]['step_state'] = self._event_writers[node]['step'] + \
                self._event_writers[node]['stepper']

        # Increase step by adding global_step to step_state
        self._event_writers[node]['step'] = self._event_writers[node]['step_state'] + global_step

        self._event_writers[node]['writer'].add_scalar('Metric[{}]'.format(key),
                                                       scalar,
                                                       self._event_writers[node]['step'])

    def close_writer(self):
        """
        Closes `SummaryWriter` for each node
        """
        # Close each open SummaryWriter
        for node in self._event_writers:
            self._event_writers[node]['writer'].close()

    def _remove_logs(self):
        """
        This is private method for removing logs files from
        tensorboard logs dir.
        """

        for file in os.listdir(self._log_dir):
            if not file.startswith('.'): # dont want to remove dotfiles
                rf = os.path.join(self._log_dir, file)
                if os.path.isdir(rf):
                    shutil.rmtree(rf)
                elif os.path.isfile(rf):
                    os.remove(rf)

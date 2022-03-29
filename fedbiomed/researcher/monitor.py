'''
monitor class to trap information sent during training and
sned it to tensordboard
'''

import os
import shutil
import collections
from typing import Dict, Union, Any

from torch.utils.tensorboard import SummaryWriter

from fedbiomed.common.logger import logger
from fedbiomed.researcher.environ import environ


class _MetricStore(dict):

    def add_iteration(self,
                      node: str,
                      train: bool,
                      test_on_global_updates: bool,
                      round_: int,
                      metric: dict,
                      iter_: int):
        """
        Method adding iteration to MetricStore based on node, training/testing, round and metric.

        Args:
            node (str):
            train (bool):
            test_on_global_updates (bool):
            round_ (int):
            metric (dict):
            iter_ (int)
        """
        if node not in self:
            self._register_node(node=node)

        cum_iter = []
        for metric_name, metric_value in metric.items():

            for_ = 'training' if train is True else 'testing_global_updates' \
                if test_on_global_updates is True else 'testing_local_updates'

            if metric_name not in self[node][for_]:
                self._register_metric(node=node, for_=for_, metric_name=metric_name)

            # FIXME: for now, if testing is done on global updates (before model local update)
            # last testing metric value computed on global updates at last round is overwritten
            # by the first one computed at first round 
            if round_ in self[node][for_][metric_name]:
                self[node][for_][metric_name][round_]['iterations'].append(iter_)
                self[node][for_][metric_name][round_]['values'].append(metric_value)
            else:
                self[node][for_][metric_name].update({round_: {'iterations': [iter_],
                                                               'values': [metric_value]}
                                                      })
            cum_iter.append(self._cumulative_iteration(self[node][for_][metric_name]))

        return cum_iter

    def _register_node(self, node):
        """
        Register node for the first by initializing basic information on metrics
        """
        self[node] = {
            "training": {},
            "testing_global_updates": {},  # Testing before training
            "testing_local_updates": {}  # Testing after training
        }

    def _register_metric(self, node, for_, metric_name):
        """
        Method for registering metric for the given node. It creates stating point
        for the metric from round 0.

        Args: node
        """

        # Round should start from 1 to match experiment's starting round
        self[node][for_].update({metric_name: {1: {'iterations': [], 'values': []}}})

    @staticmethod
    def _cumulative_iteration(rounds):
        """

        """
        cum_iteration = 0
        for val in rounds.values():
            if len(val['iterations']):
                iter_frequencies = collections.Counter(val['iterations'])
                last_iteration = val['iterations'][-1]
                max_iter_value = max(val['iterations'])
                cum_iteration += (max_iter_value * (iter_frequencies[last_iteration] - 1)) + last_iteration

        return cum_iteration


class Monitor:
    """
    This is the class that subscribes monitor channel and logs scalar values
    using `logger`. It also writes scalar values to tensorboard log files.
    """

    def __init__(self):
        """
        Constructor of the class. Initializes empty event writers object and
        logs directory. Removes tensorboard logs from previous experiments.
        """

        self._log_dir = environ['TENSORBOARD_RESULTS_DIR']
        self._round = 1
        self._metric_store = _MetricStore()
        self._event_writers = {}
        self._round_state = 0
        self._tensorboard = False

        if os.listdir(self._log_dir):
            logger.info('Removing tensorboard logs from previous experiment')
            # Clear logs' directory from the files from other experiments.
            self._remove_logs()

    def set_round(self, round_: int):
        """

        """
        self._round = round_

        return self._round

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
            # Save iteration value
            cumulative_iter, *_ = self._metric_store.add_iteration(
                node=msg['node_id'],
                train=msg['train'],
                test_on_global_updates=msg['test_on_global_updates'],
                metric=msg['metric'],
                round_=self._round,
                iter_=msg['iteration'])

            # Log metric result
            self._log_metric_result(message=msg, cum_iter=cumulative_iter)

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

    def close_writer(self):
        """
        Closes `SummaryWriter` for each node
        """
        # Close each open SummaryWriter
        for node in self._event_writers:
            self._event_writers[node].close()

    def _remove_logs(self):
        """
        This is private method for removing logs files from
        tensorboard logs dir.
        """

        for file in os.listdir(self._log_dir):
            if not file.startswith('.'):  # do not want to remove dotfiles
                rf = os.path.join(self._log_dir, file)
                if os.path.isdir(rf):
                    shutil.rmtree(rf)
                elif os.path.isfile(rf):
                    os.remove(rf)

    def _log_metric_result(self, message: Dict, cum_iter: int = 0):
        """
        Method for loging metric result that comes from nodes
        """

        if message['train'] is True:
            header = 'Training'
        else:
            header = 'Testing On Global Updates' if message['test_on_global_updates'] else 'Testing On Local ' \
                                                                                              'Updates'

        metric_dict = message['metric']
        metric_result = ''
        for key, val in metric_dict.items():
            metric_result += "\t\t\t\t\t {}: \033[1m{:.6f}\033[0m \n".format(key, val)

        # Loging fancy feedback for training
        logger.info("\033[1m{}\033[0m \n"
                    "\t\t\t\t\t NODE_ID: {} \n"
                    "\t\t\t\t\t{} Completed: {}/{} ({:.0f}%) \n {}"
                    "\t\t\t\t\t ---------".format(header.upper(),
                                                  message['node_id'],
                                                  '' if message['epoch'] is None else f" Epoch: {message['epoch']} |",
                                                  message['iteration'] * message['batch_samples'],
                                                  message['total_samples'],
                                                  100 * message['iteration'] / message['num_batches'],
                                                  metric_result))

        if self._tensorboard:
            # transfer data to tensorboard
            self._summary_writer(header=header.upper(),
                                 node=message['node_id'],
                                 metric=message['metric'],
                                 cum_iter=cum_iter)

    def _summary_writer(self,
                        header: str,
                        node: str,
                        metric: dict,
                        cum_iter: int):
        """
        This method is for writing scalar values using torch SummaryWriter
        It creates new summary file for each node.

        Args:
            header (str)
            node (str): node id that sends

        """

        # Initialize event SummaryWriters
        if node not in self._event_writers:
            self._event_writers[node] = SummaryWriter(log_dir=os.path.join(self._log_dir, node))

        for metric, value in metric.items():
            self._event_writers[node].add_scalar('{}/{}'.format(header.upper(), metric),
                                                 value,
                                                 cum_iter)

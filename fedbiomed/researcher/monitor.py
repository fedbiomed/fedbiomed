# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

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


class MetricStore(dict):
    """
    Storage facility, used for storing training loss and testing metric values, in order to
    display them on Tensorboard.
    Inheriting from a dictionary, providing methods to simplify queries and saving metric values.

    Storage architecture:
    ```python
    {<node>:
        {<for_>:
            {<metric_name>:
                {<round_>: { <iterations/values>: List[float] }
                }
            }
        }
    }
    ```
    Where:
    - `node`: node id
    - `for_`: either testing_global_updates, testing_local_updates, or training
    - `metric_name`: metric 's name. Custom or Custom_xxx if testing_step has been defined in TrainingPlan (custom metric)
    - `round_`: round number
    - `iterations`: index of iterations stored
    - `values`: metric value
    """

    def add_iteration(self,
                      node: str,
                      train: bool,
                      test_on_global_updates: bool,
                      round_: int,
                      metric: dict,
                      iter_: int) -> list[int]:
        """
        Method adding iteration to MetricStore based on node, training/validation, round and metric.

        Args:
            node: The node id that metric value received from
            train: Training status, If true metric value is for training, Otherwise for validation
            test_on_global_updates: If True metric value is for validation on global updates. Otherwise,
                for validation on local updates
            round_: The round that metric value has received at
            metric: Dictionary that contains metric names and their values e.g {'<metric-name>':<value>}
            iter_: Iteration number for validation/training.

        Returns
             List of cumulative iteration for each metric/validation result
        """

        if node not in self:
            self._register_node(node=node)

        cum_iter = []
        for metric_name, metric_value in metric.items():

            for_ = 'training' if train is True else 'testing_global_updates' \
                if test_on_global_updates is True else 'testing_local_updates'

            if metric_name not in self[node][for_]:
                self._register_metric(node=node, for_=for_, metric_name=metric_name)

            # FIXME: for now, if validation is done on global updates (before model local update)
            # last testing metric value computed on global updates at last round is overwritten
            # by the first one computed at first round
            if round_ in self[node][for_][metric_name]:

                # Each duplication means a new epoch for training, and it is not expected for
                # validation part. Especially for `testing_on_global_updates`. If there is a duplication
                # last value should overwrite
                duplicate = self._iter_duplication_status(round_=self[node][for_][metric_name][round_],
                                                          next_iter=iter_)
                if duplicate and test_on_global_updates:
                    self._add_new_iteration(node, for_, metric_name, round_, iter_, metric_value, True)
                else:
                    self._add_new_iteration(node, for_, metric_name, round_, iter_, metric_value)
            else:
                self._add_new_iteration(node, for_, metric_name, round_, iter_, metric_value, True)

            cum_iter.append(self._cumulative_iteration(self[node][for_][metric_name]))
        return cum_iter

    def _add_new_iteration(self,
                           node: str,
                           for_: str,
                           metric_name: str,
                           round_: int,
                           iter_: int,
                           metric_value: Union[int, float],
                           new_round: bool = False):
        """Adds new iteration based on `new_round` status. If the round is new for the metric it registers key round
        key and assigns iteration and metric value. Otherwise, appends iteration and metric value to the existing round.

        Args:
            node: The node id that metric value received from
            for_: One of (training, testing_global_updates, testing_local_updates). To indicate metric
                value belongs to which phase
            metric_name: Name of the metric to use as a key in MetricStore dict
            round_: The round that metric value has received at
            iter_: Iteration number
            metric_value: Value of the metric
            new_round: To indicate whether round should be created or new metric should append to the
                existing one. This is also enables overwriting round values when needed.
        """
        if new_round:
            self[node][for_][metric_name].update({round_: {'iterations': [iter_],
                                                           'values': [metric_value]}
                                                  })
        else:
            self[node][for_][metric_name][round_]['iterations'].append(iter_)
            self[node][for_][metric_name][round_]['values'].append(metric_value)

    @staticmethod
    def _iter_duplication_status(round_: dict, next_iter: int) -> bool:
        """ Finds out is there iteration duplication in rounds for the testing metrics.

        Args:
            round_: Dictionary that includes iteration numbers and values for single metric results
                belongs to a node and a phase (training/testing_global_update or testing_local_updates)
            next_iter: An integer indicates the iteration number for the next iteration that is going to be
                stored on the MetricStore

        Returns:
            Indicates whether is there a duplication in the round iterations
        """

        iterations = round_['iterations']
        if iterations:
            first_val = iterations[0]
            return True if next_iter == first_val else False
        else:
            return False  # No duplication

    def _register_node(self, node: str):
        """ Registers node for the first time (first iteration) by initializing basic information on metrics

        Adds the following fields to node entry:
        - training: loss values from training
        - testing_global_updates: metric values and names from validation on global updates
        - testing_local_updates: metric values and names from validation on local updates

        Args:
            node: Node id to register
        """
        self[node] = {
            "training": {},
            "testing_global_updates": {},  # Validation before training
            "testing_local_updates": {}  # Validation after training
        }

    def _register_metric(self, node: str, for_: str, metric_name: str):
        """Registers metric for the given node. It creates stating point for the metric from round 0.

        Args:
            node: The node id that metric value received from
            for_: One of (training, testing_global_updates, testing_local_updates). To indicate metric value belongs
                to which phase
            metric_name: Name of the metric to use as a key in MetricStore dict
        """

        # Round should start from 1 to match experiment's starting round
        self[node][for_].update({metric_name: {1: {'iterations': [], 'values': []}}})

    @staticmethod
    def _cumulative_iteration(rounds: dict) -> int:
        """Calculates cumulative iteration for the received metric value.

        Cumulative iteration should be calculated for each metric value received during training/validation to
        add it as next `step` in the tensorboard SummaryWriter. Please see Monitor._summary_writer.
        There are two assumptions in this implementation:

            - iterations need to be successive
            - last iteration index for each epoch should be same except last epoch

        Args:
            rounds: The dictionary that includes all the rounds for a metric, node and the phase

        Returns:
            int: cumulative iteration for the metric/validation result
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
    """ Monitors nodes scalar feed-backs during training"""

    def __init__(self, results_dir: str):
        """Constructor of the class.

        Args:
            results_dir: Directory for storing monitoring information.
        """

        self._log_dir = results_dir
        self._round = 1
        self._metric_store = MetricStore()
        self._event_writers = {}
        self._round_state = 0
        self._tensorboard = False

        if os.listdir(self._log_dir):
            logger.info('Removing tensorboard logs from previous experiment')
            # Clear logs' directory from the files from other experiments.
            self._remove_logs()

    def set_round(self, round_: int) -> int:
        """ Setts round number that metric results will be received for.

        By default, at the beginning round is equal to 1 which stands for the first round. T
        his method should be called by experiment `run_once` after each round completed, and round should be set
        to current round + 1. This will inform monitor about the current round where the metric values are getting
        received.

        Args:
            round_ : The round that metric value will be saved at they are received
        """
        self._round = round_

        return self._round

    def on_message_handler(self, msg: Dict[str, Any]):
        """ Handler for messages received through general/monitoring channel. This method is used as callback function
        in Requests class

        Args:
            msg: incoming message from Node. Must contain key named `command`, describing the nature
                of the command (currently the command is only add_scalar).
        """

        # For now monitor can only handle add_scalar messages

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
        """ Sets tensorboard flag, which is used to decide the behavior of the writing scalar values into
         tensorboard log files.

        Args:
            tensorboard: if True, data contained in AddScalarReply message will be passed to tensorboard
                         if False, fata will only be logged on the console
        """
        if isinstance(tensorboard, bool):
            self._tensorboard = tensorboard
        else:
            logger.error("tensorboard should be a boolean")
            self._tensorboard = False

    def close_writer(self):
        """ Closes `SummaryWriter` for each node """
        # Close each open SummaryWriter
        for node in self._event_writers:
            self._event_writers[node].close()

    def _remove_logs(self):
        """ Private method for removing logs files from tensorboard logs dir. """

        for file in os.listdir(self._log_dir):
            if not file.startswith('.'):  # do not want to remove dotfiles
                rf = os.path.join(self._log_dir, file)
                if os.path.isdir(rf):
                    shutil.rmtree(rf)
                elif os.path.isfile(rf):
                    os.remove(rf)

    def _log_metric_result(self, message: Dict, cum_iter: int = 0):
        """ Logs metric/scalar result that comes from nodes, and store them into tensorboard (through summary writer)
        if Tensorboard has been activated

        Args:
            message: Scalar message that is received from each node
            cum_iter: Global step/iteration for writing scalar to tensorboard log files
        """

        if message['train'] is True:
            header = 'Training'
        else:
            header = 'Validation On Global Updates' if message['test_on_global_updates'] else 'Validation On Local ' \
                                                                                              'Updates'

        metric_dict = message['metric']
        metric_result = ''
        for key, val in metric_dict.items():
            metric_result += "\t\t\t\t\t {}: \033[1m{:.6f}\033[0m \n".format(key, val)
        _min_iteration = min( message['batch_samples'],
                             message['total_samples'])
        # Loging fancy feedback for training
        logger.info(
            "\033[1m{}\033[0m \n"
            "\t\t\t\t\t NODE_ID: {} \n"
            "\t\t\t\t\t Round {}{} Iteration: {}/{} ({:.0f}%) | Samples: {}/{}\n {}"
            "\t\t\t\t\t ---------".format(
                header.upper(),
                message['node_id'],
                self._round,
                ' |' if message['epoch'] is None else f" Epoch: {message['epoch']} |",
                message["iteration"],
                message["num_batches"],
                100 * message["iteration"] / message["num_batches"],
                message["num_samples_trained"] if message["num_samples_trained"] is not None else _min_iteration,
                message['total_samples'],
                metric_result)
        )

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
        """Writes scalar values using torch SummaryWriter. It creates new summary file for each node.

        Args:
            header: The header/title for the plot that is going to be displayed on the tensorboard
            node: Node id to categorize result on the tensorboard by each node
            metric: Metric values
            cum_iter: Iteration number for the metric that is going to be added as scalar
        """

        # Initialize event SummaryWriters
        if node not in self._event_writers:
            self._event_writers[node] = SummaryWriter(log_dir=os.path.join(self._log_dir, node))

        for metric, value in metric.items():
            self._event_writers[node].add_scalar('{}/{}'.format(header.upper(), metric),
                                                 value,
                                                 cum_iter)

import logging
import os
import json
import inspect
from typing import Callable, Union, Dict, Any, TypeVar, Type

from fedbiomed.common.logger import logger
from fedbiomed.researcher.environ import environ
from fedbiomed.common.fedbiosklearn import SGDSkLearnModel
from fedbiomed.common.torchnn import TorchTrainingPlan

from fedbiomed.researcher.filetools import create_exp_folder, choose_bkpt_file, \
    create_unique_link, create_unique_file_link, find_breakpoint_path
from fedbiomed.researcher.aggregators import fedavg, aggregator
from fedbiomed.researcher.strategies.strategy import Strategy
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.monitor import Monitor

_E = TypeVar("Experiment")  # only for typing


class Experiment:
    """
    This class represents the orchestrator managing the federated training
    """

    def __init__(self,
                 tags: tuple,
                 nodes: list = None,
                 model_class: Union[Type[Callable], Callable] = None,
                 model_path: str = None,
                 model_args: dict = {},
                 training_args: dict = None,
                 rounds: int = 1,
                 aggregator: Union[Type[aggregator.Aggregator], aggregator.Aggregator] = None,
                 node_selection_strategy: Union[Type[Strategy], Strategy] = None,
                 save_breakpoints: bool = False,
                 training_data: Union [dict, FederatedDataSet] = None,
                 tensorboard: bool = False,
                 experimentation_folder: str = None
                 ):

        """ Constructor of the class.


        Args:
            tags (tuple): tuple of string with data tags
            nodes (list, optional): list of node_ids to filter the nodes
                                    to be involved in the experiment.
                                    Defaults to None (no filtering).
            model_class (Union[Type[Callable], Callable], optional): name or
                                    instance (object) of the model class to use
                                    for training.
                                    Should be a str type when using jupyter notebook
                                    or a Callable when using a simple python
                                    script.
            model_path (string, optional) : path to file containing model code
            model_args (dict, optional): contains output and input feature
                                        dimension. Defaults to None.
            training_args (dict, optional): contains training parameters:
                                            lr, epochs, batch_size...
                                            Defaults to None.
            rounds (int, optional): the number of communication rounds
                                    (nodes <-> central server).
                                    Defaults to 1.
            aggregator (Union[Type[aggregator.Aggregator], aggregator.Aggregator], optional):
                                    class or object defining the method
                                    for aggregating local updates.
                                    Default to None (uses fedavg.FedAverage() for training)
            node_selection_strategy (Union[Type[Strategy], Strategy], optional):
                                    class or object defining how nodes are sampled at each round
                                    for training, and how non-responding nodes are managed.
                                    Defaults to None (uses DefaultStrategy for training)
            save_breakpoints (bool, optional): whether to save breakpoints or
                                                not. Breakpoints can be used
                                                for resuming a crashed
                                                experiment. Defaults to False.
            training_data (Union [dict, FederatedDataSet], optional):
                    FederatedDataSet object or
                    dict of the node_id of nodes providing datasets for the experiment,
                    datasets for a node_id are described as a list of dict, one dict per dataset.
                    Defaults to None, datasets are searched from `tags` and `nodes`.
            tensorboard (bool): Tensorboard flag for displaying scalar values
                                during training in every node. If it is true,
                                monitor will write scalar logs into
                                `./runs` directory.
            experimentation_folder (str, optional): choose a specific name for the
                    folder where experimentation result files and breakpoints are stored.
                    This should just contain the name for the folder not a path.
                    The name is used as a subdirectory of `environ[EXPERIMENTS_DIR])`.
                    - Caveat : if using a specific name this experimentation will not be
                    automatically detected as the last experimentation by `load_breakpoint`
                    - Caveat : do not use a `experimentation_folder` name finishing
                    with numbers ([0-9]+) as this would confuse the last experimentation
                    detection heuristic by `load_breakpoint`.
        """

        self._tags = tags
        self._nodes = nodes
        self._reqs = Requests()

        if training_data is None:
            # no data passed : search for nodes either having tags that matches the tags
            # the researcher is looking for (`self._tags`) or based on node id
            # (`self._nodes`)
            training_data = self._reqs.search(self._tags, self._nodes)
        if type(training_data).__name__ != 'FederatedDataSet':
            # convert data to a data object if needed
            self._fds = FederatedDataSet(training_data)
        else:
            self._fds = training_data

        self._round_init = 0  # start from round 0
        self._node_selection_strategy = node_selection_strategy
        self._aggregator = aggregator

        self._experimentation_folder = create_exp_folder(experimentation_folder)

        self._model_class = model_class
        self._model_path = model_path
        self._model_args = model_args
        self._training_args = training_args
        self._rounds = rounds
        self._job = Job(reqs=self._reqs,
                        model=self._model_class,
                        model_path=self._model_path,
                        model_args=self._model_args,
                        training_args=self._training_args,
                        data=self._fds,
                        keep_files_dir=self.experimentation_path)

        # structure (dict ?) for additional parameters to the strategy
        # currently unused, to be defined when needed
        #self._sampled = None

        self._aggregated_params = {}
        self._save_breakpoints = save_breakpoints

        #  Monitoring loss values with tensorboard
        if tensorboard:
            self._monitor = Monitor()
            self._reqs.add_monitor_callback(self._monitor.on_message_handler)
        else:
            self._monitor = None
            # Remove callback. Since reqeust class is singleton callback
            # function might be already added into request before.
            self._reqs.remove_monitor_callback()


    @property
    def training_replies(self):
        return self._job.training_replies

    @property
    def aggregated_params(self):
        return self._aggregated_params

    @property
    def model_instance(self):
        return self._job.model

    @property
    def experimentation_folder(self):
        return self._experimentation_folder

    @property
    def experimentation_path(self):
        return os.path.join(environ['EXPERIMENTS_DIR'], self._experimentation_folder)


    def run(self, sync=True):
        """Runs an experiment, ie trains a model on nodes for a
        given number of rounds.
        It involves the following steps:


        Args:
            sync (bool, optional): whether synchronous execution is required
            or not.
            Defaults to True.

        Raises:
            NotImplementedError: [description]
        """
        if self._aggregator is None:
            self._aggregator = fedavg.FedAverage()
        else:
            if inspect.isclass(self._aggregator):
                self._aggregator = self._aggregator()
        
        if self._node_selection_strategy is None:
            # Default sample_nodes: train all nodes
            # Default refine: Raise error with any failure and stop the
            # experiment
            self._node_selection_strategy = DefaultStrategy(self._fds)
        else:
            if inspect.isclass(self._node_selection_strategy):
                self._node_selection_strategy = self._node_selection_strategy(self._fds)

        if not sync:
            raise NotImplementedError("One day....")

        # Run experiment
        if self._round_init >= self._rounds:
            logger.info("Round limit reached. Nothing to do")
            return

        for round_i in range(self._round_init, self._rounds):
            # Sample nodes using strategy (if given)
            self._job.nodes = self._node_selection_strategy.sample_nodes(round_i)
            logger.info('Sampled nodes in round ' + str(round_i) + ' ' + str(self._job.nodes))
            # Trigger training round on sampled nodes
            self._job.start_nodes_training_round(round=round_i)

            # refining/normalizing model weigths received from nodes
            model_params, weights = self._node_selection_strategy.refine(self._job.training_replies[round_i], round_i)

            # aggregate model from nodes to a global model
            aggregated_params = self._aggregator.aggregate(model_params,
                                                           weights)
            # write results of the aggregated model in a temp file
            aggregated_params_path = self._job.update_parameters(aggregated_params)
            logger.info(f'Saved aggregated params for round {round_i} in {aggregated_params_path}')

            self._aggregated_params[round_i] = {'params': aggregated_params,
                                                'params_path': aggregated_params_path}
            if self._save_breakpoints:
                self._save_state(round_i)

        if self._monitor is not None:
            # Close SummaryWriters for tensorboard
            self._monitor.close_writer()


    def model_file(self, display: bool = True ):
        
        """ This method displays saved final model for the experiment 
            that will be send to the nodes for training. 
        """
        model_file = self._job.model_file

        # Display content so researcher can copy
        if display:
            with open(model_file) as file:
                content = file.read()
                file.close()
                print(content)

        return self._job.model_file

    def check_model_status(self):
        
        """ Method for checking model status whether it is approved or 
            not by the nodes
        """
        responses = self._job.check_model_is_approved_by_nodes()

        return responses


    def _save_state(self, round: int=0):
        """
        Saves a state of the training at a current round.
        The following Experiment attributes will be saved:
         - round_number
         - round_number_due
         - tags
         - experimentation_folder
         - aggregator
         - node_selection_strategy
         - training_data
         - training_args
         - model_args
         - model_path
         - model_class
         - aggregated_params
        Attributes returned by the Job will also be saved

        Args:
            - round (int, optional): number of rounds already executed.
              Starts from 0. Defaults to 0.
        """

        breakpoint_path, breakpoint_file_name = \
            choose_bkpt_file(self._experimentation_folder, round)

        job_state = self._job.save_state(breakpoint_path, round)
        state = {
            # these are both Experiment and Job attributes : should be set also
            # in Experiment to better split breakpoint between the two classes
            'training_data': self._fds.data(),
            'training_args': self._training_args,
            'model_args': self._model_args,
            'model_path': self._job.model_file, # may not exist in Experiment with current version
            'model_class': self._job.model_class, # not properly
                              # formatted in Experiment with current version
            #
            # these are pure Experiment attributes
            'round_number': round + 1,
            'round_number_due': self._rounds,
            'experimentation_folder': self._experimentation_folder,
            'aggregator': self._aggregator.save_state(),
            'node_selection_strategy': self._node_selection_strategy.save_state(),
            'tags': self._tags,
            'aggregated_params': self._save_aggregated_params(
                                        self._aggregated_params, breakpoint_path)
        }
        state.update(job_state)

        # rewrite paths in breakpoint : use the links in breakpoint directory
        state['model_path'] = create_unique_link(
            breakpoint_path,
            # - Need a file with a restricted characters set in name to be able to import as module
            'model_' + str(round), '.py',
            # - Prefer relative path, eg for using experiment result after
            # experiment in a different tree
            os.path.join('..', os.path.basename(state["model_path"]))
            )

        # save state into a json file.
        breakpoint_file_path = os.path.join(breakpoint_path, breakpoint_file_name)
        with open(breakpoint_file_path, 'w') as bkpt:
            json.dump(state, bkpt)
        logger.info(f"breakpoint for round {round} saved at \
            {os.path.dirname(breakpoint_file_path)}")


    @classmethod
    def load_breakpoint(cls: Type[_E],
                        breakpoint_folder_path: str = None ) -> _E:
        """
        Loads breakpoint (provided a breakpoint has been saved)
        so experience can be resumed. Useful if training has crashed
        researcher side or if user wants to resume experiment.

        Args:
            - cls (Type[_E]): Experiment class
            - breakpoint_folder_path (str, optional): path of the breakpoint folder.
              Path can be absolute or relative eg: "var/experiments/Experiment_xx/breakpoints_xx".
              If None, loads latest breakpoint of the latest experiment.
              Defaults to None.

        Returns:
            - _E: Reinitialized experiment. With given object,
              user can then use `.run()` method to pursue model training.
        """


        # get breakpoint folder path (if it is None) and
        # state file
        breakpoint_folder_path, state_file = find_breakpoint_path(breakpoint_folder_path)
        breakpoint_folder_path = os.path.abspath(breakpoint_folder_path)

        # TODO: check if all elements needed for breakpoint are present
        with open(os.path.join(breakpoint_folder_path, state_file), "r") as f:
            saved_state = json.load(f)
        #print(saved_state)


        # -----  retrieve breakpoint training data ---
        bkpt_fds = FederatedDataSet(saved_state.get('training_data'))
        
        # -----  retrieve breakpoint sampling strategy ----
        bkpt_sampling_strategy_args = saved_state.get(
            "node_selection_strategy"
        )
        import_str = cls._import_module(bkpt_sampling_strategy_args)
        exec(import_str)
        bkpt_sampling_strategy = eval(bkpt_sampling_strategy_args.get("class"))
        bkpt_sampling_strategy = bkpt_sampling_strategy(bkpt_fds)
        bkpt_sampling_strategy.load_state(bkpt_sampling_strategy_args)

        # ----- retrieve federator -----
        bkpt_aggregator_args = saved_state.get("aggregator")
        import_str = cls._import_module(bkpt_aggregator_args)
        exec(import_str)
        bkpt_aggregator = eval(bkpt_aggregator_args.get("class"))
        bkpt_aggregator = bkpt_aggregator()
        bkpt_aggregator.load_state(bkpt_aggregator_args)

        # ------ initializing experiment -------

        loaded_exp = cls(tags=saved_state.get('tags'),
                         nodes=None,   # list of previous nodes is contained in training_data
                         model_class=saved_state.get("model_class"),
                         model_path=saved_state.get("model_path"),
                         model_args=saved_state.get("model_args"),
                         training_args=saved_state.get("training_args"),
                         rounds=saved_state.get("round_number_due"),
                         aggregator=bkpt_aggregator,
                         node_selection_strategy=bkpt_sampling_strategy,
                         save_breakpoints=True,
                         training_data=bkpt_fds,
                         experimentation_folder=saved_state.get('experimentation_folder')
                         )

        # ------- changing `Experiment` attributes -------
        loaded_exp._round_init = saved_state.get('round_number')
        loaded_exp._aggregated_params = loaded_exp._load_aggregated_params(
                                            saved_state.get('aggregated_params'),
                                            loaded_exp.model_instance.load
                                            )

        # ------- changing `Job` attributes -------
        loaded_exp._job.load_state(saved_state)

        logging.info(f"experimentation reload from {breakpoint_folder_path} successful!")
        return loaded_exp


    @staticmethod
    def _save_aggregated_params(aggregated_params_init: dict, breakpoint_path: str) -> Dict[int, dict]:
        """Extracts and format fields from aggregated_params that need
        to be saved in breakpoint. Creates link to the params file from the `breakpoint_path`
        and use them to reference the params files.

        Args:
            - breakpoint_path (str): path to the directory where breakpoints files
                and links will be saved

        Returns:
            - Dict[int, dict] : extract from `aggregated_params`
        """
        aggregated_params = {}
        for key, value in aggregated_params_init.items():
            
            params_path = create_unique_file_link(breakpoint_path,
                                            value.get('params_path'))
            aggregated_params[key] = { 'params_path': params_path }

        return aggregated_params


    @staticmethod
    def _load_aggregated_params(aggregated_params: Dict[int, dict], func_load_params: Callable
                ) -> Dict[int, dict]:
        """Reconstruct experiment results aggregated params structure
        from a breakpoint so that it is identical to a classical `_aggregated_params`

        Args:
            - aggregated_params (Dict[int, dict]) : JSON formatted aggregated_params
              extract from a breakpoint
            - func_load_params (Callable) : function for loading parameters
              from file to aggregated params data structure

        Returns:
            - Dict[int, dict] : reconstructed aggregated params from breakpoint
        """
        # needed for iteration on dict for renaming keys
        keys = [ key for key in aggregated_params.keys() ]
        # JSON converted all keys from int to string, need to revert
        for key in keys:
            aggregated_params[int(key)] = aggregated_params.pop(key)

        for aggreg in aggregated_params.values():
            aggreg['params'] = func_load_params(aggreg['params_path'], to_params=True)

        return aggregated_params


    @staticmethod
    def _import_module(args: Dict[str, Any]):
        """Build string containing module import command to run before
        instantiating an object of the type described in args
        
        Args:
            - args (Dict[str, Any]) : breakpoint arguments containing info
              for module import

        Returns:
            -  str: import command to be run
        """

        module_class = args.get("class")
        module_path = args.get("module")
        # node is using a fedbiomed node sampling strategy
        import_str = 'from ' + module_path + ' import ' + module_class

        return import_str

    @staticmethod
    def _create_object(args: Dict[str, Any]) -> Callable:
        """
        ???
        """
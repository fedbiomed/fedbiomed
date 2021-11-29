import logging
import os
import json
import shutil
import re
import inspect

from fedbiomed.common.logger import logger
from typing import Callable, Union, Tuple, Dict, Any, List, TypeVar, Type
from fedbiomed.researcher.environ import environ
from fedbiomed.common.fedbiosklearn import SGDSkLearnModel
from fedbiomed.common.torchnn import TorchTrainingPlan

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
                 model_class: Union[str, Callable] = None,
                 model_path: str = None,
                 model_args: dict = {},
                 training_args: dict = None,
                 rounds: int = 1,
                 aggregator: aggregator.Aggregator = fedavg.FedAverage(),
                 node_selection_strategy: Strategy = None,
                 save_breakpoints: bool = False,
                 training_data: dict = None,
                 tensorboard: bool = False,
                 experimentation_folder: str = None
                 ):

        """ Constructor of the class.


        Args:
            tags (tuple): tuple of string with data tags
            nodes (list, optional): list of node_ids to filter the nodes
                                    to be involved in the experiment.
                                    Defaults to None (no filtering).
            model_class (Union[str, Callable], optional): name of the
                                    model class to use for training. Should
                                    be a str type when using jupyter notebook
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
            aggregator (aggregator.Aggregator): class defining the method
                                                for aggragating local updates.
                                Default to fedavg.FedAvg().
            node_selection_strategy (Strategy): class defining how nodes
                                                  are sampled at each round
                                                  for training, and how
                                                  non-responding nodes
                                                  are managed. Defaults to
                                                  None (ie DefaultStrategy)
            save_breakpoints (bool, optional): whether to save breakpoints or
                                                not. Breakpoints can be used
                                                for resuming a crashed
                                                experiment. Defaults to False.
            training_data (dict, optional): dict of the node_id of nodes providing
                    datasets for the experiment. Datasets for a node_id are
                    described as a list of dict, one dict per dataset.
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
        # (below) search for nodes either having tags that matches the tags
        # the researcher is looking for (`self._tags`) or based on node id
        # (`self._nodes`)
        if training_data is None:
            # normal case
            self._training_data = self._reqs.search(self._tags,
                                                    self._nodes)
        else:
            # case where loaded from saved breakpoint
            self._training_data = training_data

        self._round_init = 0  # start from round 0
        self._fds = FederatedDataSet(self._training_data)
        self._node_selection_strategy = node_selection_strategy
        self._aggregator = aggregator

        self._experimentation_folder = \
            self._create_experimentation_folder(experimentation_folder)

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
                        data=self._fds)

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
            logger.info('Saved aggregated params for round {round_i} in {aggregated_params_path}')

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


    def _create_experimentation_folder(self, experimentation_folder=None):
        """Creates a folder for the current experiment (ie the current run of the model).
        Experiment files to keep are stored here: model file, all versions of node parameters,
        all versions of aggregated parameters, breakpoints.

        The created folder is a subdirectory of environ[EXPERIMENTS_DIR]

        Args:
            experimentation_folder (str, optional): optionaly provide an experimentation
                folder name. This should just contain the name of the folder not a path.
                Default: if no folder name is given, generate a `Experiment_x` name where `x-1`
                is the number of experiments already run (`x`=0 for the first experiment)

        Raises:
            PermissionError: cannot create experimentation folder
            OSError: cannot create experimentation folder
            ValueError: bad `experimentation_folder` argument

        Returns:
            str: experimentation folder
        """

        if not os.path.isdir(environ['EXPERIMENTS_DIR']):
            try:
                os.makedirs(environ['EXPERIMENTS_DIR'], exist_ok=True)
            except (PermissionError, OSError) as err:
                logger.error(f"Can not save experiment files because\
                    {environ['EXPERIMENTS_DIR']} folder could not be created\
                        due to {err}")
                raise

        # if no name is given for the experiment folder we choose one
        if not experimentation_folder:
             # FIXME: improve method robustness (here nb of exp equals nb of files
            # in directory)
            all_files = os.listdir(environ['EXPERIMENTS_DIR'])
            experimentation_folder = "Experiment_" + str(len(all_files))
        else:
            if os.path.basename(experimentation_folder) != experimentation_folder:
                # experimentation folder cannot be a path
                raise ValueError("Bad experimentation folder {experimentation_folder} - \
                    it cannot be a path")

        try:
            os.makedirs(os.path.join(environ['EXPERIMENTS_DIR'], experimentation_folder),
                        exist_ok=True)
        except (PermissionError, OSError) as err:
            logger.error(f"Can not save experiment files because\
                    {environ['EXPERIMENTS_DIR']}/{experimentation_folder} \
                    folder could not be created due to {err}")
            raise
        
        return experimentation_folder


    def _create_breakpoint_file_and_folder(self,
                                           round: int = 0) -> Tuple[str, str]:
        """It creates a breakpoint file for each round.

        Args:
            round (int, optional): the current number of rounds minus one.
            Starts from 0. Defaults to 0.

        Returns:
            breakpoint_folder_path (str): name of the created folder that
            will contain all data for the current round
            breakpoint_file (str): name of the file that will
            contain the state of an experiment.
        """
        breakpoint_folder = "breakpoint_" + str(round)
        breakpoint_folder_path = os.path.join(environ['EXPERIMENTS_DIR'],
                                              self._experimentation_folder,
                                              breakpoint_folder)
        try:
            os.makedirs(breakpoint_folder_path, exist_ok=True)
        except (PermissionError, OSError) as err:
            logger.error(f"Can not save breakpoint folder at\
                {breakpoint_folder_path} due to some error {err} ")

        breakpoint_file = breakpoint_folder + ".json"
        return breakpoint_folder_path, breakpoint_file

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
            round (int, optional): number of rounds already executed.
            Starts from 0. Defaults to 0.
        """
        job_state = self._job.save_state(round)
        state = {
            # these are both Experiment and Job attributes : should be set also
            # in Experiment to better split breakpoint between the two classes
            'training_data': self._training_data,
            'training_args': self._training_args,
            'model_args': self._model_args,
            'model_path': self._job._model_file, # may not exist in Experiment with current version
            'model_class': self._job._repository_args.get('model_class'), # not properly
                              # formatted in Experiment with current version
            #
            # these are pure Experiment attributes
            'round_number': round + 1,
            'round_number_due': self._rounds,
            'experimentation_folder': self._experimentation_folder,
            'aggregator': self._aggregator.save_state(),
            'node_selection_strategy': self._node_selection_strategy.save_state(),
            'tags': self._tags,
            'aggregated_params': self._save_aggregated_params()
        }

        state.update(job_state)
        breakpoint_path, breakpoint_file_name = self._create_breakpoint_file_and_folder(round)

        # TODO: optimize - one copy for the experiment is enough
        # Need a file with a restricted characters set in name to be able to import as module
        copied_model_file = "model_" + str(round) + ".py"
        copied_model_path = os.path.join(breakpoint_path,
                                         copied_model_file)
        shutil.copy2(state["model_path"], copied_model_path)
        state["model_path"] = copied_model_path

        # save state into a json file.
        breakpoint_path = os.path.join(breakpoint_path, breakpoint_file_name)
        with open(breakpoint_path, 'w') as bkpt:
            json.dump(state, bkpt)
        logger.info(f"breakpoint for round {round} saved at \
            {os.path.dirname(breakpoint_path)}")


    def _save_aggregated_params(self) -> Dict[int, dict]:
        """Extracts and format fields from aggregated_params that need
        to be saved in breakpoint

        Returns:
            Dict[int, dict] : extract from `self._aggregated_params`
        """
        aggregated_params = { key: { 'params_path': value.get('params_path') } 
            for key, value in self._aggregated_params.items() }

        return aggregated_params

    def _load_aggregated_params(self, aggregated_params: Dict[int, dict]) -> Dict[int, dict]:
        """Reconstruct experiment results aggregated params structure
        from a breakpoint so that it is identical to a classical `_aggregated_params`

        Args:
            aggregated_params (Dict[int, dict]): JSON formatted aggregated_params
              extract from a breakpoint

        Returns:
            Dict[int, dict] : reconstructed `aggregated_params` from breakpoint
        """
        # needed for iteration on dict for renaming keys
        keys = [ key for key in aggregated_params.keys() ]
        # JSON converted all keys from int to string, need to revert
        for key in keys:
            aggregated_params[int(key)] = aggregated_params.pop(key)

        for aggreg in aggregated_params.values():
            aggreg['params'] = self.model_instance.load(
                aggreg['params_path'], to_params=True)

        return aggregated_params

    @staticmethod
    def _get_latest_file(pathfile: str,
                         list_name_file: List[str],
                         only_folder: bool = False) -> str:
        """Gets latest file from folder specified in `list_name_file`
        from the following convention:
            the more recent folder is the file written as `myfile_xx`
            where `xx` is the higher integer amongst files in `list_name_file`

        Args:
            pathfile (str): path towards folder on system
            list_name_file (List[str]): a list containing files
            only_folder (bool, optional): whether to consider only
            folder names or to consider both  file and folder names.
            Defaults to False.

        Raises:
            FileNotFoundError: triggered if none of the names
            in folder doesnot match with naming convention.

        Returns:
            str: More recent file name given naming convention.
        """
        latest_nb = 0
        latest_folder = None
        for exp_folder in list_name_file:

            exp_match = re.search(r'[0-9]*$',
                                  exp_folder)

            if len(exp_folder) != exp_match.span()[0]:

                dir_path = os.path.join(pathfile, exp_folder)
                if not only_folder or os.path.isdir(dir_path):
                    f_idx = exp_match.span()[0]
                    order = int(exp_folder[f_idx:])

                    if order >= latest_nb:
                        latest_nb = order
                        latest_folder = exp_folder

        if latest_folder is None and len(list_name_file) != 0:

            raise FileNotFoundError("None of those are breakpoints{}".format(", ".join(list_name_file)))
        return latest_folder

    @staticmethod
    def _find_breakpoint_path(breakpoint_folder_path: str = None) -> Tuple[str, str]:
        """Finds breakpoint path, regarding if
        user specifies a specific breakpoint path or
        considers only the latest breakpoint.

        Args:
            breakpoint_folder_path (str, optional):path towards breakpoint. If None,
            (default), consider latest breakpoint saved on default path
            (provided at least one breakpoint exists). Defaults to None.

        Raises:
            FileNotFoundError: triggered either if breakpoint cannot be found,
            or cannot be parsed
            Exception: triggered if breakpoint folder is empty.
            FileNotFoundError: [description]

        Returns:
            str: folder location toward breakpoint (unchanged if
            specified by user)
            str: latest experiment and breakpoint folder.
        """
        # First, let's test if folder is a real folder path
        if breakpoint_folder_path is None:
            try:
                # retrieve latest experiment

                # content of breakpoint folder
                experiment_folders = os.listdir(environ['EXPERIMENTS_DIR'])

                latest_exp_folder = Experiment._get_latest_file(
                    environ['EXPERIMENTS_DIR'],
                    experiment_folders,
                    only_folder=True)

                latest_exp_folder = os.path.join(environ['EXPERIMENTS_DIR'],
                                                  latest_exp_folder)

                bkpt_folders = os.listdir(latest_exp_folder)

                breakpoint_folder_path = Experiment._get_latest_file(
                    latest_exp_folder,
                    bkpt_folders,
                    only_folder=True)

                breakpoint_folder_path = os.path.join(latest_exp_folder,
                                                 breakpoint_folder_path)
            except FileNotFoundError as err:
                logger.error("cannot find a breakpoint in:" + environ['EXPERIMENTS_DIR'] + " - " + err)
                raise FileNotFoundError("Cannot find latest breakpoint \
                    saved. Are you sure you have saved at least one breakpoint?")
            except TypeError as err:
                # case where `Experiment._get_latest_file`
                # Fails (ie return `None`)
                logger.error(err)

                #### REVIEW: latest_exp_folder may be undefined here (it try block breaks before its definition)
                raise FileNotFoundError(f"found an empty breakpoint folder\
                    at {latest_exp_folder}")
        else:
            if not os.path.isdir(breakpoint_folder_path):
                raise FileNotFoundError(f"{breakpoint_folder_path} is not a folder")

            # check if folder is a valid breakpoint

            # get breakpoint material
            # regex : breakpoint_\d\.json

        #
        # verify the validity of the breakpoint content
        # TODO: be more robust
        all_breakpoint_materials = os.listdir(breakpoint_folder_path)
        if len(all_breakpoint_materials) == 0:
            raise Exception("breakpoint folder is empty !")

        state_file = None
        for breakpoint_material in all_breakpoint_materials:
            # look for the json file containing experiment state
            # (it should be named `brekpoint_xx.json`)
            json_match = re.fullmatch(r'breakpoint_\d*\.json',
                                      breakpoint_material)

            if json_match is not None:
                logging.debug(f"found json file containing states at\
                    {breakpoint_material}")
                state_file = breakpoint_material

        if state_file is None:
            logging.error(f"Cannot find JSON file containing\
                model state at {breakpoint_folder_path}. Aborting")
            raise FileNotFoundError(f"Cannot find JSON file containing\
                model state at {breakpoint_folder_path}. Aborting")
            #sys.exit(-1)
        return breakpoint_folder_path, state_file

    @classmethod
    def load_breakpoint(cls: Type[_E],
                        breakpoint_folder_path: str = None ) -> _E:
        """
        Loads breakpoint (provided a breakpoint has been saved)
        so experience can be resumed. Useful if training has crashed
        researcher side or if user wants to resume experiment.

        Args:
            cls (Type[_E]): Experiment class
            breakpoint_folder_path (str, optional): path of the breakpoint folder.
            Path can be absolute or relative eg: "var/experiments/Experiment_xx/breakpoints_xx".
            If None, loads latest breakpoint of the latest experiment.
            Defaults to None.

        Returns:
            _E: Reinitialized experiment. With given object,
            user can then use `.run()` method to pursue model training.
        """


        # get breakpoint folder path (if it is None) and
        # state file
        breakpoint_folder_path, state_file = cls._find_breakpoint_path(breakpoint_folder_path)
        breakpoint_folder_path = os.path.abspath(breakpoint_folder_path)

        # TODO: check if all elements needed for breakpoint are present
        with open(os.path.join(breakpoint_folder_path, state_file), "r") as f:
            saved_state = json.load(f)
        #print(saved_state)


        # TODO: for both node sampling strategy & aggregator
        # deal with saved parameters

        # -----  retrieve breakpoint sampling strategy ----
        bkpt_sampling_strategy_args = saved_state.get(
            "node_selection_strategy"
        )
        import_str = cls._import_module(bkpt_sampling_strategy_args)
        exec(import_str)
        bkpt_sampling_strategy = eval(bkpt_sampling_strategy_args.get("class"))

        # ----- retrieve federator -----
        bkpt_aggregator_args = saved_state.get("aggregator")
        import_str = cls._import_module(bkpt_aggregator_args)
        exec(import_str)
        bkpt_aggregator = eval(bkpt_aggregator_args.get("class"))

        # ------ initializing experiment -------

        loaded_exp = cls(tags=saved_state.get('tags'),
                         nodes=None,   # list of previous nodes is contained in training_data
                         model_class=saved_state.get("model_class"),
                         model_path=saved_state.get("model_path"),
                         model_args=saved_state.get("model_args"),
                         training_args=saved_state.get("training_args"),
                         rounds=saved_state.get("round_number_due"),
                         aggregator=bkpt_aggregator(),
                         node_selection_strategy=bkpt_sampling_strategy,
                         save_breakpoints=True,
                         training_data=saved_state.get('training_data'),
                         experimentation_folder=saved_state.get('experimentation_folder')
                         )

        # ------- changing `Experiment` attributes -------
        loaded_exp._round_init = saved_state.get('round_number')
        loaded_exp._aggregated_params = \
            loaded_exp._load_aggregated_params(saved_state.get('aggregated_params'))

        # ------- changing `Job` attributes -------
        loaded_exp._job.load_state(saved_state)

        logging.debug(f"reloading from {breakpoint_folder_path} successful!")
        return loaded_exp

    @staticmethod
    def _import_module(args: Dict[str, Any]):
        """Build string containing module import command to run before
        instantiating an object of the type described in args
        
        Return:
          str: import command to be run
        """

        module_class = args.get("class")
        module_path = args.get("module", "custom")
        if module_path == "custom":
            # case where user is loading its own custom
            # node sampling strategy
            import_str = 'import ' + module_class
        else:
            # node is using a fedbiomed node sampling strategy
            import_str = 'from ' + module_path + ' import ' + module_class
        logging.debug(f"{module_class} loaded !")

        return import_str


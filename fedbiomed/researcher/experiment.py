import logging
import os
import json
import shutil
import re

from fedbiomed.common.logger import logger
from typing import Callable, Union, Tuple, Dict, Any, List, TypeVar, Type
from fedbiomed.researcher.environ import BREAKPOINTS_DIR 
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
    _training_data = None  # should contain client and dataset values
    
    def __init__(self,
                 tags: tuple,
                 clients: list = None,
                 model_class: Union[str, Callable] = None,
                 model_path: str = None,
                 model_args: dict = {},
                 training_args: dict = None,
                 rounds: int = 1,
                 aggregator: aggregator.Aggregator = fedavg.FedAverage(),
                 client_selection_strategy: Strategy = None,
                 save_breakpoints: bool = False,
                 tensorboard: bool = False
                 ):

        """ Constructor of the class.


        Args:
            tags (tuple): tuple of string with data tags
            clients (list, optional): list of client_ids to filter the nodes
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
                                    (clients <-> central server).
                                    Defaults to 1.
            aggregator (aggregator.Aggregator): class defining the method
                                                for aggragating local updates.
                                Default to fedavg.FedAvg().
            client_selection_strategy (Strategy): class defining how clients
                                                  are sampled at each round
                                                  for training, and how
                                                  non-responding clients
                                                  are managed. Defaults to
                                                  None (ie DefaultStartegy)
            save_breakpoints (bool, optional): whether to save breakpoints or
                                                not. Breakpoints can be used
                                                for resuming a crashed
                                                experiment. Defaults to False.

            tensorboard (bool): Tensorboard flag for displaying scalar values 
                                during tarning in every node. If it is true, 
                                monitor will write scalar logs in the
                                var/tensorboard directory
        """ 
        
        self._tags = tags
        self._clients = clients
        self._reqs = Requests()
        # (below) search for nodes either having tags that matches the tags
        # the researcher is looking for (`self._tags`) or based on client id
        # (`self._clients`)
        if self._training_data is None:
            # normal case
            training_data = self._reqs.search(self._tags,
                                              self._clients)
        else:
            # case where loaded from saved breakpoint
            training_data = self._training_data
        self._round_init = 0  # start from round 0
        self._fds = FederatedDataSet(training_data)
        self._client_selection_strategy = client_selection_strategy
        self._aggregator = aggregator

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
        self._sampled = None

        self._aggregated_params = {}
        self._save_breakpoints = save_breakpoints
        #  folder will be created
        self._monitor = Monitor(tensorboard=tensorboard)
        self._training_data = None  # reset variable to `None`

    @property
    def training_replies(self):
        return self._job.training_replies

    @property
    def aggregated_params(self):
        return self._aggregated_params

    @property
    def model_instance(self):
        return self._job.model

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
        if self._client_selection_strategy is None or self._sampled is None:
            # Default sample_clients: train all clients
            # Default refine: Raise error with any failure and stop the
            # experiment
            self._client_selection_strategy = DefaultStrategy(self._fds)
        else:
            self._client_selection_strategy = self._client_selection_strategy(self._fds, self._sampled)

        if self._save_breakpoints:
            self._create_breakpoints_folder()
            self._create_breakpoint_exp_folder()
        if not sync:
            raise NotImplementedError("One day....")

        # Run experiment
        for round_i in range(self._round_init, self._rounds):
            # Sample clients using strategy (if given)
            self._job.clients = self._client_selection_strategy.sample_clients(round_i)
            logger.info('Sampled clients in round ' + str(round_i) + ' ' + str(self._job.clients))
            # Trigger training round on sampled clients
            self._job.start_clients_training_round(round=round_i)

            # refining/normalizing model weigths received from nodes
            model_params, weights = self._client_selection_strategy.refine(self._job.training_replies[round_i], round_i)

            # aggregate model from nodes to a global model
            aggregated_params = self._aggregator.aggregate(model_params,
                                                           weights)
            # write results of the aggregated model in a temp file
            aggregated_params_path = self._job.update_parameters(aggregated_params)

            self._aggregated_params[round_i] = {'params': aggregated_params,
                                                'params_path': aggregated_params_path}
            if self._save_breakpoints:
                self._save_state(round_i)
                
            # Increase round state in the monitor
            self._monitor.increase_round()
        
        # Close SummaryWriters for tensorboard
        self._monitor.close_writer()

    def _create_breakpoints_folder(self):
        """Creates a general folder for storing breakpoints (if non existant)
        into the `BREAKPOINTS_DIR` folder.
        """
        self._breakpoint_path_file = BREAKPOINTS_DIR
        if not os.path.isdir(self._breakpoint_path_file):
            try:
                os.makedirs(self._breakpoint_path_file, exist_ok=True)
            except (PermissionError, OSError) as err:
                logger.error(f"Can not save breakpoints files because\
                    {self._breakpoint_path_file} folder could not be created\
                        due to {err}")
        
    def _create_breakpoint_exp_folder(self):
        """Creates a breakpoint folder for the current experiment (ie the
        current run of the model). This folder is located at
        `BREAKPOINTS_DIR_DIR/Experiment_x` where `x-1` is the number of experiments
        already run (`x`=0 for the first experiment)
        """
        # FIXME: improve method robustness (here nb of exp equals nb of files
        # in directory)
        all_files = os.listdir(self._breakpoint_path_file)
        if not hasattr(self, "_exp_breakpoint_folder"):
            self._exp_breakpoint_folder = "Experiment_" + str(len(all_files))
        try:
            os.makedirs(os.path.join(self._breakpoint_path_file,
                                     self._exp_breakpoint_folder),
                        exist_ok=True)
        except (PermissionError, OSError) as err: 
            logger.error(f"Can not save breakpoints files because\
                    {self._breakpoint_path_file} folder could not be created\
                        due to {err}")
            
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
        breakpoint_folder_path = os.path.join(self._breakpoint_path_file,
                                              self._exp_breakpoint_folder,
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
        The following attributes will be saved:
         - 'round_number'
         - round_number_due
         - tags
         - 'aggregator'
         - 'client_selection_strategy'
         - 'round_success'
         - researcher_id
         - job_id
         - training_data
         - training_args
         - model_args
         - command
         - model_path
         - params_path
         - model_class
         - training_replies
        
        Args:
            round (int, optional): number of rounds already executed.
            Starts from 0. Defaults to 0.
        """
        self._job.save_state(round)  # create attribute `_job.state`
        job_state = self._job.state
        state = {
            'round_number': round + 1,
            'round_number_due': self._rounds,
            'aggregator': self._aggregator.save_state(),
            'client_selection_strategy': self._client_selection_strategy.save_state(),
            'round_success': True,
            'tags': self._tags
        }
        
        state.update(job_state)
        breakpoint_path, breakpoint_file_name = self._create_breakpoint_file_and_folder(round)
        
        
        # copy model parameters and model to breakpoint folder
        for client_id, param_path in state['params_path'].items():
            copied_param_file = "params_" + client_id + ".pt"
            copied_param_path = os.path.join(breakpoint_path,
                                             copied_param_file)
            shutil.copy2(param_path, copied_param_path)
            state['params_path'][client_id] = copied_param_path
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
    def _find_breakpoint_path(breakpoint_folder: str = None) -> Tuple[str, str]:
        """Finds breakpoint path, regarding if
        user specifies a specific breakpoint path or
        considers only the latest breakpoint.

        Args:
            breakpoint_folder (str, optional):path towards breakpoint. If None, 
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
        if breakpoint_folder is None:
            try:
                # retrieve latest experiment
                default_breakpoints_folder = BREAKPOINTS_DIR
                experiment_folders = os.listdir(default_breakpoints_folder)
                
                latest_exp_folder = Experiment._get_latest_file(
                                                    default_breakpoints_folder,
                                                    experiment_folders,
                                                    only_folder=True)
                latest_exp_folder = os.path.join(default_breakpoints_folder,
                                                 latest_exp_folder)
                bkpt_folders = os.listdir(latest_exp_folder)
                breakpoint_folder = Experiment._get_latest_file(
                                                latest_exp_folder,
                                                bkpt_folders,
                                                only_folder=True)
                
                breakpoint_folder = os.path.join(latest_exp_folder,
                                                 breakpoint_folder)
            except FileNotFoundError as err:
                logger.error(err)
                raise FileNotFoundError("Cannot find latest breakpoint \
                    saved. Are you sure you have saved at least one breakpoint?")
            except TypeError as err:
                # case where `Experiment._get_latest_file`
                # Fails (ie return `None`)
                logger.error(err)
                raise FileNotFoundError(f"found an empty breakpoint folder\
                    at {latest_exp_folder}")
        else:
            if not os.path.isdir(breakpoint_folder):
                if os.path.isfile(breakpoint_folder):
                    raise FileNotFoundError(f"{breakpoint_folder} \
                        is not a folder but a file")
                else:
                    
                    # trigger an exception
                    raise FileNotFoundError(f"Cannot find {breakpoint_folder}!")
            # check if folder is a valid breakpoint
            
            # get breakpoint material
            # regex : breakpoint_\d\.json

        all_breakpoint_materials = os.listdir(breakpoint_folder)
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

            else:
                continue
        if state_file is None:
            logging.error(f"Cannot find JSON file containing\
                model state at {breakpoint_folder}. Aborting")
            raise FileNotFoundError(f"Cannot find JSON file containing\
                model state at {breakpoint_folder}. Aborting")
            #sys.exit(-1)
        return breakpoint_folder, state_file

    @classmethod
    def load_breakpoint(cls: Type[_E],
                        breakpoint_folder: str = None,
                        extra_rounds: int = 0) -> _E:
        """
        Loads breakpoint (provided a breakpoint has been saved)
        so experience can be resumed. Useful if training has crashed
        researcher side or if user wants to resume experiment.

        Args:
            cls (Type[_E]): Experiment class
            breakpoint_folder (str, optional): path of the breakpoint folder.
            Path should be: "breakpoints/Experiment_xx/breakpoints_xx".
            If None, loads latest breakpoint of the latest experiment.
            Defaults to None.
            extra_rounds (int, optional): executes extra training rounds.
            Defaults to 1.

        Returns:
            _E: Reinitialized experiment. With given object,
            user can then use `.run()` method to pursue model training.
        """
        # get breakpoint folder path (if it is None) and 
        # state file
        breakpoint_folder, state_file = Experiment._find_breakpoint_path(breakpoint_folder)

        # TODO: check if all elements needed for breakpoint are present
        with open(os.path.join(breakpoint_folder, state_file), "r") as f:
            saved_state = json.load(f)
        

        # TODO: for both client sampling strategy & aggregator
        # deal with saved parameters

        # -----  retrieve breakpoint sampling starategy ----
        bkpt_sampling_startegy_args = saved_state.get(
                                                "client_selection_strategy"
                                                     )
        import_str = cls._instancialize_module(bkpt_sampling_startegy_args)
        # (above) importing client strategy
        
        exec(import_str)
        bkpt_sampling_startegy = eval(bkpt_sampling_startegy_args.get("class"))
        
        # ----- retrieve federator -----
        bkpt_aggregator_args = saved_state.get("aggregator")
        import_str = cls._instancialize_module(bkpt_aggregator_args)

        exec(import_str)
        bkpt_aggregator = eval(bkpt_aggregator_args.get("class"))
        
        # remaining round to resume before end of experiment
        remaining_round = max(extra_rounds + \
            saved_state.get('round_number_due', 1), 1) 
        # ------ initializing experiment -------
        cls._training_data = saved_state.get('training_data')
        
        loaded_exp = cls(tags=saved_state.get('tags'),
                         clients=saved_state.get('client_id'),
                         model_class=saved_state.get("model_class"),
                         model_path=saved_state.get("model_path"),
                         model_args=saved_state.get("model_args"),
                         training_args=saved_state.get("training_args"),
                         rounds=remaining_round,
                         aggregator=bkpt_aggregator(),
                         client_selection_strategy=bkpt_sampling_startegy,
                         save_breakpoints=True,
                         )

        # get experiment folder for breakpoint
        loaded_exp._exp_breakpoint_folder = os.path.dirname(breakpoint_folder)
        loaded_exp._round_init = saved_state.get('round_number', 0)
        loaded_exp._rounds = extra_rounds + saved_state.get('round_number_due', 1)
        # ------- changing `Job` attributes -------
        loaded_exp._job._id = saved_state.get('job_id')
        loaded_exp._job._data = FederatedDataSet(
                                        saved_state.get('training_data')
                                                )
        loaded_exp._load_training_replies(saved_state.get('training_replies'),
                                          saved_state.get("params_path")
                                          )
        loaded_exp._job._researcher_id = saved_state.get('researcher_id')
        logging.debug(f"reloading from {breakpoint_folder} successful!")
        return loaded_exp

    @staticmethod
    def _instancialize_module(args: Dict[str, Any],
                              class_key: str = "class",
                              module_key: str = "module"):
        
        module_class = args.get(class_key)
        module_path = args.get(module_key, "custom")
        if module_path == "custom":
            # case where user is loading its own custom
            # client sampling strategy
            import_str = 'import ' + module_class
        else:
            # client is using a fedbiomed client sampling strategy
            import_str = 'from ' + module_path + ' import ' + module_class
        logging.debug(f"{module_class} loaded !")

        return import_str

    def _load_training_replies(self,
                               training_replies: Dict[int, List[dict]],
                               params_path: Dict[str, str]):
        self._job._load_training_replies(training_replies,
                                         params_path)

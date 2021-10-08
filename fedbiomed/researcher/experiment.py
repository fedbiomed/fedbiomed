import os
import json
import shutil

from fedbiomed.common.logger import logger
from typing import ByteString, Callable, Union
from fedbiomed.researcher.environ import VAR_DIR

from fedbiomed.researcher.aggregators import fedavg, aggregator
from fedbiomed.researcher.strategies.strategy import Strategy
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.datasets import FederatedDataSet


class Experiment:
    """
    This class represents the orchestrator managing the federated training
    """

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
                 save_breakpoints: bool = True
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
        """
        self._tags = tags
        self._clients = clients
        self._reqs = Requests()
        # (below) search for nodes either having tags that matches the tags
        # the researcher is looking for (`self._tags`) or based on client id
        # (`self._clients`)
        self._fds = FederatedDataSet(self._reqs.search(self._tags,
                                                       self._clients))
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
        for round_i in range(self._rounds):
            # Sample clients using strategy (if given)
            self._job.clients = self._client_selection_strategy.sample_clients(round_i)
            logger.info('Sampled clients in round ' + str(round_i) + ' ' + str(self._job.clients))
            # Trigger training round on sampled clients
            self._job.start_clients_training_round(round=round_i)

            # refining/normalizing model weigths received from nodes
            model_params, weights = self._client_selection_strategy.refine( self._job.training_replies[round_i], round_i)

            # aggregate model from nodes to a global model
            aggregated_params = self._aggregator.aggregate(model_params,
                                                           weights)
            # write results of the aggregated model in a temp file
            aggregated_params_path = self._job.update_parameters(aggregated_params)

            self._aggregated_params[round_i] = {'params': aggregated_params,
                                                'params_path': aggregated_params_path}
            if self._save_breakpoints:
                self._save_state(round_i)

    def _create_breakpoints_folder(self):
        self._breakpoint_path_file = os.path.join(VAR_DIR, "breakpoints")
        if not os.path.isdir(self._breakpoint_path_file):
            try:
                os.makedirs(self._breakpoint_path_file)
            except PermissionError:
                logger.error(f"Can not save breakpoints files because\
                    {self._breakpoint_path_file} folder could not be created\
                        due to PermissionError")
        
    def _create_breakpoint_exp_folder(self):
        all_file = os.listdir(self._breakpoint_path_file)
        self._exp_breakpoint_folder = "Experiment_" + str(len(all_file))
        os.makedirs(os.path.join(self._breakpoint_path_file,
                                 self._exp_breakpoint_folder))
        
    def _create_breakpoint_file_and_folder(self, round:int=0) -> str:
        breakpoint_folder = "breakpoint_" + str(round)
        breakpoint_folder_path = os.path.join(self._breakpoint_path_file,
                                              self._exp_breakpoint_folder,
                                              breakpoint_folder)
        os.makedirs(breakpoint_folder_path)
        breakpoint_file = breakpoint_folder + ".json"
        return breakpoint_folder_path, breakpoint_file

    @staticmethod
    def _get_class_name(obj: object) -> str:
        if hasattr(obj, "__name__"):
            return obj.__name__
        elif hasattr(obj, "__str__"):
            return str(obj)
        elif hasattr(obj, "__repr__"):
            return repr(obj)
        else:
            return "none"
        
    def _save_state(self, round: int=0):
        """
        """
        self._job.save_state(round)  # create attribute `_job.state`
        job_state = self._job.state
        state = {
            #"params_path_file":
            #"aggregated_params":self._job.model_instance.save(self._aggregated_params[round],
            'round_number': round,
            'aggregator': self._aggregator.save_state(),
            'client_selection_strategy': self._client_selection_strategy.save_state(),
            'round_success': True
        }
        state.update(job_state)
        breakpoint_path, breakpoint_file_name = self._create_breakpoint_file_and_folder(round)
        with open(os.path.join(breakpoint_path, breakpoint_file_name), 'w') as bkpt:
            json.dump(state, bkpt)
        print(f"breakpoint saved at {breakpoint_file_name}")
        
        #copy model parameters and model to breakpoint folder
        for client_id, param_path in state['params_path'].items():
            shutil.copy2(param_path, os.path.join(breakpoint_path,
                                                  "client_" + client_id + ".pt"))
        shutil.copy2(state["model_path"], os.path.join(breakpoint_path,
                                                  "model_" + str(round) + ".py"))
from fedbiomed.common.logger import logger
from typing import Callable, Union

from fedbiomed.researcher.aggregators import fedavg, aggregator
from fedbiomed.researcher.strategies.strategy import Strategy
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.monitoring import Monitoring

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

        self._monitor = Monitoring(tensorboard=tensorboard)

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

            # Notify feedback class
            self._monitor.increase_round()

        # Close summary writer
        self._monitor.close_writer()
        
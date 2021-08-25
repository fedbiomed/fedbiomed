from fedbiomed.researcher.aggregators import fedavg
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
                 model_class: str = None,
                 model_path: str = None,
                 model_args: dict = {},
                 training_args: dict = None,
                 rounds=int,
                 aggregator=fedavg,
                 client_selection_strategy: Strategy = None,
                 ):

        """ Constructor of the class

        Args:
            tags (tuple): tuple of string with data tags
            clients (list, optional): list of client_ids to filter the nodes to be involved in 
                                      the experiment. Defaults to None (no filtering).
            model_class (string, optional): name of the model class to use for training
            model_path (string, optional) : path to file containing model code
            model_args (dict, optional): contains output and input feature dimension. 
                                            Defaults to None.
            training_args (dict, optional): contains training parameters: lr, epochs, batch_size...
                                            Defaults to None.
            rounds (int): the number of communication rounds (clients <-> central server)
            aggregator (class): class defining the method for aggragating local updates.
                                Default to fedavg
            client_selection_strategy (class): class defining how clients are sampled at each round for training,
                                                and how non-responding clients are managed.
                                                Default to None
        """
        self._tags = tags
        self._clients = clients
        self._reqs = Requests()
        self._fds = FederatedDataSet(self._reqs.search(self._tags, self._clients))
        self._client_selection_strategy = client_selection_strategy
        self._aggregator = aggregator

        self._model_class = model_class
        self._model_path = model_path
        self._model_args = model_args
        self._training_args = training_args
        self._rounds = rounds
        self._job = Job(reqs=self._reqs,
                model=self._model_class, model_path=self._model_path, model_args=self._model_args,
                training_args=self._training_args, data=self._fds)

        # structure (dict ?) for additional parameters to the strategy
        # currently unused, to be defined when needed
        self._sampled = None

        self._aggregated_params = {}

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
            sync (bool, optional): whether synchronous execution is required or not.
            Defaults to True.

        Raises:
            NotImplementedError: [description]
        """
        if self._client_selection_strategy is None or self._sampled is None:
            # Default sample_clients: train all clients
            # Default refine: Raise error with any failure and stop the experiment
            self._client_selection_strategy = DefaultStrategy(self._fds)
        else:
            self._client_selection_strategy = self._client_selection_strategy(self._fds, self._sampled)

        if not sync:
            raise NotImplementedError("One day....")

        # Run experiment
        for round_i in range(self._rounds):
            # Sample clients using strategy (if given)
            self._job.clients = self._client_selection_strategy.sample_clients(round_i)
            print('Sampled clients in round ', round_i, ' ', self._job.clients)
            # Trigger training round on sampled clients
            self._job.start_clients_training_round(round=round_i)

            # refining/normalizing model weigths recieved from nodes
            model_params, weights = self._client_selection_strategy.refine( self._job.training_replies[round_i], round_i)
            
            # aggregate model from nodes to a global model
            aggregated_params = self._aggregator.aggregate(model_params, weights)
            # write results of the aggregated model in a temp file
            aggregated_params_path = self._job.update_parameters(aggregated_params)

            self._aggregated_params[round_i] = { 'params': aggregated_params, 'params_path': aggregated_params_path }

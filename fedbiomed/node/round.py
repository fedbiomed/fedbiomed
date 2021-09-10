import sys
import uuid
import time

from fedbiomed.common.repository import Repository
from fedbiomed.common.message import NodeMessages
from fedbiomed.node.environ import CACHE_DIR, CLIENT_ID, TMP_DIR, UPLOADS_URL

from fedbiomed.common.logger import logger


class Round:
    """ This class repesents the training part execute by a node in a given round
    """
    def __init__(self,
                 model_kwargs: dict = None,
                 training_kwargs: dict = None,
                 dataset: dict = None,
                 model_url: str= None,
                 model_class: str = None,
                 params_url: str = None,
                 job_id: str = None,
                 researcher_id: str = None,
                 logger = None):
        """Constructor of the class

        Args:
            model_kwargs ([dict]): contains model args
            training_kwargs ([dict]): contains model args
            dataset ([dict]): dataset to use in this round
            repository_url ([str]): repository url where the hub function is stored
            hub_function ([str]): hub function
            init_params ([str]): url of init params file
            job_id ([str]): job id
            researcher_id ([str]): researcher id
            logger ([HistoryLogger])
        """
        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs
        self.dataset = dataset
        self.model_url = model_url
        self.model_class = model_class
        self.params_url = params_url
        self.job_id = job_id
        self.researcher_id = researcher_id
        self.logger = logger

        self.repository = Repository(UPLOADS_URL, TMP_DIR, CACHE_DIR)

    def run_model_training(self):
        """This method runs a model training and upload model params

        Returns:
            [NodeMessages]: returns the corresponding node message, trainReply instance
        """
        is_failed = False
        error_message = ''

        # Download model, training routine, execute it and return model results
        try:
            # module name cannot contain dashes
            import_module = 'my_model_' + str(uuid.uuid4().hex)
            status, _ = self.repository.download_file(self.model_url, import_module + '.py')
            if (status != 200):
                is_failed = True
                error_message = "Cannot download model file: " + self.model_url
            else:
                status, params_path = self.repository.download_file(self.params_url, 'my_model_' + str(uuid.uuid4()) + '.pt')
                if (status != 200) or params_path is None:
                    is_failed = True
                    error_message = "Cannot download param file: " + self.params_url
        except Exception as e:
            is_failed = True
            error_message = "Cannot download model files:" + str(e)

        # import module, declare the model, load parameters
        if not is_failed:
            try:
                sys.path.insert(0, TMP_DIR)
                exec('import ' + import_module,  globals())
                sys.path.pop(0)
                train_class = eval(import_module + '.' + self.model_class)
                if self.model_kwargs is None or len(self.model_kwargs)==0:
                    model = train_class()
                else:
                    model = train_class(self.model_kwargs)
            except Exception as e:
                is_failed = True
                error_message = "Cannot instantiate model object: " + str(e)

        # import model params into the model instance
        if not is_failed:
            try:
                model.load(params_path, to_params = False)
            except Exception as e:
                is_failed = True
                error_message = "Cannot initialize model parameters:" + str(e)

        # Run the training routine
        if not is_failed:
            results = {}
            try:
                training_kwargs_with_history = dict(logger=self.logger, **self.training_kwargs)
                logger.info(training_kwargs_with_history)
                model.set_dataset(self.dataset['path'])
                rtime_before = time.perf_counter()
                ptime_before = time.process_time()
                model.training_routine(**training_kwargs_with_history)
                rtime_after = time.perf_counter()
                ptime_after = time.process_time()
            except Exception as e:
                is_failed = True
                error_message = "Cannot train model: " + str(e)

        if not is_failed:
            # Upload results
            results['researcher_id'] = self.researcher_id
            results['job_id'] = self.job_id
            results['model_params'] = model.state_dict()
            results['history'] = self.logger.history
            results['client_id'] = CLIENT_ID
            try:
                # TODO : should test status code but not yet returned by upload_file
                filename = TMP_DIR + '/node_params_' + str(uuid.uuid4()) + '.pt'
                model.save(filename, results)
                res = self.repository.upload_file(filename)
                logger.info("results uploaded successfully ")
            except Exception as e:
                    is_failed = True
                    error_message = "Cannot upload results: " + str(e)

        # end : clean the namespace
        try:
            del model
            del import_module
        except:
            pass

        if not is_failed:
            return NodeMessages.reply_create({'client_id': CLIENT_ID,
                    'job_id': self.job_id, 'researcher_id': self.researcher_id,
                    'command': 'train', 'success': True, 'dataset_id': self.dataset['dataset_id'],
                    'params_url': res['file'], 'msg': '',
                    'timing': { 'rtime_training': rtime_after - rtime_before, 'ptime_training': ptime_after - ptime_before } }).get_dict()
        else:
            logging.error(error_message)
            return NodeMessages.reply_create({'client_id': CLIENT_ID,
                    'job_id': self.job_id, 'researcher_id': self.researcher_id,
                    'command': 'train', 'success': False, 'dataset_id': '',
                    'params_url': '', 'msg': error_message,
                    'timing': {} }).get_dict()

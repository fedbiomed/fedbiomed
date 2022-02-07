import sys
import os
import uuid
import time
from typing import Union

from fedbiomed.common.repository import Repository
from fedbiomed.common.message import NodeMessages, TrainReply
from fedbiomed.node.history_monitor import HistoryMonitor
from fedbiomed.node.model_manager import ModelManager
from fedbiomed.node.environ import environ
from fedbiomed.common.logger import logger

import traceback

class Round:
    """ This class repesents the training part execute by a node in a given round
    """
    def __init__(self,
                 model_kwargs: dict = None,
                 training_kwargs: dict = None,
                 dataset: dict = None,
                 model_url: str = None,
                 model_class: str = None,
                 params_url: str = None,
                 job_id: str = None,
                 researcher_id: str = None,
                 monitor: HistoryMonitor = None,
                 node_args: Union[dict, None] = None):

        """Constructor of the class

        Args:
            - model_kwargs (dict): contains model args
            - training_kwargs (dict): contains model characteristics,
                especially input  dimension (key: 'in_features')
                and output dimension (key: 'out_features')
            - dataset ([dict]): dataset details to use in this round.
                It contains the dataset name, dataset's id,
                data path, its shape, its
                description...
            - model_url (str): url from which to download model
            - model_class (str): name of the training plan
                (eg 'MyTrainingPlan')
            - params_url (str): url from which to upload/dowload model params
            - job_id (str): job id
            - researcher_id (str): researcher id
            - monitor (HistoryMonitor)
            - node_args (Union[dict, None]): command line arguments for node. Can include:
                - gpu (bool): propose use a GPU device if any is available.
                - gpu_num (Union[int, None]): if not None, use the specified GPU device instead of default
                    GPU device if this GPU device is available.
                - gpu_only (bool): force use of a GPU device if any available, even if researcher
                    doesnt request for using a GPU.
        """
        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs
        self.dataset = dataset
        self.model_url = model_url
        self.model_class = model_class
        self.params_url = params_url
        self.job_id = job_id
        self.researcher_id = researcher_id
        self.monitor = monitor
        self.model_manager = ModelManager()
        self.node_args = node_args
        self.repository = Repository(environ['UPLOADS_URL'], environ['TMP_DIR'], environ['CACHE_DIR'])

    def run_model_training(self) -> TrainReply:
        """This method downloads model file; then runs the training of a model
        and finally uploads model params

        Returns:
            [NodeMessages]: returns the corresponding node message,
            trainReply instance
        """
        is_failed = False
        error_message = ''

        # Download model, training routine, execute it and return model results
        try:
            # module name cannot contain dashes
            import_module = 'my_model_' + str(uuid.uuid4().hex)
            status, _ = self.repository.download_file(self.model_url,
                                                      import_module + '.py')

            if (status != 200):
                is_failed = True
                error_message = "Cannot download model file: " + self.model_url
            else:             
                if environ["MODEL_APPROVAL"]:
                    approved, model = self.model_manager.check_is_model_approved(os.path.join(environ["TMP_DIR"], import_module + '.py')) 
                    if not approved:
                        is_failed = True
                        error_message = f'Requested model is not approved by the node: {environ["NODE_ID"]}'
                    else:
                        logger.info(f'Model has been approved by the node {model["name"]}')
            
            if not is_failed:
                status, params_path = self.repository.download_file(
                    self.params_url,
                    'my_model_' + str(uuid.uuid4()) + '.pt')
                if (status != 200) or params_path is None:
                    is_failed = True
                    error_message = "Cannot download param file: "\
                        + self.params_url

        except Exception as e:
            is_failed = True
            # FIXME: this will trigger if model is not approved by node
            error_message = "Cannot download model files:" + str(e)

        # import module, declare the model, load parameters
        if not is_failed:
            try:
                sys.path.insert(0, environ['TMP_DIR'])
                # (below) import TrainingPlan created by Researcher on node
                exec('import ' + import_module,  globals())
                sys.path.pop(0)
                # (below) instantiate model as `train_class`
                train_class = eval(import_module + '.' + self.model_class)
                if self.model_kwargs is None or len(self.model_kwargs) == 0:
                    # case where no args have been found (default)
                    model = train_class()
                else:
                    # case where args have been found  (and passed)
                    model = train_class(self.model_kwargs)
            except Exception as e:
                is_failed = True
                error_message = "Cannot instantiate model object: " + str(e)

        # import model params into the model instance
        if not is_failed:
            try:
                model.load(params_path, to_params=False)
            except Exception as e:
                is_failed = True
                error_message = "Cannot initialize model parameters:" + str(e)

        # Run the training routine
        if not is_failed:
            # Caution: always provide values for node-side arguments
            # (monitor, node_args) especially if they are security
            # related, to avoid overloading by malicious researcher.
            #
            # We want to have explicit message in case of overloading attempt
            # (and continue training) though by default it fails with
            # "dict() got multiple values for keyword argument"
            node_side_args = [ 'monitor', 'node_args' ]
            for arg in node_side_args:
                if arg in self.training_kwargs:
                    del self.training_kwargs[arg]
                    logger.warning(f'Researcher trying to set node-side training parameter {arg}. '
                        f' Maybe a malicious researcher attack.')

        if not is_failed:
            training_kwargs_with_history = dict(monitor=self.monitor,
                                                node_args=self.node_args,
                                                **self.training_kwargs)
            logger.info(f'training with arguments {training_kwargs_with_history}')

        if not is_failed:
            try:
                results = {}
                model.set_dataset(self.dataset['path'])
                rtime_before = time.perf_counter()
                ptime_before = time.process_time()
                model.training_routine(**training_kwargs_with_history)
                rtime_after = time.perf_counter()
                ptime_after = time.process_time()
            except Exception as e:
                is_failed = True
                error_message = "Cannot train model in round: " + str(e)

        if not is_failed:
            # Upload results
            results['researcher_id'] = self.researcher_id
            results['job_id'] = self.job_id
            results['model_params'] = model.after_training_params()
            results['history'] = self.monitor.history
            results['node_id'] = environ['NODE_ID']
            try:
                # TODO : should test status code but not yet returned
                # by upload_file
                filename = environ['TMP_DIR'] + '/node_params_' + str(uuid.uuid4()) + '.pt'
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
        except Exception:
            pass

        if not is_failed:
            return NodeMessages.reply_create({'node_id': environ['NODE_ID'],
                        'job_id': self.job_id,
                        'researcher_id': self.researcher_id,
                        'command': 'train',
                        'success': True,
                        'dataset_id': self.dataset['dataset_id'],
                        'params_url': res['file'],
                        'msg': '',
                        'timing': {
                            'rtime_training': rtime_after - rtime_before,
                            'ptime_training': ptime_after - ptime_before }
                                  }).get_dict()
        else:
            logger.error(error_message)
            return NodeMessages.reply_create({'node_id': environ['NODE_ID'],
                        'job_id': self.job_id,
                        'researcher_id': self.researcher_id,
                        'command': 'train',
                        'success': False,
                        'dataset_id': '',
                        'params_url': '',
                        'msg': error_message,
                        'timing': {} }).get_dict()

import os
import hashlib
from typing import Any, Dict, Tuple
import uuid
from fedbiomed.common.exceptions import FedbiomedModelManagerError

from tinydb import TinyDB, Query
from datetime import datetime
from python_minifier import minify
from tabulate import tabulate

from fedbiomed.node.environ import environ
from fedbiomed.common.constants import HashingAlgorithms, ModelTypes
from fedbiomed.common.message import NodeMessages
from fedbiomed.common.repository import Repository
from fedbiomed.common.logger import logger


# Collect provided hashing function into a dict
HASH_FUNCTIONS = {
    HashingAlgorithms.SHA256.value    : hashlib.sha256,
    HashingAlgorithms.SHA384.value    : hashlib.sha384,
    HashingAlgorithms.SHA512.value    : hashlib.sha512,
    HashingAlgorithms.SHA3_256.value  : hashlib.sha3_256,
    HashingAlgorithms.SHA3_384.value  : hashlib.sha3_384,
    HashingAlgorithms.SHA3_512.value  : hashlib.sha3_512,
    HashingAlgorithms.BLAKE2B.value   : hashlib.blake2s,
    HashingAlgorithms.BLAKE2S.value   : hashlib.blake2s,
}

class ModelManager:


    def __init__(self):

        """ Class constructor for ModelManager. It creates a DB object \
            for the table named as `Models` and builds a query object to query
            the database.
        """
        self._tinydb = TinyDB(environ["DB_PATH"])
        self._db = self._tinydb.table('Models')
        self._database = Query()
        self._repo = Repository(environ['UPLOADS_URL'], environ['TMP_DIR'], environ['CACHE_DIR'])

    def _create_hash(self, path):

        """ Method for creating hash with given model file

            Args:
                path (str): Model file path

        """
        hash_algo = environ['HASHING_ALGORITHM']

        with open(path, "r") as model:

            # Minify model file using python_minifier module
            content = model.read()
            mini_content = minify( content,
                                   remove_annotations=False,
                                   combine_imports=False,
                                   remove_pass=False,
                                   hoist_literals=False,
                                   remove_object_base=True,
                                   rename_locals=False )

            # Hash model content based on active hashing algorithm
            if hash_algo in HashingAlgorithms.list():
                hashing = HASH_FUNCTIONS[hash_algo]()
            else:
                raise FedbiomedModelManagerError(f'Unkown hashing algorithm in the `environ` {environ["HASHING_ALGORITHM"]}')

        # Create hash from model minified model content and encoded as `utf-8`
        hashing.update(mini_content.encode('utf-8'))

        return hashing.hexdigest(), hash_algo


    def register_model(self,
                        name: str,
                        description: str,
                        path: str,
                        model_type: str = 'registered',
                        model_id: str = None
                        ) -> True:

        """ This method approves/registers model file thourgh CLI.

            Args:
                name        (str): Model file name. The name should be unique. Otherwise methods
                                   throws an Exception
                descripion  (str): Description fro model file.
                path        (str): Exact path for the model that will be registered
                model_type  (str): Default is `registered`. It means that model has been registered
                                   by a user/hospital. Other value can be `default` which indicates
                                   that model is default (models for tutorials/examples)
                model_id    (str): Pre-defined id for model. Default is None. When it is Nonde method
                                    creates unique id for the model.

        """

        # Check model type is valid
        if model_type not in ModelTypes.list():
            raise FedbiomedModelManagerError(f'Unkown model type: {model_type}')

        if not model_id:
            model_id = 'model_' + str(uuid.uuid4())

        # Check model path whether is registered before
        self._db.clear_cache()
        models_path_search = self._db.search(self._database.model_path == path)
        models_name_search = self._db.search(self._database.name == name)
        if models_path_search:
            raise FedbiomedModelManagerError(f'This model has been added already: {path}')
        elif models_name_search:
            raise FedbiomedModelManagerError(f'There is already a model added with same name: {name}. Please use different name')
        else:

            # Create hash and save it into db
            model_hash, algorithm = self._create_hash(path)
            # MOdel file cereation date
            ctime = datetime.fromtimestamp(os.path.getctime(path)).strftime("%d-%m-%Y %H:%M:%S.%f")
            # Model file modificaiton date
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d-%m-%Y %H:%M:%S.%f")
            # Model file registiration date
            rtime = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")

            model_object = dict(name=name, description=description,
                                hash=model_hash, model_path=path,
                                model_id=model_id, model_type=model_type,
                                algorithm=algorithm, date_created=ctime,
                                date_modified=mtime, date_registered=rtime)

            self._db.insert(model_object)

            return True

    def check_hashes_for_registered_models(self):

        """ This method checks regsitered models to make sure model files are exist
            and hashing algortihm is matched with specified algorithm in the config
            file
        """

        self._db.clear_cache()
        models = self._db.search(self._database.model_type.all('registered'))
        logger.info('Checking hashes for registered models...')
        if not models:
            logger.info('There is no models registered')
        else:
            for model in models:
                # If model file is exists
                if os.path.isfile(model['model_path']):
                    if model['algorithm'] != environ['HASHING_ALGORITHM']:
                        logger.info(f'Recreating hashing for : {model["name"]} \t {model["model_id"]}')
                        hashing, algorithm = self._create_hash(model['model_path'])
                        self._db.update( {'hash' : hashing, 'algorithm' : algorithm },
                                        self._database.model_id.all(model["model_id"]))
                else:
                    # Remove doc because model file is not exist anymore
                    logger.info(f'Model : {model["name"]} could not found in : {model["model_path"]}, will be removed')
                    self._db.remove(doc_ids=[model.doc_id])

    def check_is_model_approved(self, path) -> Tuple[bool, Dict[str, Any]]:

        """ This method checks wheter model is approved by the node. It send a query to
        database to search for hash of requested model. If it the hash matches with one of the
        models hashes in the DB, it approves requested model.

            Args:
                path (str): The path of requested model file by researcher after downloading
                            model file from file repository.
        """

        # Create hash for requested model
        req_model_hash, _ = self._create_hash(path)
        self._db.clear_cache()

        # If node allows defaults models search hash for all model types
        # otherwise search only for `registerd` models
        if environ['ALLOW_DEFAULT_MODELS']:
            models = self._db.search( self._database.hash == req_model_hash)
        else:
            models = self._db.search( (self._database.model_type == 'registered') & (self._database.hash == req_model_hash))

        if models:
            approved = True
            approved_model = models[0] # Search request returns an array
        else:
            approved = False
            approved_model = None

        return approved, approved_model

    def reply_model_status_request(self, msg, messaging):

        """ This method is called directly from Node.py when
        it recevies ModelStatusRequest. It checks requested model file
        whether it is approved or not and sends ModelStatusReply to
        researcher.

        Args:

            msg         (dict): Message that is receivied from researcher.
                                Formatted as ModelStatusRequest
            messaging   (MQTT):  MQTT client to send reply  to researcher
        """


        # Main header for the model status request
        header = {
            'researcher_id': msg['researcher_id'],
            'node_id': environ['NODE_ID'],
            'job_id': msg['job_id'],
            'model_url': msg['model_url'],
            'command': 'model-status'
        }

        try:

            # Create model file with id and downlioad
            model_name = 'my_model_' + str(uuid.uuid4().hex)
            status, _ = self._repo.download_file(msg['model_url'], model_name + '.py')
            if (status != 200):
                reply = { **header,
                            'success': False,
                            'approval_obligation' : False,
                            'is_approved' : False,
                            'msg': f'Can not download model file. {msg["model_url"]}'}
            else:
                if environ["MODEL_APPROVAL"]:
                    is_approved, _ = self.check_is_model_approved(os.path.join(environ["TMP_DIR"], model_name + '.py'))
                    if not is_approved:
                        reply = { **header,
                                'success' : True,
                                'approval_obligation' : True,
                                'is_approved' : False,
                                'msg' : 'Model is not approved by the node' }
                    else:
                        reply = { **header,
                                'success' : True,
                                'approval_obligation' : True,
                                'is_approved' : True,
                                'msg' : 'Model is approved by the node'}
                else:
                    reply = { **header,
                            'success' : True,
                            'approval_obligation' : False,
                            'is_approved' : False ,
                            'msg' : 'This node does not require model approval (maybe for debuging purposes). '}

        except Exception as e:
                reply = { **header,
                            'success': False,
                            'approval_obligation' : False,
                            'is_approved' : False,
                            'msg': f'An error occured when downloading model file. {msg["model_url"]} , {e}'}

        # Send check model status answer to researher
        messaging.send_message( NodeMessages.reply_create(reply).get_dict())

        return

    def register_update_default_models(self):

        """ This method registers or updated default methods. When the is started
        trhorugh CLI if environ['ALLOW_DEFAULT_MODELS'] is enabled. It will check the
        files saved into `default_models` directory and update/register them based
        on following conditions.

          - Registers: If there is a new modelfile which isn't saved into db
          - Updates: if model is modified
          - Updates: if hashing algorithm has changed in config file.
        """
        self._db.clear_cache()

        # Get model files saved in the directory
        models_file = os.listdir(environ['DEFAULT_MODELS_DIR'])

        # Get only default models from DB
        models = self._db.search(self._database.model_type == 'default')

        # Get model names from list of models
        models_name_db   = [ model['name'] for model in models]

        # Default models not in database
        models_not_saved = list(set(models_file) - set(models_name_db))
        # Defaults models that has been deleted from file system but not in DB
        models_deleted =  list(set(models_name_db) - set(models_file))
        # Models has already saved and exist in the database
        models_exists = list(set(models_file) - set(models_not_saved))

        # Register new default models
        for model in models_not_saved:
            self.register_model(name = model,
                                description = "Default model" ,
                                path = os.path.join(environ['DEFAULT_MODELS_DIR'], model),
                                model_type = 'default')

        # Remove models that has been removed from file system
        for model_name in models_deleted:
            model_doc = self._db.get(self._database.name == model_name )
            logger.info(f'Removed default model file has been detected, it will be removed from DB as well: {model_name}')
            self._db.remove(doc_ids = [model_doc.doc_id])

        # Update models
        for model in models_exists:
            path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            model_info = self._db.get(self._database.name == model)

            # Check if hasing algorithm has changed
            if model_info['algorithm'] != environ['HASHING_ALGORITHM']:
                logger.info(f'Recreating hashing for : {model_info["name"]} \t {model_info["model_id"]}')
                hash, algorithm = self._create_hash(os.path.join(environ['DEFAULT_MODELS_DIR'], model))
                self._db.update( {'hash' : hash, 'algorithm': algorithm},
                                self._database.model_path == path)
            # If default model file is modified update hashing
            elif mtime > datetime.strptime(model_info['date_modified'], "%d-%m-%Y %H:%M:%S.%f"):
                logger.info(f"Modified default model file has been detected. Hashing will be updated for: {model}")
                hash, algorithm = self._create_hash(os.path.join(environ['DEFAULT_MODELS_DIR'], model))
                self._db.update( {'hash' : hash, 'algorithm': algorithm,
                                'date_modified': mtime.strftime("%d-%m-%Y %H:%M:%S.%f") },
                                self._database.model_path == path)

    def update_model(self, model_id: str, path: str):

        """ Method for updating model files. Updates models hash value with provided
            model file. It also update `data_modified`, `date_created` and
            `model_path` in case of provided different model file than the other one.

            Args:

                model_id (str): Id of the model
                path     (str): The path where model file is stored
        """

        self._db.clear_cache()
        model = self._db.get(self._database.model_id == model_id)

        if model['model_type'] == ModelTypes.REGISTERED.value:

            # Get modification date
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            # Get creation date
            ctime = datetime.fromtimestamp(os.path.getctime(path))

            hash, algorithm = self._create_hash(path)
            self._db.update( {'hash' : hash, 'algorithm': algorithm,
                            'date_modified': mtime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                            'date_created' : ctime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                            'model_path' :  path },
                            self._database.model_id == model_id)
        else:
            raise FedbiomedModelManagerError(f'You cannot update default models. Please update them through their files saved in `default_models` directory and restart your node')

        return True

    def delete_model(self, model_id: str ):

        """ Remove model file from database. This model does not delete
        any registered model file and it only remove `registered` type of models.
        Default models should be removed from the directory

        Args:

            model_id  (str): The id of the registered model

        """

        self._db.clear_cache()
        model = self._db.get(self._database.model_id == model_id)

        if model['model_type'] == ModelTypes.REGISTERED.value:

            self._db.remove(doc_ids = [model.doc_id])
        else:
            raise Exception(f'For default models, please remove model file from `default_models` and restart your node')

        return True

    def list_approved_models(self, verbose: bool = True):

        """ Method for listing approved model files

            Args:
                verbose (bool): Default is True. When it is True, print
                                list of model in tabular format.
        """

        self._db.clear_cache()
        models = self._db.all()

        # Drop some keys for security reseasons
        for doc in models:
            doc.pop('model_path')
            doc.pop('hash')
            doc.pop('date_modified')
            doc.pop('date_created')

        if verbose:
            print(tabulate(models, headers='keys'))

        return models

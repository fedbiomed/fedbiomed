import os 
from tinydb import TinyDB, Query
from datetime import datetime
import hashlib
from fedbiomed.node.environ import environ
from fedbiomed.common.constants import HashingAlgorithms
from fedbiomed.common.logger import logger
from tabulate import tabulate
import uuid

class ModelManager:


    def __init__(self):

        """ Class constructur """

        self.tinydb = TinyDB(environ["DB_PATH"])
        self.db = self.tinydb.table('Models') 
        self.database = Query()

    def _create_hash(self, path):

        """ Method for creating hash with given model file"""

        with open(path, "r") as model:
            if environ['HASHING_ALGORITHM'] == "SHA256":
                hashing = hashlib.sha256()
                content = model.read()
                hashing.update(content.encode('utf-8'))
            else:
                hashing = hashlib.sha512()
                content = model.read()
                hashing.update(content.encode('utf-8'))

        return hashing.hexdigest(), environ['HASHING_ALGORITHM']    
    

    def register_model(self, 
                        name: str, 
                        description: str, 
                        path: str, 
                        model_type: str = 'registered',
                        model_id: str = None,
                        verbose : bool = False):

        """ This method is for approving/registering model file"""
        
        if not model_id:
            model_id = 'model_' + str(uuid.uuid4())

        # Check model path whether is registered before    
        self.db.clear_cache()
        models_path_search = self.db.search(self.database.model_path.all(path))
        models_name_search = self.db.search(self.database.name.all(name))
        if models_path_search: 
            raise Exception(f'This model has been added already: {path}')
        elif models_name_search:
            raise Exception(f'There already a model added by same name: {name}. Please use different name')
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

            self.db.insert(model_object)

            return True

    def check_hashes_for_registered_models(self):
        
        """ This method checks regsitered model to control model files are exist 
            or hash'ng algortihm has changed
        """

        self.db.clear_cache()
        models = self.db.search(self.database.model_type.all('registered'))
        algorithm = [doc['algorithm'] for doc in models ]

        if len(algorithm) > 0 and algorithm[0] != environ['HASHING_ALGORITHM']:
            logger.info("Hashing algorithm has been changed, model hashes is getting updated")
            for model in models:
                # If model file is exists
                if os.path.isfile(model['model_path']):
                    hashing, algorithm = self._create_hash(model['model_path'])
                    self.db.update( {'hash' : hashing, 'algorithm' : algorithm }, 
                                    self.database.model_id.all(model["model_id"]))
                else:
                    # Remove doc because its model file is not exist anymore
                    self.db.remove(doc_ids=[model.doc_id])
        else:
            logger.info(f'Current hashing algorithm : {environ["HASHING_ALGORITHM"]}')


    def check_is_model_approved(self, path):
        
        """ This method checks wheter model is approved by the node

            Args:
                path (str): The path of requested model file by researcher after downloading
                            model file from file repository. 
        """

        req_model_hash = self._create_hash(path)
        self.db.clear_cache()
        models = self.db.all()

        approved = False
        approved_model = None

        for model in models:
            if req_model_hash == model["hash"]:
                approved = True
                approved_model = model
        
        return approved, approved_model


    def register_update_default_models(self):

        """ This method registers or updated default methods. When the is started
        trhorugh CLI if environ['ALLOW_DEFAULT_MODELS'] is enabled. It will check the 
        files saved into `default_models` directory and update/register them based 
        on following conditions.

          - Registers: If there is a new modelfile which isn't saved into db
          - Updates: if model is modified
          - Updates: if hashing algorithm has changed in config file.   
        """

        models_path = os.path.join(environ["ROOT_DIR"], 'envs' , 'development' , 'default_models')
        
        # Get model files saved in the directory
        models_file = os.listdir(models_path)

        # Get only default models from DB
        models = self.db.search(self.database.model_type.all('default'))
        
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
                                path = os.path.join(models_path, model),
                                model_type = 'default')

        # Remove models that has been removed from file system
        for model in models_deleted:
            model = self.db.get(self.database.name.all(model))
            logger.info(f'Removed default model file has been detected, it will be removed from DB as well: {model}')
            self.db.remove(doc_ids = [model.doc_id])
            
        # Update models 
        for model in models_exists:
            path = os.path.join(models_path, model)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            model_info = self.db.get(self.database.name.all(model))

            # Check if hasing algorithm has changed
            if model_info['algorithm'] != environ['HASHING_ALGORITHM']:
                hash, algorithm = self._create_hash(os.path.join(models_path, model))
                self.db.update( {'hash' : hash, 'algorithm': algorithm}, 
                                self.database.model_path.all(path))
            # If default model file is modified update hashing 
            elif mtime > datetime.strptime(model_info['date_modified'], "%d-%m-%Y %H:%M:%S.%f"):
                logger.info(f"Modified default model file has been detected. Hashing will be updated for: {model}")
                hash, algorithm = self._create_hash(os.path.join(models_path, model))
                self.db.update( {'hash' : hash, 'algorithm': algorithm}, 
                                self.database.model_path.all(path))


    def delete_model(self, model_name: str ):
        pass

    def list_approved_models(self, verbose: bool = True):
        
        """Method for listing approved model files"""

        self.db.clear_cache()
        models = self.db.all()

        # Drop some keys for security reseasons
        for doc in models:
            doc.pop('model_path')
            doc.pop('hash')
            doc.pop('date_modified')
            doc.pop('date_created')

        if verbose:
            print(tabulate(models, headers='keys'))   
        
        return models



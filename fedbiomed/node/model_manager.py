import os 
from tinydb import TinyDB, Query
import hashlib
from fedbiomed.node.environ import environ
from fedbiomed.common.constants import SecurityLevels
from fedbiomed.common.logger import logger
from tabulate import tabulate
import uuid

class ModelManager:


    def __init__(self):

        """ Class constructur """

        self.db = TinyDB(environ["MODEL_DB_PATH"])
        self.database = Query()

    def _create_hash(self, path):

        """ Method for creating hash with given model file"""

        # Get hash algorithm 
        algorithm = self._get_hash_algorithm()

        with open(path, "r") as model:
            if algorithm == "SHA256":
                hashing = hashlib.sha256()
                content = model.read()
                hashing.update(content.encode('utf-8'))
            else:
                hashing = hashlib.sha512()
                content = model.read()
                hashing.update(content.encode('utf-8'))

        return hashing.hexdigest(), algorithm    
    

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
        if len(models_path_search) != 0 or len(models_name_search) != 0:
            if verbose: 
                logger.info('This model has been registered before, will not be added to database again')
        else:
            # Create hash and save it into db
            model_hash, algorithm = self._create_hash(path)
            model_object = dict( name=name, description=description, 
                                hash=model_hash, model_path=path, 
                                model_id=model_id, model_type=model_type, 
                                algorithm=algorithm) 

            self.db.insert(model_object)

    def update_hashes(self):

        self.db.clear_cache()
        models = self.db.all()
        algorithm = [doc['algorithm'] for doc in models ]
        hash_algo = self._get_hash_algorithm()

        if len(algorithm) > 0 and algorithm[0] != hash_algo:
            logger.info('Security level has been chaged updating model hashes')
            for model in models:
                print(model)
                hashing, algorithm = self._create_hash(model['model_path'])
                self.db.update( {'hash' : hashing, 'algorithm' : algorithm }, 
                                self.database.model_id.all(model["model_id"]) )
        else:
            logger.info(f'Security level : {environ["SECURITY_LEVEL"]}')


    @staticmethod
    def _get_hash_algorithm():

        """ Get hashing algorithm based on security level """

        return "SHA256" if environ['SECURITY_LEVEL'] == SecurityLevels.LOW.value else "SHA512"

    def check_is_model_approved(self, path):
        
        """ This method checks wheter model is approved by the node"""
        req_model_hash = self._create_hash(path)
        models = self.list_approved_models(verbose = False)

        approved = False
        approved_model = None

        for model in models:
            if req_model_hash == model["hash"]:
                approved = True
                approved_model = model
        
        return approved, approved_model


    def register_default_models(self):

        """ This method is for registering new default methods"""
        models_path = os.path.join(environ["ROOT_DIR"], 'envs' , 'development' , 'default_models')
        default_models = os.listdir(models_path)
        for model_file in default_models:
            model_name = 'default_' + model_file.split('.')[0]
            self.register_model(name = model_name, 
                                description = "Default model" , 
                                path = os.path.join(models_path, model_file),
                                model_type = 'default')

        
    def list_approved_models(self, verbose: bool = True):
        
        """Method for listing approved model files"""

        self.db.clear_cache()
        models = self.db.all()

        for doc in models:
            doc.pop('model_path')

        if verbose:
            print(tabulate(models, headers='keys'))   
        
        return models
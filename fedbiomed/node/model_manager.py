import os 
from tinydb import TinyDB, Query
import hashlib
from fedbiomed.node.environ import MODEL_DB_PATH, ROOT_DIR
from fedbiomed.common.logger import logger
from tabulate import tabulate
import uuid

class ModelManager:


    def __init__(self):

        """ Class constructur """

        self.db = TinyDB(MODEL_DB_PATH)
        self.database = Query()

    def _create_hash(self, path):

        """ Method for creating hash with given model file"""
     
        with open(path, "r") as model:
            hmd5 = hashlib.md5()
            content = model.read()
            hmd5.update(content.encode('utf-8'))

        return hmd5.hexdigest()    
    

    def register_model(self, 
                        name: str, 
                        description: str, 
                        path: str, 
                        model_id: str = None):

        """ This method is for approving/registering model file"""
        
        if not model_id:
            model_id = 'model_' + str(uuid.uuid4())

        # Check model path whether is registered before    
        self.db.clear_cache()
        models_path_search = self.db.search(self.database.model_path.all(path))
        models_name_search = self.db.search(self.database.name.all(name))
        if len(models_path_search) == 0 or len(models_name_search) == 0:
            logger.info('This model has been registered before')
            pass
  
        # Create hash and save it into db
        model_hash = self._create_hash(path)
        model_object = dict( name=name, description=description, 
                             hash=model_hash, model_path=path, 
                             model_id=model_id ) 
        self.db.insert(model_object)


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
        models_path = os.path.join(ROOT_DIR, 'envs' , 'development' , 'models')
        default_models = os.listdir(models_path)
        for model_file in default_models:
            model_name = 'default_' + model_file.split('.')[0]
            self.register_model(name = model_name, 
                                description = "Default model" , 
                                path = os.path.join(models_path, model_file) )

        
    def list_approved_models(self, verbose: bool = True):
        
        """Method for listing approved model files"""

        self.db.clear_cache()
        models = self.db.all()

        for doc in models:
            doc.pop('model_path')

        if verbose:
            print(tabulate(models, headers='keys'))   
        
        return models
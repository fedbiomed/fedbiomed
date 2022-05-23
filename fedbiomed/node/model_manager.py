"""Manages model approval for a node.
"""


from datetime import datetime
import hashlib
import os

from numpy import isin
from python_minifier import minify
import shutil
from tabulate import tabulate
from tinydb import TinyDB, Query
from typing import Any, Dict, List, Tuple, Union
import uuid

from fedbiomed.common.constants import HashingAlgorithms, ModelApprovalStatus, ModelTypes, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedDatasetManagerError, FedbiomedModelManagerError, FedbiomedRepositoryError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages
from fedbiomed.common.messaging import Messaging
from fedbiomed.common.repository import Repository

from fedbiomed.node.environ import environ


# Collect provided hashing function into a dict
HASH_FUNCTIONS = {
    HashingAlgorithms.SHA256.value: hashlib.sha256,
    HashingAlgorithms.SHA384.value: hashlib.sha384,
    HashingAlgorithms.SHA512.value: hashlib.sha512,
    HashingAlgorithms.SHA3_256.value: hashlib.sha3_256,
    HashingAlgorithms.SHA3_384.value: hashlib.sha3_384,
    HashingAlgorithms.SHA3_512.value: hashlib.sha3_512,
    HashingAlgorithms.BLAKE2B.value: hashlib.blake2s,
    HashingAlgorithms.BLAKE2S.value: hashlib.blake2s,
}


class ModelManager:
    """Manages model approval for a node.
    """
    def __init__(self):
        """Class constructor for ModelManager.

        Creates a DB object for the table named as `Models` and builds a query object to query
        the database.
        """
        self._tinydb = TinyDB(environ["DB_PATH"])
        self._db = self._tinydb.table('Models')
        self._database = Query()
        self._repo = Repository(environ['UPLOADS_URL'], environ['TMP_DIR'], environ['CACHE_DIR'])

    def _create_hash(self, path: str):
        """Creates hash with given model file

        Args:
            path: Model file path

        Raises:
            FedBiomedModelManagerError: file cannot be open
            FedBiomedModelManagerError: file cannot be minified
            FedBiomedModelManagerError: Hashing algorithm does not exist in HASH_FUNCTION table
        """
        hash_algo = environ['HASHING_ALGORITHM']

        try:
            with open(path, "r") as model:
                content = model.read()
        except FileNotFoundError:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + f" model file {path} not found on system")
        except PermissionError:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + f" cannot open model file {path} due" +
                                                                        " to unsatisfactory privelge")
        except OSError:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + f" cannot open model file {path} " +
                                             "(file might have been corrupted)")

        # Minify model file using python_minifier module
        try:
            mini_content = minify(content,
                                  remove_annotations=False,
                                  combine_imports=False,
                                  remove_pass=False,
                                  hoist_literals=False,
                                  remove_object_base=True,
                                  rename_locals=False)
        except Exception as err:
            # minify doesnot provide any specific exception
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + f"cannot minify file {path}"
                                             f"details: {err}")
        # Hash model content based on active hashing algorithm
        if hash_algo in HashingAlgorithms.list():
            hashing = HASH_FUNCTIONS[hash_algo]()
        else:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + 'Unknown hashing algorithm in the `environ`' +
                                             f' {environ["HASHING_ALGORITHM"]}')

        # Create hash from model minified model content and encoded as `utf-8`
        hashing.update(mini_content.encode('utf-8'))

        return hashing.hexdigest(), hash_algo

    def register_model(self,
                       name: str,
                       description: str,
                       path: str,
                       model_type: str = 'registered',
                       model_id: str = None,
                       researcher_id: str = None
                       ) -> True:
        """Approves/registers model file thourgh CLI.

        Args:
            name: Model file name. The name should be unique. Otherwise methods
                throws an Exception FedbiomedModelManagerError
            description: Description fro model file.
            path: Exact path for the model that will be registered
            model_type: Default is `registered`. It means that model has been registered
                by a user/hospital. Other value can be `default` which indicates
                that model is default (models for tutorials/examples)
            model_id: Pre-defined id for model. Default is None. When it is Nonde method
                creates unique id for the model.

        Raises:
            FedbiomedModelManagerError: `model_type` is not `registered` or `default`
            FedbiomedModelManagerError: model is already registered into database
            FedbiomedModelManagerError: model name is already used for saving another model

        Returns:
            Currently always returns True.
        """

        # Check model type is valid
        if model_type not in ModelTypes.list():
            raise FedbiomedModelManagerError(f'Unknown model type: {model_type}')

        if not model_id:
            model_id = 'model_' + str(uuid.uuid4())

        # Check model path whether is registered before
        self._db.clear_cache()

        models_path_search = self._db.search(self._database.model_path == path)
        models_name_search = self._db.search(self._database.name == name)

        if models_path_search:
            raise FedbiomedModelManagerError(f'This model has been added already: {path}')
        elif models_name_search:
            raise FedbiomedModelManagerError(f'There is already a model added with same name: {name}'
                                             '. Please use different name')
        else:

            # Create hash and save it into db
            model_hash, algorithm = self._create_hash(path)
            # Model file creation date
            ctime = datetime.fromtimestamp(os.path.getctime(path)).strftime("%d-%m-%Y %H:%M:%S.%f")
            # Model file modification date
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d-%m-%Y %H:%M:%S.%f")
            # Model file registration date
            rtime = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")

            model_object = dict(name=name, description=description,
                                hash=model_hash, model_path=path,
                                model_id=model_id, model_type=model_type,
                                model_status=ModelApprovalStatus.APPROVED.value,
                                algorithm=algorithm, date_created=ctime,
                                date_modified=mtime, date_registered=rtime,
                                date_last_action=rtime,
                                researcher_id=None)

            if researcher_id is not None:
                model_object.update({'researcher_id': researcher_id})
            try:
                self._db.insert(model_object)
            except ValueError as err:
                raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + " : database insertion failed with"
                                                 f" following error: {str(err)}")
            return True

    def check_hashes_for_registered_models(self):
        """Checks registered models.

        Makes sure model files exists and hashing algorithm is matched with specified
        algorithm in the config file.

        Raises:
            FedbiomedModelManagerError: cannot update model list in database
        """

        self._db.clear_cache()
        models = self._db.search(self._database.model_type.all('registered'))
        logger.info('Checking hashes for registered models')
        if not models:
            logger.info('There are no models registered')
        else:
            for model in models:
                # If model file is exists
                if os.path.isfile(model['model_path']):
                    if model['algorithm'] != environ['HASHING_ALGORITHM']:
                        logger.info(f'Recreating hashing for : {model["name"]} \t {model["model_id"]}')
                        hashing, algorithm = self._create_hash(model['model_path'])
                        rtime = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")
                        try:
                            self._db.update({'hash': hashing,
                                             'algorithm': algorithm,
                                             'date_last_action': rtime},
                                            self._database.model_id.all(model["model_id"]))
                        except ValueError as err:
                            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value +
                                                             ": database update failed, with error "
                                                             f" {str(err)}")
                else:
                    # Remove doc because model file is not exist anymore
                    logger.info(f'Model : {model["name"]} could not found in : {model["model_path"]}, will be removed')
                    try:
                        self._db.remove(doc_ids=[model.doc_id])
                    except RuntimeError as err:
                        raise FedbiomedModelManagerError(ErrorNumbers.FB606.value +
                                                         "database remove operation failed, with following error: ",
                                                         f"{str(err)}")

    def check_is_model_requested(self, path: str) -> bool:
        # Create hash for requested model
        req_model_hash, _ = self._create_hash(path)
        self._db.clear_cache()
        _all_requested_model = (self._database.model_type == ModelTypes.REQUESTED.value)
        _all_models_which_have_req_hash = (self._database.hash == req_model_hash)
        models = self._db.search(_all_requested_model & _all_models_which_have_req_hash)
        if models:
            requested = True
        else:
            requested = False

        return requested

    def check_is_model_approved(self, path: str) -> Tuple[bool, Dict[str, Any]]:
        """Checks whether model is approved by the node.

        Sends a query to database to search for hash of requested model.
        If it is the hash matches with one of the
        models hashes in the DB, it approves requested model.

        Args:
            path: The path of requested model file by researcher after downloading
                model file from file repository.

        Returns:
            A tuple (approved, approved_model) where

                - approved: Whether model has been approved or not
                - approved_model: Dictionary containing fields
                    related to the model. If database search request failed,
                    returns None instead.
        """

        # Create hash for requested model
        req_model_hash, _ = self._create_hash(path)
        self._db.clear_cache()

        # If node allows defaults models search hash for all model types
        # otherwise search only for `registered` models
        if environ['ALLOW_DEFAULT_MODELS']:
            _all_models_registered = (self._database.model_type != ModelTypes.REQUESTED.value)
            # _all_models_which_have_req_hash = (self._database.hash == req_model_hash)
            # models = self._db.search(_all_models_registered & _all_models_which_have_req_hash)
        else:
            _all_models_registered = (self._database.model_type == ModelTypes.REGISTERED.value)
        _all_models_which_have_req_hash = (self._database.hash == req_model_hash)
        models = self._db.search(_all_models_registered & _all_models_which_have_req_hash)

        if models:
            approved = True
            approved_model = models[0]  # Search request returns an array
        else:
            approved = False
            approved_model = None

        return approved, approved_model

    def create_py_model_from_txt(self, model_path: str) -> str:
        # remove '*.py' extension of `model_path` and rename it into `*.txt`
        model_path_txt, _ = os.path.splitext(model_path)
        model_path_txt += '.txt'

        # save the content of the model into a plain '*.txt' file
        shutil.copyfile(model_path, model_path_txt)
        return model_path_txt

    def reply_model_approval_request(self, msg: dict, messaging: Messaging):
        reply = {
            'researcher_id': msg['researcher_id'],
            'node_id': environ['NODE_ID'],
            # 'model_url': msg['model_url'],
            'sequence': msg['sequence'],
            'status': 0,  # HTTP status (set by default to 0, non existing HTTP status code)
            'command': 'approval'
        }

        is_approved = False
        non_downaloadable = False
        try:
            # model_id = str(uuid.uuid4())
            model_name = "model_" + str(uuid.uuid4())
            status, _ = self._repo.download_file(msg['model_url'], model_name + '.py')

            reply['status'] = status

            # check if model has already been registered into database
            tmp_file = os.path.join(environ["TMP_DIR"], model_name + '.py')
            model_to_check = self.create_py_model_from_txt(tmp_file)
            is_approved, _ = self.check_is_model_approved(model_to_check)

        except FedbiomedRepositoryError as fed_err:
            logger.error(f"Cannot download model from server due to error: {fed_err}")
            reply['success'] = False
            non_downaloadable = True
        except FedbiomedModelManagerError as fed_err:
            logger.error(f"Can not check whether model has already be registered or not due to error: {fed_err}")

        if not is_approved  and not non_downaloadable:
            # move model into corresponding directory (from TMP_DIR to MODEL_DIR)
            try:
                logger.debug("Storing TrainingPlan into requested model directory")
                model_path = os.path.join(environ['MODEL_DIR'], model_name + '.py')
                shutil.move(tmp_file, model_path)

                # Model file creation date
                ctime = datetime.fromtimestamp(os.path.getctime(model_path)).strftime("%d-%m-%Y %H:%M:%S.%f")

                model_hash, hash_algo = self._create_hash(model_to_check)
                model_object = dict(name=model_name,
                                    description = msg['description'],
                                    hash=model_hash,
                                    model_path=model_path,
                                    model_id=model_name,
                                    model_type=ModelTypes.REQUESTED.value,
                                    model_status=ModelApprovalStatus.PENDING.value,
                                    algorithm=hash_algo,
                                    date_created=ctime,
                                    date_modified=ctime,
                                    date_registered=ctime,
                                    date_last_action=None,
                                    researcher_id=msg['researcher_id']
                                    )

                if self.check_is_model_requested(model_to_check):
                    logger.debug("Model already awaiting for approval")
                self._db.upsert(model_object, self._database.hash == model_hash)
                # `upsert` stands for update and insert in TinyDB. This prevents any duplicate, that can happen
                # if same model is sent twice to Node for approval
                reply['success'] = True
                logger.debug("Model successfully received by Node for approval")
            except (PermissionError, FileNotFoundError, OSError) as err:
                reply['success'] = False
                logger.error(f"Cannot save model into directory due to error : {err}")
        elif is_approved and not non_downaloadable:
            logger.warning("Model has already been registered in database. aborting")
            reply['success'] = True
        else:
            # case where model is non-downloadable
            reply['success'] = False

        # Send model approval acknowledge answer to researcher
        messaging.send_message(NodeMessages.reply_create(reply).get_dict())

    def reply_model_status_request(self, msg: dict, messaging: Messaging):
        """Checks whether requested model file is approved or not and sends ModelStatusReply to
            researcher.

        Called directly from Node.py when it receives ModelStatusRequest.

        Args:
            msg: Message that is received from researcher. Formatted as ModelStatusRequest
            messaging: MQTT client to send reply  to researcher
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

            # Create model file with id and download
            model_name = 'my_model_' + str(uuid.uuid4().hex)
            status, _ = self._repo.download_file(msg['model_url'], model_name + '.py')
            if status != 200:
                # FIXME: should 'approval_obligation' be always false when model cannot be downloaded,
                # regardless of environment variable "MODEL_APPROVAL"?
                reply = {**header,
                         'success': False,
                         'approval_obligation': False,
                         'is_approved': False,
                         'msg': f'Can not download model file. {msg["model_url"]}'}
            else:
                if environ["MODEL_APPROVAL"]:
                    is_approved, _ = self.check_is_model_approved(os.path.join(environ["TMP_DIR"], model_name + '.py'))
                    if not is_approved:
                        reply = {**header,
                                 'success': True,
                                 'approval_obligation': True,
                                 'is_approved': False,
                                 'msg': 'Model is not approved by the node'}
                    else:
                        reply = {**header,
                                 'success': True,
                                 'approval_obligation': True,
                                 'is_approved': True,
                                 'msg': 'Model is approved by the node'}
                else:
                    reply = {**header,
                             'success': True,
                             'approval_obligation': False,
                             'is_approved': False,
                             'msg': 'This node does not require model approval (maybe for debuging purposes).'}
        except FedbiomedModelManagerError as fed_err:
            reply = {**header,
                     'success': False,
                     'approval_obligation': False,
                     'is_approved': False,
                     'msg': ErrorNumbers.FB606.value +
                     f': Cannot check if model has been registered. Details {fed_err}'}

        except FedbiomedRepositoryError as fed_err:
            reply = {**header,
                     'success': False,
                     'approval_obligation': False,
                     'is_approved': False,
                     'msg': ErrorNumbers.FB604.value + ': An error occured when downloading model file.'
                     f' {msg["model_url"]} , {fed_err}'}
        except Exception as e:
            reply = {**header,
                     'success': False,
                     'approval_obligation': False,
                     'is_approved': False,
                     'msg': ErrorNumbers.FB606.value + ': An unknown error occured when downloading model file.'
                     f' {msg["model_url"]} , {e}'}

        # Send check model status answer to researcher
        messaging.send_message(NodeMessages.reply_create(reply).get_dict())

        return

    def register_update_default_models(self):
        """Registers or updates default models.

        Launched when the node is started through CLI, if environ['ALLOW_DEFAULT_MODELS'] is enabled.
        Checks the files saved into `default_models` directory and update/register them based on following conditions:

        - Registers if there is a new model-file which isn't saved into db.
        - Updates if model is modified or if hashing algorithm has changed in config file.

        Raises:
            FedbiomedModelManagerError: cannot read or update model database
        """
        self._db.clear_cache()

        # Get model files saved in the directory
        models_file = os.listdir(environ['DEFAULT_MODELS_DIR'])

        # Get only default models from DB
        models = self._db.search(self._database.model_type == 'default')

        # Get model names from list of models
        models_name_db = [model.get('name') for model in models if isinstance(model, dict)]

        # Default models not in database
        models_not_saved = list(set(models_file) - set(models_name_db))
        # Default models that have been deleted from file system but not in DB
        models_deleted = list(set(models_name_db) - set(models_file))
        # Models have already saved and exist in the database
        models_exists = list(set(models_file) - set(models_not_saved))

        # Register new default models
        for model in models_not_saved:
            self.register_model(name=model,
                                description="Default model",
                                path=os.path.join(environ['DEFAULT_MODELS_DIR'], model),
                                model_type='default')

        # Remove models that have been removed from file system
        for model_name in models_deleted:
            try:
                model_doc = self._db.get(self._database.name == model_name)
                logger.info('Removed default model file has been detected,'
                            f' it will be removed from DB as well: {model_name}')

                self._db.remove(doc_ids=[model_doc.doc_id])
            except RuntimeError as err:
                raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + ": failed to update database, "
                                                 f" with error {str(err)}")
        # Update models
        for model in models_exists:
            path = os.path.join(environ['DEFAULT_MODELS_DIR'], model)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            try:
                model_info = self._db.get(self._database.name == model)
            except RuntimeError as err:
                raise FedbiomedModelManagerError(ErrorNumbers.FB606.value +
                                                 f": failed to get model _info for model {model}"
                                                 f"Details : {str(err)}")
            # Check if hashing algorithm has changed
            try:
                if model_info['algorithm'] != environ['HASHING_ALGORITHM']:
                    logger.info(f'Recreating hashing for : {model_info["name"]} \t {model_info["model_id"]}')
                    hash, algorithm = self._create_hash(os.path.join(environ['DEFAULT_MODELS_DIR'], model))
                    self._db.update({'hash': hash, 'algorithm': algorithm,
                                     'date_last_action': datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")},
                                    self._database.model_path == path)
                # If default model file is modified update hashing
                elif mtime > datetime.strptime(model_info['date_modified'], "%d-%m-%Y %H:%M:%S.%f"):
                    logger.info(f"Modified default model file has been detected. Hashing will be updated for: {model}")
                    hash, algorithm = self._create_hash(os.path.join(environ['DEFAULT_MODELS_DIR'], model))
                    self._db.update({'hash': hash, 'algorithm': algorithm,
                                     'date_modified': mtime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                                     'date_last_action': datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")},
                                    self._database.model_path == path)
            except ValueError as err:
                # triggered if database update failed (see `update` method in tinydb code)
                raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + ": Failed to update database, with error: "
                                                 f"{str(err)}")

    def update_model_hash(self, model_id: str, path: str) -> True:
        """Updates model file entry in model database.

        Updates model hash value for provided model file. It also updates
        `data_modified`, `date_created` and
        `model_path` in case the provided model file is different from the currently registered one.

        Args:
            model_id: Id of the model
            path: The path where model file is stored

        Returns:
            Currently always returns True.

        Raises:
            FedbiomedModelManagerError: cannot read or update the model in database
        """

        self._db.clear_cache()
        try:
            model = self._db.get(self._database.model_id == model_id)
        except RuntimeError as err:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + ": get request on database failed."
                                             f" Details: {str(err)}")
        if model['model_type'] != ModelTypes.DEFAULT.value:

            # Get modification date
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            # Get creation date
            ctime = datetime.fromtimestamp(os.path.getctime(path))

            hash, algorithm = self._create_hash(path)
            try:
                self._db.update({'hash': hash, 'algorithm': algorithm,
                                 'date_modified': mtime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                                 'date_created': ctime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                                 'date_last_action': datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f"),
                                 'model_path': path},
                                self._database.model_id == model_id)
            except ValueError as err:
                raise FedbiomedModelManagerError(ErrorNumbers.value + ": update database failed. Details :"
                                                 f"{str(err)}")
        else:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + 'You cannot update default models. Please '
                                             'update them through their files saved in `default_models` directory '
                                             'and restart your node')

        return True

    def update_model_status(self, model_id: str, model_status: ModelApprovalStatus) -> True:
        self._db.clear_cache()
        try:
            model = self._db.get(self._database.model_id == model_id)
        except RuntimeError as err:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + ": get request on database failed."
                                             f" Details: {str(err)}")
        if model['model_status'] == model_status.value:
            logger.warning(f" model {model_id} has already the following model status {model_status.value}")
            return
        else:
            model_path = model['model_path']
            # Get modification date
            mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
            # Get creation date
            ctime = datetime.fromtimestamp(os.path.getctime(model_path))
            self._db.update({'model_status': model_status.value,
                             'date_modified': mtime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                             'date_created': ctime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                             'date_last_action': datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")
                             },
                            self._database.model_id == model_id)
            logger.info(f"Model {model_id} status changed to {model_status.value} !")

        return True

    def approve_model(self, model_id: str) -> True:
        self.update_model_status(model_id, ModelApprovalStatus.APPROVED)

    def reject_model(self, model_id: str) -> True:
        self.update_model_status(model_id, ModelApprovalStatus.REJECTED)

    def delete_model(self, model_id: str) -> True:
        """Removes model file from database.

        Only removes `registered` type of models from the database.
        Does not remove the corresponding model file from the disk.
        Default models should be removed from the directory

        Args:
            model_id: The id of the registered model

        Returns:
            Currently always returns True.

        Raises:
            FedBiomedModelManagerError: cannot read model from the database
            FedBiomedModelManagerError: model is not a `registered` model (thus a `default` model)
        """

        self._db.clear_cache()
        try:
            model = self._db.get(self._database.model_id == model_id)

            if model['model_type'] == ModelTypes.REGISTERED.value:

                self._db.remove(doc_ids=[model.doc_id])
            else:
                raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + 'For default models, please remove'
                                                 ' model file from `default_models` and restart your node')
        except RuntimeError as err:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + ": cannot get model from database."
                                             f"Details: {str(err)}")
        return True

    def list_models(self, sort_by: Union[str, None] = None,
                    only: Union[None, ModelApprovalStatus, List[ModelApprovalStatus]] = None,
                    verbose: bool = True) -> List:
        """Lists approved model files

        Args:
            verbose: When it is True, print list of model in tabular format.
                Default is True.

        Returns:
            A list of models that have
                been found as `registered`. Each model is in fact a dictionary
                containing fields (note that following fields are removed :'model_path',
                'hash', dates due to privacy reasons).
        """

        self._db.clear_cache()

        if isinstance(only, (ModelApprovalStatus, list)):
            # filetring model based on their status
            if not isinstance(only, list):
                # convert everything into a list
                only = [only]
            only = [x.value for x in only if isinstance(x, ModelApprovalStatus)]
            # extract value from ModelApprovalStatus
            models = self._db.search(self._database.model_status.one_of(only))

        else:
            models = self._db.all()  
        # Drop some keys for security reasons
        _tags_to_remove = ['model_path',
                           'hash',
                           'date_modified',
                           'date_created']
        for doc in models:
            for tag_to_remove in _tags_to_remove:
                try:
                    doc.pop(tag_to_remove)
                except KeyError:
                    logger.warning(f"missing entry in database: {tag_to_remove} for model {doc}")
                    
        if sort_by is not None:
            # sorting model fields by column attributes
            if self._db.search(self._database[sort_by].exists()) and sort_by not in _tags_to_remove:
                models = sorted(models, key= lambda x: x[sort_by])
            else:
                logger.warning(f"Field {sort_by} is not available in dataset")

        

        if verbose:
            print(tabulate(models, headers='keys'))
        #print("MODELS", models)
        return models

    # def list_model(self, sort_by_date: bool = True, sort_by_status: bool = False,
    #                only: Union[None, ModelApprovalStatus] = None):
    #     pass
    

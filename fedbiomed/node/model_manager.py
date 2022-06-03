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
from fedbiomed.common.exceptions import FedbiomedModelManagerError, FedbiomedRepositoryError
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
                       model_type: str = ModelTypes.REGISTERED.value,
                       model_id: str = None,
                       researcher_id: str = None
                       ) -> True:
        """Approves/registers model file through CLI.

        Args:
            name: Model file name. The name should be unique. Otherwise, methods
                throws an Exception FedbiomedModelManagerError
            description: Description for model file.
            path: Exact path for the model that will be registered
            model_type: Default is `registered`. It means that model has been registered
                by a user/hospital. Other value can be `default` which indicates
                that model is default (models for tutorials/examples)
            model_id: Pre-defined id for model. Default is None. When it is Nonde method
                creates unique id for the model.
            researcher_id: ID of the researcher who is owner/requester of the model file

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
        model_hash, algorithm = self._create_hash(path)

        # Check model whether it was registered before
        self._db.clear_cache()

        try:
            models_path_search = self._db.get(self._database.model_path == path)
            models_name_search = self._db.get(self._database.name == name)
            models_hash_search = self._db.get(self._database.hash == model_hash)
        except RuntimeError as err:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + ": search request on database failed."
                                             f" Details: {str(err)}")        

        if models_path_search:
            raise FedbiomedModelManagerError(f'This model has been added already: {path}')
        elif models_name_search:
            raise FedbiomedModelManagerError(f'There is already a model added with same name: {name}'
                                             '. Please use different name')
        elif models_hash_search:
            raise FedbiomedModelManagerError('There is already an existing model in database same code hash, '
                                             f'model name is "{models_hash_search["name"]}"')
        else:

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
                                algorithm=algorithm,
                                researcher_id=researcher_id,
                                date_created=ctime,
                                date_modified=mtime,
                                date_registered=rtime,
                                date_last_action=rtime
                                )

            try:
                self._db.insert(model_object)
            except ValueError as err:
                raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + " : database insertion failed with"
                                                 f" following error: {str(err)}")
            return True

    def check_hashes_for_registered_models(self):
        """Checks registered models (models either rejected or approved).

        Makes sure model files exists and hashing algorithm is matched with specified
        algorithm in the config file.

        Raises:
            FedbiomedModelManagerError: cannot update model list in database
        """

        self._db.clear_cache()
        models = self._db.search(self._database.model_type.all(ModelTypes.REGISTERED.value))
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

    def check_model_status(self,
                           model_path: str,
                           state: Union[ModelApprovalStatus, ModelTypes, None]) -> Tuple[bool, Dict[str, Any]]:
        """Checks whether model has the specified status.

        Sends a query to database to search for hash of requested model.
        If it is the hash matches with one of the
        models hashes in the DB, and if model has the specified status {approved, rejected, pending} 
        or model_type {registered, requested, default}.

        Args:
            model_path: The path of requested model file by researcher after downloading
                model file from file repository.
            state: model status or model type, to check against model. `None` accepts
                any model status or type.

        Returns:
            A tuple (is_status, model) where

                - status: Whether model exists in database
                    with specified status (returns True) or not (False)
                - model: Dictionary containing fields
                    related to the model. If database search request failed,
                    returns None instead.
        """
        # Create hash for requested model
        req_model_hash, _ = self._create_hash(model_path)
        self._db.clear_cache()

        # If node allows defaults models search hash for all model types
        # otherwise search only for `registered` models

        if state is None:
            _all_models_with_status = None
        elif isinstance(state, ModelApprovalStatus):
            _all_models_with_status = (self._database.model_status == state.value)
        elif isinstance(state, ModelTypes):
            _all_models_with_status = (self._database.model_type == state.value)
        else:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + " : status should be either" + 
                                             f" ModelApprovalStatus or ModelTypes, but got {type(state)}")
        _all_models_which_have_req_hash = (self._database.hash == req_model_hash)

        # TODO: more robust implementation
        # current implementation (with `get`) makes supposition that there is at most
        # one model with a given hash in the database
        if _all_models_with_status is None:
            # check only against hash
            model = self._db.get(_all_models_which_have_req_hash)
        else:
            # check against hash and status
            model = self._db.get(_all_models_with_status & _all_models_which_have_req_hash)
   
        if model:
            is_status = True
        else:
            is_status = False
            model = None

        return is_status, model      

    def get_model_from_database(self, model_path: str) -> Union[Dict[str, Any], None]:
        """Gets model from database, by its hash

        Args:
            model_path: model path where the file is saved, in order to compute its hash
            !!! info "model file MUST be a *.txt file."

        Returns:
            model: model entry found in the dataset if query in database succeed. Otherwise, returns 
            None.
        """
        req_model_hash, _ = self._create_hash(model_path)
        self._db.clear_cache()
        _all_models_which_have_req_hash = (self._database.hash == req_model_hash)

        # TODO: more robust implementation
        # hashes in database should be unique, but we don't verify it
        # (and do we properly enforce it ?)
        model = self._db.get(_all_models_which_have_req_hash)

        if not model:
            model = None

        return model

    def create_txt_model_from_py(self, model_path: str) -> str:
        """Creates a text model file (*.txt extension) from a python (*.py) model file,
        in the directory where the python model file belongs to.

        Args:
            model_path (str): path to the model file (with *.py) extension

        Returns:
            model_path_txt (str): path to new model file (with *.txt extension)
        """
        # remove '*.py' extension of `model_path` and rename it into `*.txt`
        model_path_txt, _ = os.path.splitext(model_path)
        model_path_txt += '.txt'

        # save the content of the model into a plain '*.txt' file
        shutil.copyfile(model_path, model_path_txt)
        return model_path_txt

    def reply_model_approval_request(self, msg: dict, messaging: Messaging):
        """Submits a model file (TrainingPlan) for approval. Needs an action from Node

        Args:
            msg: approval request message, recieved from Researcher
            messaging: MQTT client to send reply  to researcher
        """
        reply = {
            'researcher_id': msg['researcher_id'],
            'node_id': environ['NODE_ID'],
            # 'model_url': msg['model_url'],
            'sequence': msg['sequence'],
            'status': 0,  # HTTP status (set by default to 0, non existing HTTP status code)
            'command': 'approval'
        }

        is_existant = False
        non_downaloadable = False

        try:
            # model_id = str(uuid.uuid4())
            model_name = "model_" + str(uuid.uuid4())
            status, _ = self._repo.download_file(msg['model_url'], model_name + '.py')

            reply['status'] = status

            # check if model has already been registered into database
            tmp_file = os.path.join(environ["TMP_DIR"], model_name + '.py')
            model_to_check = self.create_txt_model_from_py(tmp_file)
            is_existant, _ = self.check_model_status(model_to_check, None)

        except FedbiomedRepositoryError as fed_err:
            logger.error(f"Cannot download model from server due to error: {fed_err}")
            reply['success'] = False
            non_downaloadable = True
        except FedbiomedModelManagerError as fed_err:
            logger.error(f"Can not check whether model has already be registered or not due to error: {fed_err}")

        if not is_existant  and not non_downaloadable:
            # move model into corresponding directory (from TMP_DIR to MODEL_DIR)
            try:
                logger.debug("Storing TrainingPlan into requested model directory")
                model_path = os.path.join(environ['MODEL_DIR'], model_name + '.py')
                shutil.move(tmp_file, model_path)

                # Model file creation date
                ctime = datetime.fromtimestamp(os.path.getctime(model_path)).strftime("%d-%m-%Y %H:%M:%S.%f")

                model_hash, hash_algo = self._create_hash(model_to_check)
                model_object = dict(name=model_name,
                                    description=msg['description'],
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
                                    researcher_id=msg['researcher_id'],
                                    notes=None
                                    )

                self._db.upsert(model_object, self._database.hash == model_hash)
                # `upsert` stands for update and insert in TinyDB. This prevents any duplicate, that can happen
                # if same model is sent twice to Node for approval
                reply['success'] = True
                logger.debug(f"Model '{msg['description']}' successfully received by Node for approval")
            except (PermissionError, FileNotFoundError, OSError) as err:
                reply['success'] = False
                logger.error(f"Cannot save model '{msg['description']} 'into directory due to error : {err}")
        elif is_existant and not non_downaloadable:
            if self.check_model_status(model_to_check, ModelApprovalStatus.PENDING)[0]:
                logger.info(f"Model '{msg['description']}' already sent for Approval (status Pending). "
                            "Please wait for Node approval.")
            elif self.check_model_status(model_to_check, ModelApprovalStatus.APPROVED)[0]:
                logger.info(f"Model '{msg['description']}' is already Approved. Ready to train on this model.")
            else:
                logger.warning(f"Model '{msg['description']}' already exists in database. Aborting")
            reply['success'] = True
        else:
            # case where model is non-downloadable
            reply['success'] = False

        # Send model approval acknowledge answer to researcher
        messaging.send_message(NodeMessages.reply_create(reply).get_dict())

    def reply_model_status_request(self, msg: dict, messaging: Messaging):
        """Returns requested model file status {approved, rejected, pending}
        and sends ModelStatusReply to researcher.

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
                #  regardless of environment variable "MODEL_APPROVAL"?
                reply = {**header,
                         'success': False,
                         'approval_obligation': False,
                         'status': 'Error',
                         'msg': f'Can not download model file. {msg["model_url"]}'}
            else:
                model_file = os.path.join(environ["TMP_DIR"], model_name + '.py')
                model = self.get_model_from_database(model_file)
                if model is not None:
                    model_status = model.get('model_status', 'Not Registered')
                else:
                    model_status = 'Not Registered'

                if environ["MODEL_APPROVAL"]:
                    if model_status == ModelApprovalStatus.APPROVED.value:
                        msg = "Model has been approved by the node, training can start"
                    elif model_status == ModelApprovalStatus.PENDING.value:
                        msg = "Model is pending: waiting for a review"
                    elif model_status == ModelApprovalStatus.REJECTED.value:
                        msg = "Model has been rejected by the node, training is not possible"
                    else:
                        msg = f"Unknown model / model not in database (status {model_status})"
                    reply = {**header,
                             'success': True,
                             'approval_obligation': True,
                             'status': model_status,
                             'msg': msg}

                else:
                    reply = {**header,
                             'success': True,
                             'approval_obligation': False,
                             'status': model_status,
                             'msg': 'This node does not require model approval (maybe for debuging purposes).'}
        except FedbiomedModelManagerError as fed_err:
            reply = {**header,
                     'success': False,
                     'approval_obligation': False,
                     'status': 'Error',
                     'msg': ErrorNumbers.FB606.value +
                     f': Cannot check if model has been registered. Details {fed_err}'}

        except FedbiomedRepositoryError as fed_err:
            reply = {**header,
                     'success': False,
                     'approval_obligation': False,
                     'status': 'Error',
                     'msg': ErrorNumbers.FB604.value + ': An error occured when downloading model file.'
                     f' {msg["model_url"]} , {fed_err}'}
        except Exception as e:
            reply = {**header,
                     'success': False,
                     'approval_obligation': False,
                     'status': 'Error',
                     'msg': ErrorNumbers.FB606.value + ': An unknown error occured when downloading model file.'
                     f' {msg["model_url"]} , {e}'}
        # finally:
        #     # Send check model status answer to researcher
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
        hash, algorithm = self._create_hash(path)

        self._db.clear_cache()

        # Check if identical model already exists in database
        try:
            models_path_search = self._db.get(self._database.model_path == path)
            models_hash_search = self._db.get(self._database.hash == hash)
        except RuntimeError as err:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + ": search request on database failed."
                                             f" Details: {str(err)}")        

        if models_path_search:
            raise FedbiomedModelManagerError(f'This model has been added already: {path}')
        elif models_hash_search:
            raise FedbiomedModelManagerError('There is already an existing model in database same code hash, '
                                             f'model name is "{models_hash_search["name"]}"')

        # Register model
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

    def _update_model_status(self,
                             model_id: str,
                             model_status: ModelApprovalStatus, 
                             notes: Union[str, None] = None) -> True:
        """Updates model entry ([`model_status`] field) for a given [`model_id`] in the database

        Args:
            model_id: id of the model
            model_status: new model status {approved, rejected, pending}
            notes: additional notes to enter into the database, explaining why model
                has been approved or rejected for instance. Defaults to None.

        Returns:
            True: currently always returns True

        Raises:
            FedbiomedModelManagerError:         
        """
        self._db.clear_cache()
        try:
            model = self._db.get(self._database.model_id == model_id)
        except RuntimeError as err:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + ": get request on database failed."
                                             f" Details: {str(err)}")
        if model['model_status'] == model_status.value:
            logger.warning(f" model {model_id} has already the following model status {model_status.value}")
            return True
        else:
            model_path = model['model_path']
            # Get modification date
            mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
            # Get creation date
            ctime = datetime.fromtimestamp(os.path.getctime(model_path))
            self._db.update({'model_status': model_status.value,
                             'date_modified': mtime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                             'date_created': ctime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                             'date_last_action': datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f"),
                             'notes': notes},
                            self._database.model_id == model_id)
            logger.info(f"Model {model_id} status changed to {model_status.value} !")

        return True

    def approve_model(self, model_id: str, extra_notes: Union[str, None] = None) -> True:
        """Approves a model stored into the database given its [`model_id`] 

        Args:
            model_id: id of the model.
            extra_notes: notes detailing why model has been approved.
            Defaults to None.

        Returns:
            True: currently always returns True
        """
        res = self._update_model_status(model_id,
                                        ModelApprovalStatus.APPROVED,
                                        extra_notes)
        return res

    def reject_model(self, model_id: str, extra_notes: Union[str, None] = None) -> True:
        """Approves a model stored into the database given its [`model_id`] 

        Args:
            model_id: id of the model.
            extra_notes: notes detailing why model has been rejected.
            Defaults to None.

        Returns:
            True: currently always returns True
        """
        res = self._update_model_status(model_id,
                                        ModelApprovalStatus.REJECTED,
                                        extra_notes)
        return res

    def delete_model(self, model_id: str) -> True:
        """Removes model file from database.

        Only removes `registered` and `requested` type of models from the database.
        Does not remove the corresponding model file from the disk.
        Default models should be removed from the directory

        Args:
            model_id: The id of the registered model

        Returns:
            Currently always returns True.

        Raises:
            FedbiomedModelManagerError: cannot read model from the database
            FedbiomedModelManagerError: model is not a `registered` model (thus a `default` model)
        """

        self._db.clear_cache()
        try:
            model = self._db.get(self._database.model_id == model_id)

            if model['model_type'] != ModelTypes.DEFAULT.value:

                self._db.remove(doc_ids=[model.doc_id])
            else:
                raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + 'For default models, please remove'
                                                 ' model file from `default_models` and restart your node')
        except RuntimeError as err:
            raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + ": cannot get model from database."
                                             f"Details: {str(err)}")
        return True

    def list_models(self, sort_by: Union[str, None] = None,
                    select_status: Union[None, ModelApprovalStatus, List[ModelApprovalStatus]] = None,
                    verbose: bool = True) -> List:
        """Lists approved model files

        Args:
            sort_by: when specified, sort results by alphabetical order,
                provided sort_by is an entry in the database.
            select_status: filter list by model status or list of model statuses
            verbose: When it is True, print list of model in tabular format.
                Default is True.

        Returns:
            A list of models that have
                been found as `registered`. Each model is in fact a dictionary
                containing fields (note that following fields are removed :'model_path',
                'hash', dates due to privacy reasons).

        Raises: FedbiomedModelManagerError triggers if request to database failed
        """

        self._db.clear_cache()

        if isinstance(select_status, (ModelApprovalStatus, list)):
            # filtering model based on their status
            if not isinstance(select_status, list):
                # convert everything into a list
                select_status = [select_status]
            select_status = [x.value for x in select_status if isinstance(x, ModelApprovalStatus)]
            # extract value from ModelApprovalStatus
            try:
                models = self._db.search(self._database.model_status.one_of(select_status))
            except RuntimeError as rerr:
                raise FedbiomedModelManagerError(ErrorNumbers.FB606.value + 
                                                 ": request failed when looking for a model into database with" +
                                                 f" error: {rerr}")

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
                models = sorted(models, key= lambda x: (x[sort_by] is None, x[sort_by]))
            else:
                logger.warning(f"Field {sort_by} is not available in dataset")

        if verbose:
            print(tabulate(models, headers='keys'))

        return models

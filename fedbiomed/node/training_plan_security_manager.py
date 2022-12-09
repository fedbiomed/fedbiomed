# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manages training plan approval for a node.
"""

from datetime import datetime
import hashlib
import os
import re
from python_minifier import minify
import shutil
from tabulate import tabulate
from tinydb import TinyDB, Query, where
from typing import Any, Dict, List, Tuple, Union
import uuid

from fedbiomed.common.constants import HashingAlgorithms, TrainingPlanApprovalStatus, TrainingPlanStatus, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedTrainingPlanSecurityManagerError, FedbiomedRepositoryError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import NodeMessages
from fedbiomed.common.messaging import Messaging
from fedbiomed.common.repository import Repository
from fedbiomed.common.validator import SchemeValidator, ValidateError
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

trainingPlansSearchScheme = SchemeValidator({"by": {"rules": [str], "required": True},
                                             "text": {"rules": [str], "required": True}})


class TrainingPlanSecurityManager:
    """Manages training plan approval for a node.
    """

    def __init__(self):
        """Class constructor for TrainingPlanSecurityManager.

        Creates a DB object for the table named as `Training plans` and builds a query object to query
        the database.
        """

        self._tinydb = TinyDB(environ["DB_PATH"])
        # dont use DB read cache for coherence when updating from multiple sources (eg: GUI and CLI)
        self._db = self._tinydb.table(name="TrainingPlans", cache_size=0)
        self._database = Query()
        self._repo = Repository(environ['UPLOADS_URL'], environ['TMP_DIR'], environ['CACHE_DIR'])

        self._tags_to_remove = ['training_plan_path',
                                'hash',
                                'date_modified',
                                'date_created']

    @staticmethod
    def _create_hash(path: str):
        """Creates hash with given training plan file

        Args:
            path: Training plan file path

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad parameter type
            FedbiomedTrainingPlanSecurityManagerError: file cannot be open
            FedbiomedTrainingPlanSecurityManagerError: file cannot be minified
            FedbiomedTrainingPlanSecurityManagerError: Hashing algorithm does not exist in HASH_FUNCTION table
        """

        hash_algo = environ['HASHING_ALGORITHM']

        if not isinstance(path, str):
            raise FedbiomedTrainingPlanSecurityManagerError(ErrorNumbers.FB606.value + f': {path} is not a path')

        try:
            with open(path, "r") as training_plan:
                content = training_plan.read()
        except FileNotFoundError:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f": training plan file {path} not found on system")
        except PermissionError:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f": cannot open training plan file {path} due" +
                " to unsatisfactory privilege")
        except OSError:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f": cannot open training plan file {path} " +
                "(file might have been corrupted)")

        # Minify training plan file using python_minifier module
        try:
            mini_content = minify(content,
                                  remove_annotations=False,
                                  combine_imports=False,
                                  remove_pass=False,
                                  hoist_literals=False,
                                  remove_object_base=True,
                                  rename_locals=False)
        except Exception as err:
            # minify doesn't provide any specific exception
            raise FedbiomedTrainingPlanSecurityManagerError(ErrorNumbers.FB606.value + f": cannot minify file {path}"
                                                                                       f"details: {err}")
        # Hash training plan content based on active hashing algorithm
        if hash_algo in HashingAlgorithms.list():
            hashing = HASH_FUNCTIONS[hash_algo]()
        else:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + ': unknown hashing algorithm in the `environ`' +
                f' {environ["HASHING_ALGORITHM"]}')

        # Create hash from training plan minified training plan content and encoded as `utf-8`
        hashing.update(mini_content.encode('utf-8'))

        return hashing.hexdigest(), hash_algo

    def _check_training_plan_not_existing(self,
                                          name: Union[str, None] = None,
                                          path: Union[str, None] = None,
                                          hash_: Union[str, None] = None,
                                          algorithm: Union[str, None] = None) -> None:
        """Check no training plan exists in database that matches specified criteria.

        For each criterion, if criterion is not None, then check that no entry
        exists in database matching this criterion. Raise an exception if such
        entry exists.

        Hash and algorithm are checked together: if they both have non-None values,
        it is checked whether database contains an entry that both matches
        hash and algorithm.

        The current implementation of training plan database is based on the fact that:
        - training plan name is unique and
        - training plan path is unique and
        - pair of training plan hash plus hash algorithm is unique

        Args:
            name: Training plan name
            path: Training plan file path
            hash_: Training plan hash
            algorithm: Hashing algorithm

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: at least one training plan exists in DB matching a criterion
            FedbiomedTrainingPlanSecurityManagerError: database access problem
        """

        if name is not None:
            try:
                training_plans_name_get = self._db.get(self._database.name == name)
            except Exception as err:
                error = ErrorNumbers.FB606.value + ": search request on database failed." + \
                        f" Details: {str(err)}"
                logger.critical(error)
                raise FedbiomedTrainingPlanSecurityManagerError(error)
            if training_plans_name_get:
                error = ErrorNumbers.FB606.value + \
                        f': there is already a existing training plan with same name: "{name}"' + \
                        '. Please use different name'
                logger.critical(error)
                raise FedbiomedTrainingPlanSecurityManagerError(error)

        if path is not None:
            try:
                training_plans_path_get = self._db.get(self._database.training_plan_path == path)
            except Exception as err:
                error = ErrorNumbers.FB606.value + ": search request on database failed." + \
                        f" Details: {str(err)}"
                logger.critical(error)
                raise FedbiomedTrainingPlanSecurityManagerError(error)
            if training_plans_path_get:
                error = ErrorNumbers.FB606.value + f': this training plan has been added already: {path}'
                logger.critical(error)
                raise FedbiomedTrainingPlanSecurityManagerError(error)

        # TODO: to be more robust we should also check algorithm is the same
        if hash is not None or algorithm is not None:
            try:
                if algorithm is None:
                    training_plans_hash_get = self._db.get(self._database.hash == hash_)
                elif hash_ is None:
                    training_plans_hash_get = self._db.get(self._database.algorithm == algorithm)
                else:
                    training_plans_hash_get = self._db.get((self._database.hash == hash_) &
                                                           (self._database.algorithm == algorithm))
            except Exception as err:
                error = ErrorNumbers.FB606.value + ": search request on database failed." + \
                        f" Details: {str(err)}"
                logger.critical(error)
                raise FedbiomedTrainingPlanSecurityManagerError(error)
            if training_plans_hash_get:
                error = ErrorNumbers.FB606.value + \
                        ': there is already an existing training plan in database same code hash, ' + \
                        f'training plan name is "{training_plans_hash_get["name"]}"'
                logger.critical(error)
                raise FedbiomedTrainingPlanSecurityManagerError(error)

    def register_training_plan(self,
                               name: str,
                               description: str,
                               path: str,
                               training_plan_type: str = TrainingPlanStatus.REGISTERED.value,
                               training_plan_id: str = None,
                               researcher_id: str = None
                               ) -> True:
        """Approves/registers training plan file through CLI.

        Args:
            name: Training plan file name. The name should be unique. Otherwise, methods
                throws an Exception FedbiomedTrainingPlanSecurityManagerError
            description: Description for training plan file.
            path: Exact path for the training plan that will be registered
            training_plan_type: Default is `registered`. It means that training plan has been registered
                by a user/hospital. Other value can be `default` which indicates
                that training plan is default (training plans for tutorials/examples)
            training_plan_id: Pre-defined id for training plan. Default is None. When it is Nonde method
                creates unique id for the training plan.
            researcher_id: ID of the researcher who is owner/requester of the training plan file

        Returns:
            Currently always returns True

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: `training_plan_type` is not `registered` or `default`
            FedbiomedTrainingPlanSecurityManagerError: training plan is already registered into database
            FedbiomedTrainingPlanSecurityManagerError: training plan name is already used for saving another training plan
            FedbiomedTrainingPlanSecurityManagerError: database access problem
        """

        # Check training plan type is valid
        if training_plan_type not in TrainingPlanStatus.list():
            raise FedbiomedTrainingPlanSecurityManagerError(
                f'Unknown training plan (training_plan_type) type: {training_plan_type}')

        if not training_plan_id:
            training_plan_id = 'training_plan_' + str(uuid.uuid4())
        training_plan_hash, algorithm = self._create_hash(path)

        # Verify no such training plan is already registered
        self._check_training_plan_not_existing(name, path, training_plan_hash, algorithm)

        # Training plan file creation date
        ctime = datetime.fromtimestamp(os.path.getctime(path)).strftime("%d-%m-%Y %H:%M:%S.%f")
        # Training plan file modification date
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d-%m-%Y %H:%M:%S.%f")
        # Training plan file registration date
        rtime = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")

        training_plan_record = dict(name=name, description=description,
                                    hash=training_plan_hash, training_plan_path=path,
                                    training_plan_id=training_plan_id, training_plan_type=training_plan_type,
                                    training_plan_status=TrainingPlanApprovalStatus.APPROVED.value,
                                    algorithm=algorithm,
                                    researcher_id=researcher_id,
                                    date_created=ctime,
                                    date_modified=mtime,
                                    date_registered=rtime,
                                    date_last_action=rtime
                                    )

        try:
            self._db.insert(training_plan_record)
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + " : database insertion failed with"
                                           f" following error: {str(err)}")
        return True

    def check_hashes_for_registered_training_plans(self):
        """Checks registered training plans (training plans either rejected or approved).

        Makes sure training plan files exists and hashing algorithm is matched with specified
        algorithm in the config file.

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: cannot update training plan list in database
        """

        try:
            training_plans = self._db.search(self._database.training_plan_type.all(TrainingPlanStatus.REGISTERED.value))
        except Exception as e:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f"database search operation failed, with following error: {str(e)}")
        logger.info('Checking hashes for registered training plans')
        if not training_plans:
            logger.info('There are no training plans registered')
        else:
            for training_plan in training_plans:
                # If training plan file is exists
                if os.path.isfile(training_plan['training_plan_path']):
                    if training_plan['algorithm'] != environ['HASHING_ALGORITHM']:
                        logger.info(
                            f'Recreating hashing for : {training_plan["name"]} \t {training_plan["training_plan_id"]}')
                        hashing, algorithm = self._create_hash(training_plan['training_plan_path'])

                        # Verify no such training plan already exists in DB
                        self._check_training_plan_not_existing(None, None, hashing, algorithm)

                        rtime = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")
                        try:
                            self._db.update({'hash': hashing,
                                             'algorithm': algorithm,
                                             'date_last_action': rtime},
                                            self._database.training_plan_id.all(training_plan["training_plan_id"]))
                        except Exception as err:
                            raise FedbiomedTrainingPlanSecurityManagerError(ErrorNumbers.FB606.value +
                                                                            ": database update failed, with error "
                                                                            f" {str(err)}")
                else:
                    # Remove doc because training plan file is not existing anymore
                    logger.info(
                        f'Training plan : {training_plan["name"]} could not found in : '
                        f'{training_plan["training_plan_path"]}, will be removed')
                    try:
                        self._db.remove(doc_ids=[training_plan.doc_id])
                    except Exception as err:
                        raise FedbiomedTrainingPlanSecurityManagerError(
                            f"{ErrorNumbers.FB606.value}: database remove operation failed, with following error: ",
                            f"{err}")

    def check_training_plan_status(self,
                                   training_plan_path: str,
                                   state: Union[TrainingPlanApprovalStatus, TrainingPlanStatus, None]) \
            -> Tuple[bool, Dict[str, Any]]:
        """Checks whether training plan exists in database and has the specified status.

        Sends a query to database to search for hash of requested training plan.
        If the hash matches with one of the
        training plans hashes in the DB, and if training plan has the specified status {approved, rejected, pending}
        or training_plan_type {registered, requested, default}.

        Args:
            training_plan_path: The path of requested training plan file by researcher after downloading
                training plan file from file repository.
            state: training plan status or training plan type, to check against training plan. `None` accepts
                any training plan status or type.

        Returns:
            A tuple (is_status, training plan) where

                - status: Whether training plan exists in database
                    with specified status (returns True) or not (False)
                - training_plan: Dictionary containing fields
                    related to the training plan. If database search request failed,
                    returns None instead.

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad parameter type or value
            FedbiomedTrainingPlanSecurityManagerError: database access problem
        """

        # Create hash for requested training plan
        req_training_plan_hash, _ = self._create_hash(training_plan_path)

        # If node allows defaults training plans search hash for all training plan types
        # otherwise search only for `registered` training plans

        if state is None:
            _all_training_plans_with_status = None
        elif isinstance(state, TrainingPlanApprovalStatus):
            _all_training_plans_with_status = (self._database.training_plan_status == state.value)
        elif isinstance(state, TrainingPlanStatus):
            _all_training_plans_with_status = (self._database.training_plan_type == state.value)
        else:
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value} + status should be either TrainingPlanApprovalStatus or "
                f"TrainingPlanStatus, but got {type(state)}"
            )
        _all_training_plans_which_have_req_hash = (self._database.hash == req_training_plan_hash)

        # TODO: more robust implementation
        # current implementation (with `get`) makes supposition that there is at most
        # one training plan with a given hash in the database
        try:
            if _all_training_plans_with_status is None:
                # check only against hash
                training_plan = self._db.get(_all_training_plans_which_have_req_hash)
            else:
                # check against hash and status
                training_plan = self._db.get(_all_training_plans_with_status & _all_training_plans_which_have_req_hash)
        except Exception as e:
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value} database remove operation failed, with following error: {e}"
            )

        if training_plan:
            is_status = True
        else:
            is_status = False
            training_plan = None

        return is_status, training_plan

    def get_training_plan_by_name(self, training_plan_name: str) -> Union[Dict[str, Any], None]:
        """Gets training plan from database, by its name

        Args:
            training_plan_name: name of the training plan entry to search in the database

        Returns:
            training plan entry found in the database matching `training_plan_name`. Otherwise, returns None.

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad parameter type
            FedbiomedTrainingPlanSecurityManagerError: cannot read database.
        """

        if not isinstance(training_plan_name, str):
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value} training plan name {training_plan_name} is not a string"
            )

        # TODO: more robust implementation
        # names in database should be unique, but we don't verify it
        # (and do we properly enforce it ?)
        try:
            training_plan = self._db.get(self._database.name == training_plan_name)
        except Exception as e:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + ': cannot search database for training plan '
                                           f' "{training_plan_name}", error is "{e}"')

        if not training_plan:
            training_plan = None
        return training_plan

    def get_training_plan_from_database(self,
                                        training_plan_path: str
                                        ) -> Union[Dict[str, Any], None]:
        """Gets training plan from database, by its hash

        !!! info "Training plan file MUST be a *.txt file."

        Args:
            training_plan_path: training plan path where the file is saved, in order to compute its hash.

        Returns:
            training_plan: training plan entry found in the dataset if query in database succeed. Otherwise, returns
            None.

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad parameter type
            FedbiomedTrainingPlanSecurityManagerError: database access problem
        """

        if not isinstance(training_plan_path, str):
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + " : no training_plan_path specified")
        req_training_plan_hash, _ = self._create_hash(training_plan_path)

        _all_training_plans_which_have_req_hash = (self._database.hash == req_training_plan_hash)

        # TODO: more robust implementation
        # hashes in database should be unique, but we don't verify it
        # (and do we properly enforce it ?)
        try:
            training_plan = self._db.get(_all_training_plans_which_have_req_hash)
        except Exception as e:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f"database get operation failed, with following error: {str(e)}")

        if not training_plan:
            training_plan = None
        return training_plan

    def get_training_plan_by_id(self,
                                training_plan_id: str,
                                secure: bool = True,
                                content: bool = False) -> Union[Dict[str, Any], None]:
        """Get a training plan in database given his `training_plan_id`.

        Also add a `content` key to the returned dictionary.

        Args:
            training_plan_id: id of the training plan to pick from the database
            secure: if `True` then strip some security sensitive fields
            content: if `True` add content of training plan in `content` key of returned training plan. If `False` then
                `content` key value is `None`


        Returns:
            training plan entry from database through a query based on the training plan_id.
            If there is no training plan matching [`training_plan_id`], returns None

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad parameter type
            FedbiomedTrainingPlanSecurityManagerError: database access problem
        """

        if not isinstance(training_plan_id, str):
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f': training_plan_id {training_plan_id} is not a string')

        try:
            training_plan = self._db.get(self._database.training_plan_id == training_plan_id)
        except Exception as e:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f"database get operation failed, with following error: {str(e)}")

        if isinstance(training_plan, dict):
            if content:
                with open(training_plan["training_plan_path"], 'r') as file:
                    training_plan_content = file.read()
            else:
                training_plan_content = None

            if secure and training_plan is not None:
                self._remove_sensible_keys_from_request(training_plan)

            training_plan.update({"content": training_plan_content})

        return training_plan

    @staticmethod
    def create_txt_training_plan_from_py(training_plan_path: str) -> str:
        """Creates a text training plan file (*.txt extension) from a python (*.py) training plan file,
        in the directory where the python training plan file belongs to.

        Args:
            training_plan_path (str): path to the training plan file (with *.py) extension

        Returns:
            training_plan_path_txt (str): path to new training plan file (with *.txt extension)
        """
        # remove '*.py' extension of `training_plan_path` and rename it into `*.txt`
        training_plan_path_txt, _ = os.path.splitext(training_plan_path)
        training_plan_path_txt += '.txt'

        # save the content of the training plan into a plain '*.txt' file
        shutil.copyfile(training_plan_path, training_plan_path_txt)
        return training_plan_path_txt

    def reply_training_plan_approval_request(self, msg: dict, messaging: Messaging):
        """Submits a training plan file (TrainingPlan) for approval. Needs an action from Node

        Args:
            msg: approval request message, received from Researcher
            messaging: MQTT client to send reply  to researcher
        """

        reply = {
            'researcher_id': msg['researcher_id'],
            'node_id': environ['NODE_ID'],
            # 'training_plan_url': msg['training_plan_url'],
            'sequence': msg['sequence'],
            'status': 0,  # HTTP status (set by default to 0, non-existing HTTP status code)
            'command': 'approval'
        }

        is_existant = False
        downloadable_checkable = True

        try:
            # training_plan_id = str(uuid.uuid4())
            training_plan_name = "training_plan_" + str(uuid.uuid4())
            status, tmp_file = self._repo.download_file(msg['training_plan_url'], training_plan_name + '.py')

            reply['status'] = status

            # check if training plan has already been registered into database
            training_plan_to_check = self.create_txt_training_plan_from_py(tmp_file)
            is_existant, _ = self.check_training_plan_status(training_plan_to_check, None)

        except FedbiomedRepositoryError as fed_err:
            logger.error(f"Cannot download training plan from server due to error: {fed_err}")
            downloadable_checkable = False
        except FedbiomedTrainingPlanSecurityManagerError as fed_err:
            downloadable_checkable = False
            logger.error(
                f"Can not check whether training plan has already be registered or not due to error: {fed_err}")

        if not is_existant and downloadable_checkable:
            # move training plan into corresponding directory (from TMP_DIR to TRAINING_PLANS_DIR)
            try:
                logger.debug("Storing TrainingPlan into requested training plan directory")
                training_plan_path = os.path.join(environ['TRAINING_PLANS_DIR'], training_plan_name + '.py')
                shutil.move(tmp_file, training_plan_path)

                # Training plan file creation date
                ctime = datetime.fromtimestamp(os.path.getctime(training_plan_path)).strftime("%d-%m-%Y %H:%M:%S.%f")
            except (PermissionError, FileNotFoundError, OSError) as err:
                reply['success'] = False
                logger.error(f"Cannot save training plan '{msg['description']} 'into directory due to error : {err}")
            else:
                try:
                    training_plan_hash, hash_algo = self._create_hash(training_plan_to_check)
                    training_plan_object = dict(name=training_plan_name,
                                                description=msg['description'],
                                                hash=training_plan_hash,
                                                training_plan_path=training_plan_path,
                                                training_plan_id=training_plan_name,
                                                training_plan_type=TrainingPlanStatus.REQUESTED.value,
                                                training_plan_status=TrainingPlanApprovalStatus.PENDING.value,
                                                algorithm=hash_algo,
                                                date_created=ctime,
                                                date_modified=ctime,
                                                date_registered=ctime,
                                                date_last_action=None,
                                                researcher_id=msg['researcher_id'],
                                                notes=None
                                                )

                    self._db.upsert(training_plan_object, self._database.hash == training_plan_hash)
                    # `upsert` stands for update and insert in TinyDB. This prevents any duplicate, that can happen
                    # if same training plan is sent twice to Node for approval
                except Exception as err:
                    reply['success'] = False
                    logger.error(f"Cannot add training plan '{msg['description']} 'into database due to error : {err}")
                else:
                    reply['success'] = True
                    logger.debug(f"Training plan '{msg['description']}' successfully received by Node for approval")

        elif is_existant and downloadable_checkable:
            if self.check_training_plan_status(training_plan_to_check, TrainingPlanApprovalStatus.PENDING)[0]:
                logger.info(f"Training plan '{msg['description']}' already sent for Approval (status Pending). "
                            "Please wait for Node approval.")
            elif self.check_training_plan_status(training_plan_to_check, TrainingPlanApprovalStatus.APPROVED)[0]:
                logger.info(
                    f"Training plan '{msg['description']}' is already Approved. Ready to train on this training plan.")
            else:
                logger.warning(f"Training plan '{msg['description']}' already exists in database. Aborting")
            reply['success'] = True
        else:
            # case where training plan is non-downloadable or non-checkable
            reply['success'] = False

        # Send training plan approval acknowledge answer to researcher
        messaging.send_message(NodeMessages.reply_create(reply).get_dict())

    def reply_training_plan_status_request(self, msg: dict, messaging: Messaging):
        """Returns requested training plan file status {approved, rejected, pending}
        and sends TrainingPlanStatusReply to researcher.

        Called directly from Node.py when it receives TrainingPlanStatusRequest.

        Args:
            msg: Message that is received from researcher.
                Formatted as TrainingPlanStatusRequest
            messaging: MQTT client to send reply  to researcher
        """

        # Main header for the training plan status request
        header = {
            'researcher_id': msg['researcher_id'],
            'node_id': environ['NODE_ID'],
            'job_id': msg['job_id'],
            'training_plan_url': msg['training_plan_url'],
            'command': 'training-plan-status'
        }

        try:
            # Create training plan file with id and download
            training_plan_name = 'my_training_plan_' + str(uuid.uuid4().hex)
            status, training_plan_file = self._repo.download_file(msg['training_plan_url'], training_plan_name + '.py')
            if status != 200:
                # FIXME: should 'approval_obligation' be always false when training plan cannot be downloaded,
                #  regardless of environment variable "TRAINING_PLAN_APPROVAL"?
                reply = {**header,
                         'success': False,
                         'approval_obligation': False,
                         'status': 'Error',
                         'msg': f'Can not download training plan file. {msg["training_plan_url"]}'}
            else:
                training_plan = self.get_training_plan_from_database(training_plan_file)
                if training_plan is not None:
                    training_plan_status = training_plan.get('training_plan_status', 'Not Registered')
                else:
                    training_plan_status = 'Not Registered'

                if environ["TRAINING_PLAN_APPROVAL"]:
                    if training_plan_status == TrainingPlanApprovalStatus.APPROVED.value:
                        msg = "Training plan has been approved by the node, training can start"
                    elif training_plan_status == TrainingPlanApprovalStatus.PENDING.value:
                        msg = "Training plan is pending: waiting for a review"
                    elif training_plan_status == TrainingPlanApprovalStatus.REJECTED.value:
                        msg = "Training plan has been rejected by the node, training is not possible"
                    else:
                        msg = f"Unknown training plan not in database (status {training_plan_status})"
                    reply = {**header,
                             'success': True,
                             'approval_obligation': True,
                             'status': training_plan_status,
                             'msg': msg}

                else:
                    reply = {**header,
                             'success': True,
                             'approval_obligation': False,
                             'status': training_plan_status,
                             'msg': 'This node does not require training plan approval (maybe for debugging purposes).'}
        except FedbiomedTrainingPlanSecurityManagerError as fed_err:
            reply = {**header,
                     'success': False,
                     'approval_obligation': False,
                     'status': 'Error',
                     'msg': ErrorNumbers.FB606.value +
                            f': Cannot check if training plan has been registered. Details {fed_err}'}
        except FedbiomedRepositoryError as fed_err:
            reply = {**header,
                     'success': False,
                     'approval_obligation': False,
                     'status': 'Error',
                     'msg': f'{ErrorNumbers.FB604.value}: An error occurred when downloading training plan file. '
                            f'{msg["training_plan_url"]} , {fed_err}'}
        except Exception as e:
            reply = {**header,
                     'success': False,
                     'approval_obligation': False,
                     'status': 'Error',
                     'msg': f'{ErrorNumbers.FB606.value}: An unknown error occurred when downloading training plan '
                            f'file. {msg["training_plan_url"]} , {e}'}
        # finally:
        #     # Send check training plan status answer to researcher
        messaging.send_message(NodeMessages.reply_create(reply).get_dict())

        return

    def register_update_default_training_plans(self):
        """Registers or updates default training plans.

        Launched when the node is started through CLI, if environ['ALLOW_DEFAULT_TRAINING_PLANS'] is enabled.
        Checks the files saved into `default_training_plans` directory and update/register them based on following
        conditions:

        - Registers if there is a new training plan file which isn't saved into db.
        - Updates if training plan is modified or if hashing algorithm has changed in config file.

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: cannot read or update training plan database
        """

        # Get training plan files saved in the directory
        training_plans_file = os.listdir(environ['DEFAULT_TRAINING_PLANS_DIR'])

        # Get only default training plans from DB
        try:
            training_plans = self._db.search(self._database.training_plan_type == 'default')
        except Exception as e:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f"database search operation failed, with following error: {str(e)}")

        # Get training plan names from list of training plans
        training_plans_name_db = [training_plan.get('name') for training_plan in training_plans if
                                  isinstance(training_plan, dict)]

        # Default training plans not in database
        training_plans_not_saved = list(set(training_plans_file) - set(training_plans_name_db))
        # Default training plans that have been deleted from file system but not in DB
        training_plans_deleted = list(set(training_plans_name_db) - set(training_plans_file))
        # Training plans have already saved and exist in the database
        training_plans_exists = list(set(training_plans_file) - set(training_plans_not_saved))

        # Register new default training plans
        for training_plan in training_plans_not_saved:
            self.register_training_plan(name=training_plan,
                                        description="Default training plan",
                                        path=os.path.join(environ['DEFAULT_TRAINING_PLANS_DIR'], training_plan),
                                        training_plan_type='default')

        # Remove training plans that have been removed from file system
        for training_plan_name in training_plans_deleted:
            try:
                training_plan_doc = self._db.get(self._database.name == training_plan_name)
                logger.info('Removed default training plan file has been detected,'
                            f' it will be removed from DB as well: {training_plan_name}')

                self._db.remove(doc_ids=[training_plan_doc.doc_id])
            except Exception as err:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value + ": failed to update database, "
                                               f" with error {str(err)}")
        # Update training plans
        for training_plan in training_plans_exists:
            path = os.path.join(environ['DEFAULT_TRAINING_PLANS_DIR'], training_plan)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            try:
                training_plan_info = self._db.get(self._database.name == training_plan)
            except Exception as err:
                raise FedbiomedTrainingPlanSecurityManagerError(ErrorNumbers.FB606.value +
                                                                f": failed to get training_plan info for training plan {training_plan}"
                                                                f"Details : {str(err)}")
            # Check if hashing algorithm has changed
            try:
                hash, algorithm = self._create_hash(os.path.join(environ['DEFAULT_TRAINING_PLANS_DIR'], training_plan))

                if training_plan_info['algorithm'] != environ['HASHING_ALGORITHM']:
                    # Verify no such training plan already exists in DB
                    self._check_training_plan_not_existing(None, None, hash, algorithm)

                    logger.info(
                        f'Recreating hashing for : {training_plan_info["name"]} \t {training_plan_info["training_plan_id"]}')
                    self._db.update({'hash': hash, 'algorithm': algorithm,
                                     'date_last_action': datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")},
                                    self._database.training_plan_path == path)
                # If default training plan file is modified update hashing
                elif mtime > datetime.strptime(training_plan_info['date_modified'], "%d-%m-%Y %H:%M:%S.%f"):
                    # only check when hash changes
                    # else we have error because this training plan exists in database with same hash
                    if hash != training_plan_info['hash']:
                        # Verify no such training plan already exists in DB
                        self._check_training_plan_not_existing(None, None, hash, algorithm)

                    logger.info(
                        f"Modified default training plan file has been detected. Hashing will be updated for: {training_plan}")
                    self._db.update({'hash': hash, 'algorithm': algorithm,
                                     'date_modified': mtime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                                     'date_last_action': datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")},
                                    self._database.training_plan_path == path)
            except Exception as err:
                # triggered if database update failed (see `update` method in tinydb code)
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value + ": Failed to update database, with error: "
                                               f"{str(err)}")

    def update_training_plan_hash(self, training_plan_id: str, path: str) -> True:
        """Updates an existing training plan entry in training plan database.

        Training plan entry cannot be a default training plan.

        The training plan entry to update is indicated by its `training_plan_id`
        The new training plan file for the training plan is specified from `path`.

        Args:
            training_plan_id: id of the training plan to update
            path: path where new training plan file is stored

        Returns:
            Currently always returns True.

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: try to update a default training plan
            FedbiomedTrainingPlanSecurityManagerError: cannot read or update the training plan in database
        """

        # Register training plan
        try:
            training_plan = self._db.get(self._database.training_plan_id == training_plan_id)
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + ": get request on database failed."
                                           f" Details: {str(err)}")
        if training_plan['training_plan_type'] != TrainingPlanStatus.DEFAULT.value:
            hash, algorithm = self._create_hash(path)
            # Verify no such training plan already exists in DB
            self._check_training_plan_not_existing(None, path, hash, algorithm)

            # Get modification date
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            # Get creation date
            ctime = datetime.fromtimestamp(os.path.getctime(path))

            try:
                self._db.update({'hash': hash, 'algorithm': algorithm,
                                 'date_modified': mtime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                                 'date_created': ctime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                                 'date_last_action': datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f"),
                                 'training_plan_path': path},
                                self._database.training_plan_id == training_plan_id)
            except Exception as err:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value + ": update database failed. Details :"
                                               f"{str(err)}")
        else:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + 'You cannot update default training plans. Please '
                                           'update them through their files saved in `default_training_plans` directory '
                                           'and restart your node')

        return True

    def _update_training_plan_status(self,
                                     training_plan_id: str,
                                     training_plan_status: TrainingPlanApprovalStatus,
                                     notes: Union[str, None] = None) -> True:
        """Updates training plan entry ([`training_plan_status`] field) for a given [`training_plan_id`] in the database

        Args:
            training_plan_id: id of the training_plan
            training_plan_status: new training plan status {approved, rejected, pending}
            notes: additional notes to enter into the database, explaining why training plan
                has been approved or rejected for instance. Defaults to None.

        Returns:
            True: currently always returns True

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad type for parameter
            FedbiomedTrainingPlanSecurityManagerError: database access error
        """
        if not isinstance(training_plan_id, str):
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + ": parameter training_plan_id (str) has bad "
                                           f"type {type(training_plan_id)}")
        if not isinstance(training_plan_status, TrainingPlanApprovalStatus):
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + ": parameter training_plan_status (TrainingPlanApprovalStatus) has bad "
                                           f"type {type(training_plan_status)}")
        if notes is not None and not isinstance(notes, str):
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value}: parameter note (Union[str, None]) has bad type {type(notes)}")

        try:
            training_plan = self._db.get(self._database.training_plan_id == training_plan_id)
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value}: get request on database failed. Details: {err}")

        if training_plan is None:
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value}: no training plan matches provided training_plan_id {training_plan_id}"
            )
        if training_plan.get('training_plan_status') == training_plan_status.value:
            logger.warning(f" training plan {training_plan_id} has already the following training plan status "
                           f"{training_plan_status.value}")
            return True

        else:
            training_plan_path = training_plan['training_plan_path']
            # Get modification date
            mtime = datetime.fromtimestamp(os.path.getmtime(training_plan_path))
            # Get creation date
            ctime = datetime.fromtimestamp(os.path.getctime(training_plan_path))
            try:
                self._db.update({'training_plan_status': training_plan_status.value,
                                 'date_modified': mtime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                                 'date_created': ctime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                                 'date_last_action': datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f"),
                                 'notes': notes},
                                self._database.training_plan_id == training_plan_id)
            except Exception as e:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value + f"database update operation failed, with following error: {str(e)}")
            logger.info(f"Training plan {training_plan_id} status changed to {training_plan_status.value} !")

        return True

    def approve_training_plan(self, training_plan_id: str, extra_notes: Union[str, None] = None) -> True:
        """Approves a training plan stored into the database given its [`training_plan_id`]

        Args:
            training_plan_id: id of the training plan.
            extra_notes: notes detailing why training plan has been approved. Defaults to None.

        Returns:
            Currently always returns True
        """
        res = self._update_training_plan_status(training_plan_id,
                                                TrainingPlanApprovalStatus.APPROVED,
                                                extra_notes)
        return res

    def reject_training_plan(self, training_plan_id: str, extra_notes: Union[str, None] = None) -> True:
        """Approves a training plan stored into the database given its [`training_plan_id`]

        Args:
            training_plan_id: id of the training plan.
            extra_notes: notes detailing why training plan has been rejected. Defaults to None.

        Returns:
            Currently always returns True
        """
        res = self._update_training_plan_status(training_plan_id,
                                                TrainingPlanApprovalStatus.REJECTED,
                                                extra_notes)
        return res

    def delete_training_plan(self, training_plan_id: str) -> True:
        """Removes training plan file from database.

        Only removes `registered` and `requested` type of training plans from the database.
        Does not remove the corresponding training plan file from the disk.
        Default training plans should be removed from the directory

        Args:
            training_plan_id: The id of the registered training plan.

        Returns:
            Currently always returns True.

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad type for parameter
            FedbiomedTrainingPlanSecurityManagerError: cannot read or remove training plan from the database
            FedbiomedTrainingPlanSecurityManagerError: training plan is not a `registered` training plan
                (thus a `default` training plan)
        """

        if not isinstance(training_plan_id, str):
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + ": parameter training_plan_id (str) has bad "
                                           f"type {type(training_plan_id)}")

        try:
            training_plan = self._db.get(self._database.training_plan_id == training_plan_id)
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + ": cannot get training plan from database."
                                           f"Details: {str(err)}")

        if training_plan is None:
            raise FedbiomedTrainingPlanSecurityManagerError(ErrorNumbers.FB606.value +
                                                            f": training plan {training_plan_id} not in database")

        if training_plan['training_plan_type'] != TrainingPlanStatus.DEFAULT.value:
            try:
                self._db.remove(doc_ids=[training_plan.doc_id])
            except Exception as err:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value + f": cannot remove training plan from database. Details: {str(err)}"
                )
        else:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + 'For default training plans, please remove training plan file from '
                                           '`default_training_plans` and restart your node')

        return True

    def list_training_plans(
            self,
            sort_by: Union[str, None] = None,
            select_status: Union[None, TrainingPlanApprovalStatus, List[TrainingPlanApprovalStatus]] = None,
            verbose: bool = True,
            search: Union[dict, None] = None
    ) -> List[Dict[str, Any]]:

        """Lists approved training plan files

        Args:
            sort_by: when specified, sort results by alphabetical order,
                provided sort_by is an entry in the database.
            select_status: filter list by training plan status or list of training plan statuses
            verbose: When it is True, print list of training plan in tabular format.
                Default is True.
            search: Dictionary that contains `text` property to declare the text that wil be search and `by`
                property to declare text will be search on which field

        Returns:
            A list of training plans that have
                been found as `registered`. Each training plan is in fact a dictionary
                containing fields (note that following fields are removed :'training_plan_path',
                'hash', dates due to privacy reasons).

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad type for parameter
            FedbiomedTrainingPlanSecurityManagerError: database access error
        """
        if sort_by is not None and not isinstance(sort_by, str):
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f": parameter sort_by has bad type {type(sort_by)}")
        if not isinstance(verbose, bool):
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f": parameter verbose has bad type {type(verbose)}")
            # in case select_status is a list, we filter later with elements are TrainingPlanApprovalStatus
        if select_status is not None and not isinstance(select_status, TrainingPlanApprovalStatus) and \
                not isinstance(select_status, list):
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f": parameter select_status has bad type {type(select_status)}")
        if search is not None and not isinstance(search, dict):
            raise FedbiomedTrainingPlanSecurityManagerError(f"{ErrorNumbers.FB606.value}: `search` argument should be "
                                                            f"dictionary that contains `text` and `by` (that indicates "
                                                            f"field to search on)")

        if search:
            try:
                trainingPlansSearchScheme.validate(search)
            except ValidateError as e:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    f"{ErrorNumbers.FB606.value}: `search` argument is not valid. {e}")

        if isinstance(select_status, (TrainingPlanApprovalStatus, list)):
            # filtering training plan based on their status
            if not isinstance(select_status, list):
                # convert everything into a list
                select_status = [select_status]
            select_status = [x.value for x in select_status if isinstance(x, TrainingPlanApprovalStatus)]
            # extract value from TrainingPlanApprovalStatus
            try:
                if search:
                    training_plans = self._db.search(self._database.training_plan_status.one_of(select_status) &
                                                     self._database[search["by"]].matches(search["text"],
                                                                                          flags=re.IGNORECASE))
                else:
                    training_plans = self._db.search(self._database.training_plan_status.one_of(select_status))
            except Exception as err:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    f"{ErrorNumbers.FB606.value}: request failed when looking for a training plan into database with "
                    f"error: {err}"
                )

        else:
            try:
                if search:
                    training_plans = self._db.search(
                        self._database[search["by"]].matches(search["text"], flags=re.IGNORECASE))
                else:
                    training_plans = self._db.all()
            except Exception as e:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    f"{ErrorNumbers.FB606.value} database full read operation failed, with following error: {str(e)}"
                )

        # Drop some keys for security reasons
        for doc in training_plans:
            self._remove_sensible_keys_from_request(doc)

        if sort_by is not None:
            # sorting training plan fields by column attributes
            try:
                is_entry_exists = self._db.search(self._database[sort_by].exists())
            except Exception as e:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value + f"database search operation failed, with following error: {str(e)}")
            if is_entry_exists and sort_by not in self._tags_to_remove:
                training_plans = sorted(training_plans, key=lambda x: (x[sort_by] is None, x[sort_by]))
            else:
                logger.warning(f"Field {sort_by} is not available in dataset")

        if verbose:
            print(tabulate(training_plans, headers='keys'))

        return training_plans

    def _remove_sensible_keys_from_request(self, doc: Dict[str, Any]):
        # Drop some keys for security reasons

        for tag_to_remove in self._tags_to_remove:
            try:
                doc.pop(tag_to_remove)
            except KeyError:
                logger.warning(f"missing entry in database: {tag_to_remove} for training plan {doc}")

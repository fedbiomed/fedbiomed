# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manages training plan approval for a node.
"""

from datetime import datetime
import hashlib
import os
import re
from python_minifier import minify
from tabulate import tabulate
from tinydb import TinyDB, Query
from typing import Any, Dict, List, Tuple, Union
import uuid

from fedbiomed.common.constants import (
    HashingAlgorithms,
    TrainingPlanApprovalStatus,
    TrainingPlanStatus,
    ErrorNumbers,
)
from fedbiomed.common.db import DBTable
from fedbiomed.common.exceptions import FedbiomedTrainingPlanSecurityManagerError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import (
    ApprovalRequest,
    ApprovalReply,
    TrainingPlanStatusRequest,
    TrainingPlanStatusReply,
)
from fedbiomed.common.utils import SHARE_DIR

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


class TrainingPlanSecurityManager:
    """Manages training plan approval for a node."""

    def __init__(
        self,
        db: str,
        node_id: str,
        hashing: str,
        tp_approval: bool = False
    ) -> None:
        """Class constructor for TrainingPlanSecurityManager.

        Creates a DB object for the table named as `Training plans` and builds a query object to query
        the database.

        Args:
            db: Path to database file.
            node_id: ID of the active node.
            hashing: Hashing algorithm
            tp_approval: True if training plan approval is requested
        """

        self._node_id = node_id
        self._tp_approval = tp_approval
        self._default_tps = os.path.join(SHARE_DIR, 'envs', 'common', 'default_training_plans')
        self._tinydb = TinyDB(db)
        self._tinydb.table_class = DBTable
        # dont use DB read cache for coherence when updating from multiple sources (eg: GUI and CLI)
        self._db = self._tinydb.table(name="TrainingPlans", cache_size=0)
        self._database = Query()

        self._hashing = hashing
        self._tags_to_remove = ["hash", "date_modified", "date_created"]

    def _create_hash(self, source: str, from_string: str = False):
        """Creates hash with given training plan

        Args:
            source: Training plan source code, or path to training plan file
            from_string: if True read training plan from file, if False receive it as a string
        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad argument type
            FedbiomedTrainingPlanSecurityManagerError: file cannot be open
            FedbiomedTrainingPlanSecurityManagerError: file cannot be minified
            FedbiomedTrainingPlanSecurityManagerError: Hashing algorithm does not exist in HASH_FUNCTION table
        """

        hash_algo = self._hashing

        if not isinstance(source, str):
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value
                + f": {source} is not a path or string containing codes"
            )

        if not from_string:
            try:
                with open(source, "r") as training_plan:
                    source = training_plan.read()
            except FileNotFoundError as e:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value
                    + f": training plan file {source} not found on system"
                ) from e
            except PermissionError as e:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value
                    + f": cannot open training plan file {source} due"
                    + " to unsatisfactory privilege"
                ) from e
            except OSError as e:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value
                    + f": cannot open training plan file {source} "
                    + "(file might have been corrupted)"
                ) from e

        # Minify training plan file using python_minifier module
        try:
            mini_content = minify(
                source,
                remove_annotations=False,
                combine_imports=False,
                remove_pass=False,
                hoist_literals=False,
                remove_object_base=True,
                rename_locals=False,
            )
        except Exception as err:
            # minify doesn't provide any specific exception
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + f": cannot minify source code "
                f"details: {err}"
            ) from err
        # Hash training plan content based on active hashing algorithm
        if hash_algo in HashingAlgorithms.list():
            hashing = HASH_FUNCTIONS[hash_algo]()
        else:
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value}: unknown hashing algorithm in the 'config'"
                f"{self._hashing}"
            )

        # Create hash from training plan minified training plan content and encoded as `utf-8`
        hashing.update(mini_content.encode("utf-8"))

        return hashing.hexdigest(), hash_algo, source

    def _check_training_plan_not_existing(
        self,
        name: Union[str, None] = None,
        hash_: Union[str, None] = None,
        algorithm: Union[str, None] = None,
    ) -> None:
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
            training_plans_name_get = self._db.get(self._database.name == name)

            if training_plans_name_get:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    f"{ErrorNumbers.FB606.value}:  there is already a existing training plan with "
                    "same name: '{name}' Please use different name"
                )

        if hash is not None or algorithm is not None:
            if algorithm is None:
                training_plans_hash_get = self._db.get(self._database.hash == hash_)
            elif hash_ is None:
                training_plans_hash_get = self._db.get(
                    self._database.algorithm == algorithm
                )
            else:
                training_plans_hash_get = self._db.get(
                    (self._database.hash == hash_)
                    & (self._database.algorithm == algorithm)
                )

            if training_plans_hash_get:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    f"{ErrorNumbers.FB606.value}:  there is already an existing training plan in database same code "
                    f' hash, training plan name is "{training_plans_hash_get["name"]}"'
                )

    def register_training_plan(
        self,
        name: str,
        description: str,
        path: str,
        training_plan_type: str = TrainingPlanStatus.REGISTERED.value,
        training_plan_id: str = None,
        researcher_id: str = None,
    ) -> str:
        """Approves/registers training plan file through CLI.

        Args:
            name: Training plan file name. The name should be unique. Otherwise, methods
                throws an Exception FedbiomedTrainingPlanSecurityManagerError
            description: Description for training plan file.
            path: Exact path for the training plan that will be registered
            training_plan_type: Default is `registered`. It means that training plan has been registered
                by a user/hospital. Other value can be `default` which indicates
                that training plan is default (training plans for tutorials/examples)
            training_plan_id: Pre-defined id for training plan. Default is None. When it is None method
                creates unique id for the training plan.
            researcher_id: ID of the researcher who is owner/requester of the training plan file

        Returns:
            The ID of registered training plan

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: `training_plan_type` is not `registered` or `default`
            FedbiomedTrainingPlanSecurityManagerError: training plan is already registered into database
            FedbiomedTrainingPlanSecurityManagerError: training plan name is already used for saving another training plan
            FedbiomedTrainingPlanSecurityManagerError: database access problem
        """

        # Check training plan type is valid
        if training_plan_type not in TrainingPlanStatus.list():
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"Unknown training plan (training_plan_type) type: {training_plan_type}"
            )

        if not training_plan_id:
            training_plan_id = "training_plan_" + str(uuid.uuid4())
        training_plan_hash, algorithm, source = self._create_hash(path)

        # Verify no such training plan is already registered
        self._check_training_plan_not_existing(name, training_plan_hash, None)

        # Training plan file creation date
        ctime = datetime.fromtimestamp(os.path.getctime(path)).strftime(
            "%d-%m-%Y %H:%M:%S.%f"
        )
        # Training plan file modification date
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime(
            "%d-%m-%Y %H:%M:%S.%f"
        )
        # Training plan file registration date
        rtime = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")

        training_plan_record = dict(
            name=name,
            description=description,
            hash=training_plan_hash,
            training_plan=source,
            training_plan_id=training_plan_id,
            training_plan_type=training_plan_type,
            training_plan_status=TrainingPlanApprovalStatus.APPROVED.value,
            algorithm=algorithm,
            researcher_id=researcher_id,
            date_created=ctime,
            date_modified=mtime,
            date_registered=rtime,
            date_last_action=rtime,
        )

        try:
            self._db.insert(training_plan_record)
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + " : database insertion failed with"
                f" following error: {str(err)}"
            )
        return training_plan_id

    def check_hashes_for_registered_training_plans(self):
        """Checks registered training plans (training plans either rejected or approved).

        Makes sure training plan files exists and hashing algorithm is matched with specified
        algorithm in the config file.

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: cannot update training plan list in database
        """

        try:
            training_plans, docs = self._db.search(
                self._database.training_plan_type.all(
                    TrainingPlanStatus.REGISTERED.value
                ),
                add_docs=True,
            )
        except Exception as e:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value
                + f"database search operation failed, with following error: {str(e)}"
            )
        logger.info("Checking hashes for registered training plans")
        if not training_plans:
            logger.info("There are no training plans registered")
        else:
            for training_plan, doc in zip(training_plans, docs):
                # If training plan file is exists
                if training_plan["algorithm"] != self._hashing:
                    logger.info(
                        f'Recreating hashing for : {training_plan["name"]} \t {training_plan["training_plan_id"]}'
                    )
                    hashing, algorithm, _ = self._create_hash(
                        training_plan["training_plan"], from_string=True
                    )

                    # Verify no such training plan already exists in DB
                    self._check_training_plan_not_existing(None, hashing, None)

                    rtime = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")
                    try:
                        self._db.update(
                            {
                                "hash": hashing,
                                "algorithm": algorithm,
                                "date_last_action": rtime,
                            },
                            self._database.training_plan_id.all(
                                training_plan["training_plan_id"]
                            ),
                        )
                    except Exception as err:
                        raise FedbiomedTrainingPlanSecurityManagerError(
                            ErrorNumbers.FB606.value
                            + ": database update failed, with error "
                            f" {str(err)}"
                        )

    def check_training_plan_status(
        self,
        training_plan_source: str,
        state: Union[TrainingPlanApprovalStatus, TrainingPlanStatus, None],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Checks whether training plan exists in database and has the specified status.

        Sends a query to database to search for hash of requested training plan.
        If the hash matches with one of the
        training plans hashes in the DB, and if training plan has the specified status {approved, rejected, pending}
        or training_plan_type {registered, requested, default}.

        Args:
            training_plan_source: The source code of requested training plan
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
            FedbiomedTrainingPlanSecurityManagerError: bad argument type or value
            FedbiomedTrainingPlanSecurityManagerError: database access problem
        """

        # Create hash for requested training plan
        req_training_plan_hash, *_ = self._create_hash(
            training_plan_source, from_string=True
        )

        # If node allows defaults training plans search hash for all training plan types
        # otherwise search only for `registered` training plans

        if state is None:
            _all_training_plans_with_status = None
        elif isinstance(state, TrainingPlanApprovalStatus):
            _all_training_plans_with_status = (
                self._database.training_plan_status == state.value
            )
        elif isinstance(state, TrainingPlanStatus):
            _all_training_plans_with_status = (
                self._database.training_plan_type == state.value
            )
        else:
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value} + status should be either TrainingPlanApprovalStatus or "
                f"TrainingPlanStatus, but got {type(state)}"
            )
        _all_training_plans_which_have_req_hash = (
            self._database.hash == req_training_plan_hash
        )

        if _all_training_plans_with_status is None:
            # check only against hash
            training_plan = self._db.get(_all_training_plans_which_have_req_hash)
        else:
            # check against hash and status
            training_plan = self._db.get(
                _all_training_plans_with_status
                & _all_training_plans_which_have_req_hash
            )

        status = True if training_plan else False

        return status, training_plan

    def get_training_plan_by_name(
        self, training_plan_name: str
    ) -> Union[Dict[str, Any], None]:
        """Gets training plan from database, by its name

        Args:
            training_plan_name: name of the training plan entry to search in the database

        Returns:
            training plan entry found in the database matching `training_plan_name`. Otherwise, returns None.

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad argument type
            FedbiomedTrainingPlanSecurityManagerError: cannot read database.
        """

        training_plan = self._db.get(self._database.name == training_plan_name)

        return training_plan

    def get_training_plan_from_database(
        self, training_plan: str
    ) -> Union[Dict[str, Any], None]:
        """Gets training plan from database, by its hash

        !!! info "Training plan file MUST be a *.txt file."

        Args:
            training_plan: training plan source code, in order to compute its hash.

        Returns:
            Training plan entry found in the dataset if query in database succeed. Otherwise, returns
            None.

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad argument type
            FedbiomedTrainingPlanSecurityManagerError: database access problem
        """

        req_training_plan_hash, *_ = self._create_hash(training_plan, from_string=True)
        _all_training_plans_which_have_req_hash = (
            self._database.hash == req_training_plan_hash
        )

        return self._db.get(_all_training_plans_which_have_req_hash)

    def get_training_plan_by_id(
        self, training_plan_id: str, secure: bool = True
    ) -> Union[Dict[str, Any], None]:
        """Get a training plan in database given his `training_plan_id`.

        Also add a `content` key to the returned dictionary. This method is not used within the
        library source code but it is used for Fed-BioMed GUI.

        Args:
            training_plan_id: id of the training plan to pick from the database
            secure: if `True` then strip some security sensitive fields
            content: if `True` add content of training plan in `content` key of returned training plan. If `False` then
                `content` key value is `None`


        Returns:
            training plan entry from database through a query based on the training plan_id.
            If there is no training plan matching [`training_plan_id`], returns None

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad argument type
            FedbiomedTrainingPlanSecurityManagerError: database access problem
        """

        training_plan = self._db.get(
            self._database.training_plan_id == training_plan_id
        )

        if training_plan and secure:
            self._remove_sensible_keys_from_request(training_plan)

        return training_plan

    def reply_training_plan_approval_request(self, request: ApprovalRequest):
        """Submits a training plan file (TrainingPlan) for approval. Needs an action from Node

        Args:
            request: approval request message, received from Researcher
        """

        reply = {
            "researcher_id": request.researcher_id,
            "request_id": request.request_id,
            "node_id": self._node_id,
            "message": "",
            "status": 0,  # HTTP status (set by default to 0, non-existing HTTP status code)
        }

        is_existant = False
        training_plan_name = "training_plan_" + str(uuid.uuid4())
        training_plan = request.training_plan
        reply.update({"training_plan_id": training_plan_name})

        try:
            # check if training plan has already been registered into database
            is_existant, _ = self.check_training_plan_status(training_plan, None)

        except FedbiomedTrainingPlanSecurityManagerError as exp:
            logger.error(f"Error while training plan approval request {exp}")
            reply.update(
                {
                    "message": "Can not check whether training plan has already be registered or not due to error",
                    "success": False,
                }
            )

            return ApprovalReply(**reply)

        if not is_existant:
            # move training plan into corresponding directory
            try:
                training_plan_hash, hash_algo, _ = self._create_hash(
                    training_plan, from_string=True
                )
                ctime = datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f")
                training_plan_object = dict(
                    name=training_plan_name,
                    description=request.description,
                    hash=training_plan_hash,
                    training_plan=request.training_plan,
                    training_plan_id=training_plan_name,
                    training_plan_type=TrainingPlanStatus.REQUESTED.value,
                    training_plan_status=TrainingPlanApprovalStatus.PENDING.value,
                    algorithm=hash_algo,
                    date_created=ctime,
                    date_modified=ctime,
                    date_registered=ctime,
                    date_last_action=None,
                    researcher_id=request.researcher_id,
                    notes=None,
                )

                self._db.upsert(
                    training_plan_object, self._database.hash == training_plan_hash
                )
            except Exception as err:

                logger.error(
                    f"Cannot add training plan in database due to error : {err}"
                )
                reply.update(
                    {
                        "message": "Cannot add training plan into database due to error",
                        "success": False,
                    }
                )
                return ApprovalReply(**reply)

            else:
                reply["success"] = True
                logger.debug("Training plan successfully received by Node for approval")

        else:
            if self.check_training_plan_status(
                training_plan, TrainingPlanApprovalStatus.PENDING
            )[0]:
                reply.update(
                    {
                        "message": "Training plan already sent for Approval (status Pending). "
                        "Please wait for Node approval."
                    }
                )
            elif self.check_training_plan_status(
                training_plan, TrainingPlanApprovalStatus.APPROVED
            )[0]:
                reply.update(
                    {
                        "message": f"Training plan '{request.description}' is already Approved. Ready "
                        "to train on this training plan."
                    }
                )
            else:
                reply.update(
                    {"message": "Training plan already exists in database. Aborting"}
                )

            reply.update({"success": True})

        # Send training plan approval acknowledge answer to researcher
        return ApprovalReply(**reply)

    def reply_training_plan_status_request(self, request: TrainingPlanStatusRequest):
        """Returns requested training plan file status {approved, rejected, pending}
        and sends TrainingPlanStatusReply to researcher.

        Called directly from Node.py when it receives TrainingPlanStatusRequest.

        Args:
            request: Message that is received from researcher.
                Formatted as TrainingPlanStatusRequest
        """

        # Main header for the training plan status request
        reply = {
            "researcher_id": request.researcher_id,
            "request_id": request.request_id,
            "node_id": self._node_id,
            "experiment_id": request.experiment_id,
            "approval_obligation": True,
            "training_plan": request.training_plan,
            "training_plan_id": None,
        }

        try:
            training_plan = self.get_training_plan_from_database(request.training_plan)
            if training_plan is not None:
                training_plan_status = training_plan.get(
                    "training_plan_status", "Not Registered"
                )
                reply.update(
                    {"training_plan_id": training_plan.get("training_plan_id", None)}
                )
            else:
                training_plan_status = "Not Registered"

            reply.update({"success": True, "status": training_plan_status})
            if self._tp_approval:
                if training_plan_status == TrainingPlanApprovalStatus.APPROVED.value:
                    msg = "Training plan has been approved by the node, training can start"
                elif training_plan_status == TrainingPlanApprovalStatus.PENDING.value:
                    msg = "Training plan is pending: waiting for a review"
                elif training_plan_status == TrainingPlanApprovalStatus.REJECTED.value:
                    msg = "Training plan has been rejected by the node, training is not possible"
                else:
                    msg = f"Unknown training plan not in database (status {training_plan_status})"
                reply.update({"msg": msg})

            else:
                reply.update(
                    {
                        "approval_obligation": False,
                        "msg": "This node does not require training plan approval (maybe for debugging purposes).",
                    }
                )

        # Catch all exception to be able send reply back to researcher
        except Exception as exp:
            logger.error(exp)
            reply.update(
                {
                    "success": False,
                    "status": "Error",
                    "msg": f"{ErrorNumbers.FB606.value}: Cannot check if training plan has been registered due "
                    "to an internal error",
                }
            )

        return TrainingPlanStatusReply(**reply)

    def register_update_default_training_plans(self):
        """Registers or updates default training plans.

        Launched when the node is started through CLI, if `allow_default_training_plans` is enabled.
        Checks the files saved into `default_training_plans` directory and update/register them based on following
        conditions:

        - Registers if there is a new training plan file which isn't saved into db.
        - Updates if training plan is modified or if hashing algorithm has changed in config file.

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: cannot read or update training plan database
        """

        # Get training plan files saved in the directory
        training_plans_file = os.listdir(self._default_tps)

        # Get only default training plans from DB
        try:
            training_plans = self._db.search(
                self._database.training_plan_type == "default"
            )
        except Exception as e:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value
                + f"database search operation failed, with following error: {str(e)}"
            )

        # Get training plan names from list of training plans
        training_plans_dict = {
            training_plan.get("name"): training_plan for training_plan in training_plans
        }
        training_plans_name_db = list(training_plans_dict.keys())
        # Default training plans not in database
        training_plans_not_saved = list(
            set(training_plans_file) - set(training_plans_name_db)
        )
        # Default training plans that have been deleted from file system but not in DB
        training_plans_deleted = list(
            set(training_plans_name_db) - set(training_plans_file)
        )
        # Training plans have already saved and exist in the database
        training_plans_exists = list(
            set(training_plans_file) - set(training_plans_not_saved)
        )

        # Register new default training plans
        for training_plan in training_plans_not_saved:
            self.register_training_plan(
                name=training_plan,
                description="Default training plan",
                path=os.path.join(self._default_tps, training_plan),
                training_plan_type="default",
            )

        # Remove training plans that have been removed from file system
        for training_plan_name in training_plans_deleted:
            try:
                _, training_plan_doc = self._db.get(
                    self._database.name == training_plan_name, add_docs=True
                )
                logger.info(
                    "Removed default training plan file has been detected,"
                    f" it will be removed from DB as well: {training_plan_name}"
                )

                self._db.remove(doc_ids=[training_plan_doc.doc_id])
            except Exception as err:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value + ": failed to update database, "
                    f" with error {str(err)}"
                )
        # Update training plans
        for training_plan in training_plans_exists:
            path = os.path.join(self._default_tps, training_plan)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            try:
                training_plan_info = self._db.get(self._database.name == training_plan)
            except Exception as err:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value
                    + f": failed to get training_plan info for training plan {training_plan}"
                    f"Details : {str(err)}"
                )

            # Check if hashing algorithm has changed
            try:
                hash, algorithm, _ = self._create_hash(
                    os.path.join(self._default_tps, training_plan)
                )

                if training_plan_info["algorithm"] != self._hashing:
                    # Verify no such training plan already exists in DB
                    self._check_training_plan_not_existing(None, hash, algorithm)
                    logger.info(
                        f'Recreating hashing for : {training_plan_info["name"]} \t'
                        '{training_plan_info["training_plan_id"]}'
                    )

                    self._db.update(
                        {
                            "hash": hash,
                            "algorithm": algorithm,
                            "date_last_action": datetime.now().strftime(
                                "%d-%m-%Y %H:%M:%S.%f"
                            ),
                        },
                        self._database.training_plan_id
                        == training_plan_info["training_plan_id"],
                    )
                # If default training plan file is modified update hashing
                elif mtime > datetime.strptime(
                    training_plan_info["date_modified"], "%d-%m-%Y %H:%M:%S.%f"
                ):
                    # only check when hash changes
                    # else we have error because this training plan exists in database with same hash
                    if hash != training_plan_info["hash"]:
                        # Verify no such training plan already exists in DB
                        self._check_training_plan_not_existing(None, hash, algorithm)

                    logger.info(
                        "Modified default training plan file has been detected. "
                        f"Hashing will be updated for: {training_plan}"
                    )

                    self._db.update(
                        {
                            "hash": hash,
                            "algorithm": algorithm,
                            "date_modified": mtime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                            "date_last_action": datetime.now().strftime(
                                "%d-%m-%Y %H:%M:%S.%f"
                            ),
                        },
                        self._database.training_plan_id
                        == training_plan_info["training_plan_id"],
                    )
            except Exception as err:
                # triggered if database update failed (see `update` method in tinydb code)
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value
                    + ": Failed to update database, with error: "
                    f"{str(err)}"
                )

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
            training_plan = self._db.get(
                self._database.training_plan_id == training_plan_id
            )
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + ": get request on database failed."
                f" Details: {str(err)}"
            )
        if training_plan["training_plan_type"] != TrainingPlanStatus.DEFAULT.value:
            hash, algorithm, source = self._create_hash(path)
            # Verify no such training plan already exists in DB
            self._check_training_plan_not_existing(None, hash, algorithm)

            # Get modification date
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            # Get creation date
            ctime = datetime.fromtimestamp(os.path.getctime(path))

            try:
                self._db.update(
                    {
                        "hash": hash,
                        "algorithm": algorithm,
                        "date_modified": mtime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                        "date_created": ctime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                        "date_last_action": datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S.%f"
                        ),
                        "training_plan": source,
                    },
                    self._database.training_plan_id == training_plan_id,
                )
            except Exception as err:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value + ": update database failed. Details :"
                    f"{str(err)}"
                )
        else:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value
                + "You cannot update default training plans. Please "
                "update them through their files saved in `default_training_plans` directory "
                "and restart your node"
            )

        return True

    def _update_training_plan_status(
        self,
        training_plan_id: str,
        training_plan_status: TrainingPlanApprovalStatus,
        notes: Union[str, None] = None,
    ) -> True:
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

        training_plan = self._db.get(
            self._database.training_plan_id == training_plan_id
        )

        if training_plan is None:
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value}: no training plan matches provided training_plan_id {training_plan_id}"
            )

        if training_plan.get("training_plan_status") == training_plan_status.value:
            logger.warning(
                f" training plan {training_plan_id} has already the following training plan status "
                f"{training_plan_status.value}"
            )
            return True

        else:
            self._db.update(
                {
                    "training_plan_status": training_plan_status.value,
                    "date_last_action": datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f"),
                    "notes": notes,
                },
                self._database.training_plan_id == training_plan_id,
            )
            logger.info(
                f"Training plan {training_plan_id} status changed to {training_plan_status.value} !"
            )

        return True

    def approve_training_plan(
        self, training_plan_id: str, extra_notes: Union[str, None] = None
    ) -> True:
        """Approves a training plan stored into the database given its [`training_plan_id`]

        Args:
            training_plan_id: id of the training plan.
            extra_notes: notes detailing why training plan has been approved. Defaults to None.

        Returns:
            Currently always returns True
        """
        res = self._update_training_plan_status(
            training_plan_id, TrainingPlanApprovalStatus.APPROVED, extra_notes
        )
        return res

    def reject_training_plan(
        self, training_plan_id: str, extra_notes: Union[str, None] = None
    ) -> True:
        """Approves a training plan stored into the database given its [`training_plan_id`]

        Args:
            training_plan_id: id of the training plan.
            extra_notes: notes detailing why training plan has been rejected. Defaults to None.

        Returns:
            Currently always returns True
        """
        res = self._update_training_plan_status(
            training_plan_id, TrainingPlanApprovalStatus.REJECTED, extra_notes
        )
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

        try:
            training_plan, doc = self._db.get(
                self._database.training_plan_id == training_plan_id, add_docs=True
            )
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + ": cannot get training plan from database."
                f"Details: {str(err)}"
            )

        if training_plan is None:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value
                + f": training plan {training_plan_id} not in database"
            )

        if training_plan["training_plan_type"] != TrainingPlanStatus.DEFAULT.value:
            try:
                self._db.remove(doc_ids=[doc.doc_id])
            except Exception as err:
                raise FedbiomedTrainingPlanSecurityManagerError(
                    ErrorNumbers.FB606.value
                    + f": cannot remove training plan from database. Details: {str(err)}"
                )
        else:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value
                + "For default training plans, please remove training plan file from "
                "`default_training_plans` and restart your node"
            )

        return True

    def list_training_plans(
        self,
        sort_by: Union[str, None] = None,
        select_status: Union[None, List[TrainingPlanApprovalStatus]] = None,
        verbose: bool = True,
        search: Union[dict, None] = None,
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
                containing fields (note that following fields are removed :'training_plan',
                'hash', dates due to privacy reasons).

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad type for parameter
            FedbiomedTrainingPlanSecurityManagerError: database access error
        """

        # Selects all
        query = self._database.training_plan_id.exists()

        if select_status:
            # filtering training plan based on their status
            if not isinstance(select_status, list):
                # convert everything into a list
                select_status = [select_status]

            select_status = [
                x.value
                for x in select_status
                if isinstance(x, TrainingPlanApprovalStatus)
            ]
            query = self._database.training_plan_status.one_of(select_status)
            # extract value from TrainingPlanApprovalStatus

        if search:
            query = query & self._database[search["by"]].matches(
                search["text"], flags=re.IGNORECASE
            )

        try:
            training_plans = self._db.search(query)
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value}: request failed when looking for a training plan into database with "
                f"error: {err}"
            )

        # Drop some keys for security reasons
        for doc in training_plans:
            self._remove_sensible_keys_from_request(doc)

        if sort_by is not None:
            # sorting training plan fields by column attributes
            is_entry_exists = self._db.search(self._database[sort_by].exists())
            if is_entry_exists and sort_by not in self._tags_to_remove:
                training_plans = sorted(
                    training_plans, key=lambda x: (x[sort_by] is None, x[sort_by])
                )
            else:
                logger.warning(f"Field {sort_by} is not available in dataset")

        if verbose:
            training_plans_verbose = training_plans.copy()
            for tp in training_plans_verbose:
                tp.pop("training_plan")

            print(tabulate(training_plans_verbose, headers="keys"))

        return training_plans

    def _remove_sensible_keys_from_request(self, doc: Dict[str, Any]):
        # Drop some keys for security reasons

        for tag_to_remove in self._tags_to_remove:
            if tag_to_remove in doc:
                doc.pop(tag_to_remove)

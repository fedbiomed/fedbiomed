# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Manages training plan approval for a node."""

import hashlib
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from python_minifier import minify
from tabulate import tabulate
from tinydb import Query, TinyDB

from fedbiomed.common.constants import (
    ErrorNumbers,
    HashingAlgorithms,
    TrainingPlanApprovalStatus,
    TrainingPlanStatus,
)
from fedbiomed.common.db import DBTable
from fedbiomed.common.exceptions import FedbiomedTrainingPlanSecurityManagerError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import (
    ApprovalReply,
    ApprovalRequest,
    TrainingPlanStatusReply,
    TrainingPlanStatusRequest,
)

# These factories are used only to verify and migrate hashes written by older nodes.
# Historical BLAKE2B records were produced with blake2s, despite their label.
_LEGACY_HASH_FUNCTIONS = {
    "SHA256": hashlib.sha256,
    "SHA384": hashlib.sha384,
    "SHA512": hashlib.sha512,
    "SHA3_256": hashlib.sha3_256,
    "SHA3_384": hashlib.sha3_384,
    "SHA3_512": hashlib.sha3_512,
    "BLAKE2B": hashlib.blake2s,
    "BLAKE2S": hashlib.blake2s,
}


class TrainingPlanSecurityManager:
    """Manages training plan approval for a node."""

    def __init__(
        self,
        db: str,
        node_id: str,
        node_name: str,
        hashing: str = HashingAlgorithms.SHA256.value,
        tp_approval: bool = False,
    ) -> None:
        """Class constructor for TrainingPlanSecurityManager.

        Creates a DB object for the table named as `Training plans` and builds a query object to query
        the database.

        Args:
            db: Path to database file.
            node_id: ID of the active node.
            hashing: Compatibility argument. Only SHA256 is supported.
            tp_approval: True if training plan approval is requested
        """

        self._node_id = node_id
        self._node_name = node_name

        self._tp_approval = tp_approval
        if hashing != HashingAlgorithms.SHA256.value:
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value}: training plans use SHA256 hashing"
            )
        self._tinydb = TinyDB(db)
        self._tinydb.table_class = DBTable
        # dont use DB read cache for coherence when updating from multiple sources (eg: GUI and CLI)
        self._db = self._tinydb.table(name="TrainingPlans", cache_size=0)
        self._database = Query()

        self._hashing = HashingAlgorithms.SHA256.value
        self._tags_to_remove = ["hash", "date_modified", "date_created"]
        try:
            self._migrate_training_plans()
        except Exception:
            self._tinydb.close()
            raise

    def _migrate_training_plans(self) -> None:
        """Upgrade stored plans without extending any existing approval.

        Former automatic defaults become researcher requests requiring review.
        Explicit administrator decisions survive a hash upgrade only when the
        stored source still matches the previously approved digest. Validate
        every proposed change before writing the table once.
        """
        try:
            records = self._db.all()
            changes = []
            digests = {}
            identifiers = set()
            defaults = 0
            for record in records:
                identifier = record["training_plan_id"]
                if not isinstance(identifier, str) or identifier in identifiers:
                    raise ValueError("missing or duplicate training plan identifier")
                identifiers.add(identifier)
                algorithm = record["algorithm"]
                digest = record["hash"]
                fields = {}
                was_default = record.get("training_plan_type") == "default"
                if algorithm != self._hashing or was_default:
                    source = record["training_plan"]
                    new_digest, _, _ = self._create_hash(source, from_string=True)
                    factory = _LEGACY_HASH_FUNCTIONS.get(algorithm)
                    if factory is None:
                        raise ValueError("unknown legacy hashing algorithm")
                    canonical = minify(
                        source,
                        remove_annotations=False,
                        combine_imports=False,
                        remove_pass=False,
                        hoist_literals=False,
                        remove_object_base=True,
                        rename_locals=False,
                    )
                    if factory(canonical.encode("utf-8")).hexdigest() != digest:
                        raise ValueError(
                            "stored source does not match its legacy digest"
                        )
                    digest = new_digest
                    if algorithm != self._hashing:
                        fields.update(hash=digest, algorithm=self._hashing)
                if not isinstance(digest, str) or not digest:
                    raise ValueError("missing training plan digest")
                if digest in digests:
                    raise ValueError(
                        "duplicate training plan code requires administrator reconciliation"
                    )
                digests[digest] = identifier
                if was_default:
                    fields["training_plan_type"] = TrainingPlanStatus.REQUESTED.value
                    if (
                        record.get("training_plan_status")
                        != TrainingPlanApprovalStatus.REJECTED.value
                    ):
                        fields["training_plan_status"] = (
                            TrainingPlanApprovalStatus.PENDING.value
                        )
                    defaults += 1
                if fields:
                    changes.append(
                        (fields, self._database.training_plan_id == identifier)
                    )
            if changes:
                self._db.update_multiple(changes)
            if defaults:
                logger.warning(
                    "%d formerly automatic training plans now require explicit administrator review; "
                    "existing rejections are retained.",
                    defaults,
                )
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                f"{ErrorNumbers.FB606.value}: cannot migrate training plan approvals: {err}. "
                "Back up the node database and reconcile the stored plans before restarting."
            ) from err

    def _create_hash(self, source: str, from_string: bool = False):
        """Creates hash with given training plan

        Args:
            source: Training plan source code, or path to training plan file
            from_string: if True read training plan from file, if False receive it as a string
        Raises:
            FedbiomedTrainingPlanSecurityManagerError: bad argument type
            FedbiomedTrainingPlanSecurityManagerError: file cannot be open
            FedbiomedTrainingPlanSecurityManagerError: file cannot be minified
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
                ErrorNumbers.FB606.value + f": cannot minify source code details: {err}"
            ) from err
        hashing = hashlib.sha256()

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
                    f"same name: '{name}' Please use different name"
                )

        if hash_ is not None or algorithm is not None:
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
        training_plan_id: Optional[str] = None,
        researcher_id: Optional[str] = None,
    ) -> str:
        """Approves/registers training plan file through CLI.

        Args:
            name: Training plan file name. The name should be unique. Otherwise, methods
                throws an Exception FedbiomedTrainingPlanSecurityManagerError
            description: Description for training plan file.
            path: Exact path for the training plan that will be registered
            training_plan_type: `registered` for an administrator's local registration,
                or `requested` for a researcher submission.
            training_plan_id: Pre-defined id for training plan. Default is None. When it is None method
                creates unique id for the training plan.
            researcher_id: ID of the researcher who is owner/requester of the training plan file

        Returns:
            The ID of registered training plan

        Raises:
            FedbiomedTrainingPlanSecurityManagerError: `training_plan_type` is not `registered` or `requested`
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
            ) from err
        return training_plan_id

    def check_training_plan_status(
        self,
        training_plan_source: str,
        state: Union[TrainingPlanApprovalStatus, TrainingPlanStatus, None],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Checks whether training plan exists in database and has the specified status.

        Sends a query to database to search for hash of requested training plan.
        If the hash matches with one of the
        training plans hashes in the DB, and if training plan has the specified status {approved, rejected, pending}
        or training_plan_type {registered, requested}.

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

        # Match code hashes across registered and requested training plans.
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

        reply: Dict[str, Any] = {
            "researcher_id": request.researcher_id,
            "request_id": request.request_id,
            "node_id": self._node_id,
            "node_name": self._node_name,
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
        reply: Dict[str, Any] = {
            "researcher_id": request.researcher_id,
            "request_id": request.request_id,
            "node_id": self._node_id,
            "node_name": self._node_name,
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

    def update_training_plan_hash(self, training_plan_id: str, path: str) -> bool:
        """Updates an existing training plan entry in training plan database.

        The training plan entry to update is indicated by its `training_plan_id`
        The new training plan file for the training plan is specified from `path`.

        Args:
            training_plan_id: id of the training plan to update
            path: path where new training plan file is stored

        Returns:
            Currently always returns True.

        Raises:
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
            ) from err
        if training_plan is None:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value
                + f": training plan with id {training_plan_id} not found in database"
            )
        hash_, algorithm, source = self._create_hash(path)
        # Verify no such training plan already exists in DB
        self._check_training_plan_not_existing(None, hash_, algorithm)

        # Get modification date
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        # Get creation date
        ctime = datetime.fromtimestamp(os.path.getctime(path))

        try:
            self._db.update(
                {
                    "hash": hash_,
                    "algorithm": algorithm,
                    "date_modified": mtime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                    "date_created": ctime.strftime("%d-%m-%Y %H:%M:%S.%f"),
                    "date_last_action": datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f"),
                    "training_plan": source,
                },
                self._database.training_plan_id == training_plan_id,
            )
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + ": update database failed. Details :"
                f"{str(err)}"
            ) from err

        return True

    def _update_training_plan_status(
        self,
        training_plan_id: str,
        training_plan_status: TrainingPlanApprovalStatus,
        notes: Union[str, None] = None,
    ) -> bool:
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
    ) -> bool:
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
    ) -> bool:
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

    def delete_training_plan(self, training_plan_id: str) -> bool:
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
        """

        try:
            training_plan, doc = self._db.get(  # type: ignore[call-arg, misc]
                self._database.training_plan_id == training_plan_id, add_docs=True
            )
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value + ": cannot get training plan from database."
                f"Details: {str(err)}"
            ) from err

        if training_plan is None:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value
                + f": training plan {training_plan_id} not in database"
            )

        try:
            self._db.remove(doc_ids=[doc.doc_id])
        except Exception as err:
            raise FedbiomedTrainingPlanSecurityManagerError(
                ErrorNumbers.FB606.value
                + f": cannot remove training plan from database. Details: {str(err)}"
            ) from err

        return True

    def list_training_plans(
        self,
        sort_by: Union[str, None] = None,
        select_status: Union[
            None, TrainingPlanApprovalStatus, List[TrainingPlanApprovalStatus]
        ] = None,
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
            search: Dictionary that contains `text` property to declare the text that will be search and `by`
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

            _status_values = [
                x.value
                for x in select_status
                if isinstance(x, TrainingPlanApprovalStatus)
            ]
            query = self._database.training_plan_status.one_of(_status_values)
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
            ) from err

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

        # No-op at runtime: widens List[Document] to List[Dict] for mypy (lists are invariant).
        return cast(List[Dict[str, Any]], training_plans)

    def _remove_sensible_keys_from_request(self, doc: Dict[str, Any]):
        # Drop some keys for security reasons

        for tag_to_remove in self._tags_to_remove:
            if tag_to_remove in doc:
                doc.pop(tag_to_remove)

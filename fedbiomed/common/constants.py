# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Fed-BioMed constants/enums"""

from enum import Enum

"""Directory/folder name where DB files are saved"""
DB_FOLDER_NAME = "var"

"""Prefix for database files name"""
DB_PREFIX = 'db_'

"""Prefix for node ID"""
NODE_PREFIX = 'node_'


MPSPDZ_certificate_prefix = "MPSPDZ_certificate"


class _BaseEnum(Enum):
    """
    Parent class to pass default methods to enumeration classes
    """

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class ComponentType(_BaseEnum):
    """Enumeration class, used to characterize the type of component of the fedbiomed architecture

    Attributes:
        RESEARCHER: Researcher component
        NODE: Node component
    """

    RESEARCHER: int = 1
    NODE: int = 2


class HashingAlgorithms(_BaseEnum):
    """Enumeration class, used to characterize the hashing algorithms"""

    SHA256 = 'SHA256'
    SHA384 = 'SHA384'
    SHA512 = 'SHA512'
    SHA3_256 = 'SHA3_256'
    SHA3_384 = 'SHA3_384'
    SHA3_512 = 'SHA3_512'
    BLAKE2B = 'BLAKE2B'
    BLAKE2S = 'BLAKE2S'


class TrainingPlanStatus(_BaseEnum):
    """Constant values for training plan type that will be saved into db

    Attributes:
        REQUESTED: means training plan submitted in-application by the researcher
        REGISTERED: means training plan added by a hospital/node
        DEFAULT: means training plan is default training plan provided by Fed-BioMed
    """
    REQUESTED = 'requested'
    REGISTERED = 'registered'
    DEFAULT = 'default'


class TrainingPlanApprovalStatus(_BaseEnum):
    """Enumeration class for training plan approval status of a training plan on a node when training plan approval
    is active.

    Attributes:
        APPROVED: training plan was accepted for this node, can be executed now
        REJECTED: training plan was disapproved for this node, cannot be executed
        PENDING: training plan is waiting for review and approval, cannot be executed yet
    """
    APPROVED = "Approved"
    REJECTED = "Rejected"
    PENDING = "Pending"
    
    def str2enum(name: str):
        for e in TrainingPlanApprovalStatus:
            if e.value == name:
                return e
        return None


class TrainingPlans(_BaseEnum):
    """Enumeration class for Training plans """

    TorchTrainingPlan = 'TorchTrainingPlan'
    SkLearnTrainingPlan = 'SkLearnTrainingPlan'


class ProcessTypes(_BaseEnum):
    """Enumeration class for Preprocess types

    Attributes:
        DATA_LOADER: Preprocess for DataLoader
        PARAMS: Preprocess for model parameters
    """
    DATA_LOADER = 0
    PARAMS = 1


class DataLoadingBlockTypes(_BaseEnum):
    """Base class for typing purposes.

    Concrete enumeration types should be defined within the scope of their
    implementation or application. To define a concrete enumeration type,
    one must subclass this class as follows:
    ```python
    class MyLoadingBlockTypes(DataLoadingBlockTypes, Enum):
        MY_KEY: str 'myKey'
        MY_OTHER_KEY: str 'myOtherKey'
    ```

    Subclasses must respect the following conditions:
    - All fields must be str;
    - All field values must be unique.

    !!! warning
        This class must always be empty as it is not allowed to
        contain any fields!
    """
    def __init__(self, *args):
        cls = self.__class__
        if not isinstance(self.value, str):
            raise ValueError("all fields of DataLoadingBlockTypes subclasses"
                             " must be of str type")
        if any(self.value == e.value for e in cls):
            a = self.name
            e = cls(self.value).name
            raise ValueError(
                f"duplicate values not allowed in DataLoadingBlockTypes and "
                f"its subclasses: {a} --> {e}")


class DatasetTypes(_BaseEnum):
    """Types of Datasets implemented in Fed-BioMed"""
    TABULAR = 'csv'
    IMAGES = 'images'
    DEFAULT = 'default'
    MEDNIST = 'mednist'
    MEDICAL_FOLDER = 'medical-folder'
    FLAMBY = 'flamby'
    TEST = 'test'
    NONE = 'none'


class SecaggElementTypes(_BaseEnum):
    """Enumeration class for secure aggregation element types

    Attributes:
        SERVER_KEY: server key split between the parties
        BIPRIME: biprime shared between the parties
    """
    SERVER_KEY: int = 0
    BIPRIME: int = 1


class ErrorNumbers(_BaseEnum):
    """List of all error messages types"""

    # MQTT errors
    FB100 = "FB100: undetermined messaging server error"
    FB101 = "FB101: cannot connect to the messaging server"
    FB102 = "FB102: messaging server does not answer in dedicated time"
    FB103 = "FB103: messaging call error"
    FB104 = "FB104: message exchange error"

    # HTTP errors

    FB200 = "FB200: undetermined repository server error"
    FB201 = "FB201: server not reachable"
    FB202 = "FB202: server returns 404 error"
    FB203 = "FB203: server returns other 4xx or 500 error"

    # application error on node

    FB300 = "FB300: undetermined node error"
    FB301 = "FB301: Protocol error"
    FB302 = "FB302: TrainingPlan class does not load"
    FB303 = "FB303: TrainingPlan class does not contain expected methods"
    FB304 = "FB304: TrainingPlan method crashes"
    FB305 = "FB305: TrainingPlan loops indefinitely"
    FB306 = "FB306: bad URL (.py) for TrainingPlan"
    FB307 = "FB307: bad URL (.pt) for training params"
    FB308 = "FB308: bad training request ().json"
    FB309 = "FB309: bad model params (.pt)"
    FB310 = "FB310: bad data format"
    FB311 = "FB311: receiving a new computation request during a running computation"
    FB312 = "FB312: Node stopped in SIGTERM signal handler"
    FB313 = "FB313: no dataset matching request"
    FB314 = "FB314: Node round error"
    FB315 = "FB315: Error while loading the data "
    FB316 = "FB316: Data loading plan error"
    FB317 = "FB317: FLamby package import error"
    FB318 = "FB318: Secure aggregation setup error"
    FB319 = "FB319: Command not found error"
    FB320 = "FB320: bad model type"
    FB321 = "FB321: Secure aggregation delete error"
    FB322 = "FB322: Dataset registration error"
    # application error on researcher

    FB400 = "FB400: undetermined application error"
    FB401 = "FB401: aggregation crashes or returns an error"
    FB402 = "FB402: strategy method crashes or sends an error"
    FB403 = "FB403: bad URL (.pt) for model param"
    FB404 = "FB404: bad model param (.pt) format for TrainingPlan"
    FB405 = "FB405: received delayed answer for previous computation round"
    FB406 = "FB406: list of nodes is empty at data lookup phase"
    FB407 = "FB407: list of nodes became empty when training (no node has answered)"
    FB408 = "FB408: node did not answer during training"
    FB409 = "FB409: node sent Status=Error during training"
    FB410 = "FB410: bad type or value for experiment argument"
    FB411 = "FB411: cannot train an experiment that is not fully defined"
    FB412 = "FB412: cannot do model checking for experiment"
    FB413 = "FB413: cannot save or load breakpoint for experiment"
    FB414 = "FB414: bad type or value for training arguments"
    FB415 = "FB415: secure aggregation handling error"

    # node problem detected by researcher

    FB500 = "FB500: undetermined node error, detected by server"
    FB501 = "FB501: node not reachable"

    # general application errors (common to node/researcher/..)

    FB600 = "FB600: environ error"
    FB601 = "FB601: message error"
    FB602 = "FB602: logger error"
    FB603 = "FB603: task queue error"
    FB604 = "FB604: repository error"
    FB605 = "FB605: training plan error"
    FB606 = "FB606: model manager error"
    FB607 = "FB607: data manager error"
    FB608 = "FB608: torch data manager error"
    FB609 = "FB609: scikit-learn data manager error"
    FB610 = "FB610: Torch based tabular dataset creation error"
    FB611 = "FB611: Error while trying to evaluate using the specified metric"
    FB612 = "FB612: Torch based NIFTI dataset error"
    FB613 = "FB613: Medical Folder dataset error"
    FB614 = "FB614: data loading block error"
    FB615 = "FB615: data loading plan error"
    FB616 = "FB616: differential privacy controller error"
    FB617 = "FB617: FLamby dataset error"
    FB618 = "FB618: FLamby data transformation error"
    FB619 = "FB619: Certificate error"
    # oops
    FB999 = "FB999: unknown error code sent by the node"


class UserRoleType(int, _BaseEnum):
    """Enumeration class, used to characterize the type of component of the fedbiomed architecture

    Attributes:
        ADMIN: User with Admin role
        USER: Simple user
    """

    ADMIN = 1
    USER = 2


class UserRequestStatus(str, _BaseEnum):
    """Enumeration class, used to characterize the status for user registration requests

        Attributes:
            NEW: New user registration
            REJECTED: Rejected status
        """

    NEW = "NEW"
    REJECTED = "REJECTED"

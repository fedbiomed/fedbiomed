# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Fed-BioMed constants/enums"""
import sys
import os


from packaging.version import Version as FBM_Component_Version
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed import __version__
from enum import Enum


CONFIG_FOLDER_NAME = "etc"
"""Directory/folder name where configurations are saved"""

CACHE_FOLDER_NAME = "cache"
"""Directory/folder name where cache files are saved"""

TMP_FOLDER_NAME = "tmp"
"""Directory/folder name where temporary files are saved"""

NOTEBOOKS_FOLDER_NAME = "notebooks"
"""Directory/folder name used by default for notebooks"""

NODE_DATA_FOLDER = "data"
"""Directory/folder name used by Nodes to save their specific dataset"""

TUTORIALS_FOLDER_NAME = "tutorials"
"""Directory/folder name used by default for tutorials"""

DOCS_FOLDER_NAME = "docs"
"""Directory/folder name used by default for documentation"""

TENSORBOARD_FOLDER_NAME = "runs"
"""Directory/folder name where tensorboard logs are saved"""

VAR_FOLDER_NAME = "var"
"""Directory/folder name where variable files are saved"""

DB_FOLDER_NAME = VAR_FOLDER_NAME
"""Directory/folder name where DB files are saved"""

DB_PREFIX = 'db_'
"""Prefix for database files name"""

NODE_PREFIX = 'node_'
"""Prefix for node ID"""

NODE_STATE_PREFIX = 'node_state_'
"""Prefix for Node state ID"""

EXPERIMENT_PREFIX = 'exper_'
"""Prefix for experiment ID"""

REQUEST_PREFIX = 'request_'
"""Prefix for request ID"""

DEFAULT_NODE_NAME = 'fbm-node'
"""Default node component folder name"""

DEFAULT_RESEARCHER_NAME = 'fbm-researcher'
"""Default researcher component folder name"""

CERTS_FOLDER_NAME = os.path.join(CONFIG_FOLDER_NAME, 'certs')
"""FOLDER name for Certs directory"""

TRACEBACK_LIMIT = 20

DEFAULT_CERT_NAME = "FBM_certificate"
SERVER_certificate_prefix = "server_certificate"

# !!! info "Instructions for developers"
# If you make a change that changes the format / metadata / structure of one of the components below,
# you ** must update ** the version.
#
# Instructions for updating the version
#
# 1. check [versions page](https://fedbiomed.org/latest/user-guide/deployment/versions)
# for background information
# 2. bump the version below: if your change breaks backward compatibility you must increase the
# major version, else the minor version. Micro versions are supported but their use is currently discouraged.

__version__ = FBM_Component_Version(__version__)  # Fed-BioMed software version
__researcher_config_version__ = FBM_Component_Version('3')  # researcher config file version
__node_config_version__ = FBM_Component_Version('2')  # node config file version
__node_state_version__ = FBM_Component_Version('2')  # node state version
__breakpoints_version__ = FBM_Component_Version('3')  # breakpoints format version
__messaging_protocol_version__ = FBM_Component_Version('5')  # format of gRPC messages.
__secagg_element_version__ = FBM_Component_Version('2')  # format of secagg database elements
__n2n_channel_element_version__ = FBM_Component_Version('1')  # format of n2n channels database elements

# Nota: for messaging protocol version, all changes should be a major version upgrade

# Max message length as bytes
MAX_MESSAGE_BYTES_LENGTH = 4000000 - sys.getsizeof(bytes("", encoding="UTF-8"))  # 4MB

# Max number of retries for sending message (node and researcher side)
MAX_SEND_RETRIES = 5

# Max number of retries for retrieving a task when error occurs (on the node)
MAX_RETRIEVE_ERROR_RETRIES = 5

# Timeout for a node to node request
#
# Intentionally high to support scaling to great number of nodes
# In typical scenario 5 seconds is enough with 10 nodes
TIMEOUT_NODE_TO_NODE_REQUEST = 30


class _BaseEnum(Enum):
    """
    Parent class to pass default methods to enumeration classes
    """

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class MessageType(_BaseEnum):
    """Types of messages received by researcher

    Attributes:
        REPLY: reply messages (TrainReply, SearchReply, etc.)
        LOG: 'log' message (LogMessage)
        SCALAR: 'add_scalar' message (Scalar)
    """
    REPLY = "REPLY"
    LOG = "LOG"
    SCALAR = "SCALAR"

    @classmethod
    def convert(cls, type_):
        """Converts given text message to to MessageType instance"""
        try:
            return getattr(cls, type_.upper())
        except AttributeError as exp:
            raise FedbiomedError(f"There is no MessageType as {type_}") from exp


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


class SecureAggregationSchemes(_BaseEnum):
    """Enumeration class for secure aggregation schemes"""
    NONE: int = 0
    JOYE_LIBERT: int = 1
    LOM: int = 2


class SecaggElementTypes(_BaseEnum):
    """Enumeration class for secure aggregation element types

    Attributes:
        SERVER_KEY: server key split between the parties
        DIFFIE_HELLMAN: one pair of DH key for each node party, public key shared with other node parties
    """
    SERVER_KEY: int = 0
    DIFFIE_HELLMAN: int = 1

    @staticmethod
    def get_element_from_value(element_value: int):
        for element in SecaggElementTypes:
            if element.value == element_value:
                return element


class SAParameters:
    CLIPPING_RANGE: int = 3
    TARGET_RANGE: int = 2**13
    WEIGHT_RANGE: int = 2**17 # TODO: this has to be provided by the researcher, find the max range among all the nodes' weights
    #TODO: to separete from SAParameters
    KEY_SIZE: int = 2048


class ErrorNumbers(_BaseEnum):
    """List of all error messages types"""

    # GRPC errors
    FB100 = "FB100: undetermined messaging server error"

    # application error on node

    FB300 = "FB300: undetermined node error"
    FB301 = "FB301: Protocol error"
    FB302 = "FB302: TrainingPlan class does not load"
    FB303 = "FB303: TrainingPlan class does not contain expected methods"
    FB304 = "FB304: TrainingPlan method crashes"
    FB309 = "FB309: bad model params (.mpk)"
    FB310 = "FB310: bad data format"
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
    FB323 = "FB323: Node State error"
    FB324 = "FB324: Node to node overlay communication error"

    # application error on researcher

    FB400 = "FB400: undetermined application error"
    FB401 = "FB401: aggregation crashes or returns an error"
    FB402 = "FB402: strategy method crashes or sends an error"
    FB407 = "FB407: list of nodes became empty when training (all nodes failed training or did not answer)"
    FB408 = "FB408: training failed on node or node did not answer during training"
    FB409 = "FB409: node sent Status=Error during training"
    FB410 = "FB410: bad type or value for experiment argument"
    FB411 = "FB411: cannot train an experiment that is not fully defined"
    FB412 = "FB412: cannot do model checking for experiment"
    FB413 = "FB413: cannot save or load breakpoint for experiment"
    FB414 = "FB414: bad type or value for training arguments"
    FB415 = "FB415: secure aggregation handling error"
    FB416 = "FB416: federated dataset error"
    FB417 = "FB417: secure aggregation error"
    FB419 = "FB419: node state agent error"

    # general application errors (common to node/researcher/..)

    FB600 = "FB600: configuration error"
    FB601 = "FB601: message error"
    FB603 = "FB603: task queue error"
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
    FB620 = "FB620: MPC protocol error"
    FB621 = "FB621: declearn optimizer error"
    FB622 = "FB622: Model error"
    FB623 = "FB623: Secure aggregation database error"
    FB624 = "FB624: Secure aggregation crypter error"
    FB625 = "FB625: Component version error"
    FB626 = "FB626: Fed-BioMed optimizer error"
    FB627 = "FB627: Utility function error"
    FB628 = "FB628: Communication error"
    FB629 = "FB629: Diffie-Hellman KA error"
    FB630 = "FB630: Additive Secret Sharing error"
    FB631 = 'FB631: Node to node channels database error'
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

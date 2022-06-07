"""Fed-BioMed constants/enums"""

from enum import Enum


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

    SHA256: str = 'SHA256'
    SHA384: str = 'SHA384'
    SHA512: str = 'SHA512'
    SHA3_256: str = 'SHA3_256'
    SHA3_384: str = 'SHA3_384'
    SHA3_512: str = 'SHA3_512'
    BLAKE2B: str = 'BLAKE2B'
    BLAKE2S: str = 'BLAKE2S'


class ModelTypes(_BaseEnum):
    """Constant values for model type that will be saved into db

    Attributes:
        REQUESTED: means model submitted in-application by the researcher
        REGISTERED: means model added by a hospital/node
        DEFAULT: means model is default model provided by Fed-BioMed
    """
    REQUESTED: str = 'requested'
    REGISTERED: str = 'registered'
    DEFAULT: str = 'default'


class ModelApprovalStatus(_BaseEnum):
    """Enumeration class for model approval status of a model on a node when model approval is active.

    Attributes:
        APPROVED: model was accepted for this node, can be executed now
        REJECTED: model was disapproved for this node, cannot be executed
        PENDING: model is waiting for review and approval, cannot be executed yet
    """
    APPROVED: str = "Approved"
    REJECTED: str = "Rejected"
    PENDING: str = "Pending"


class TrainingPlans(_BaseEnum):
    """Enumeration class for Training plans """

    TorchTrainingPlan: str = 'TorchTrainingPlan'
    SkLearnTrainingPlan: str = 'SkLearnTrainingPlan'


class ProcessTypes(_BaseEnum):
    """Enumeration class for Preprocess types

    Attributes:
        DATA_LOADER: Preprocess for DataLoader
        PARAMS: Preprocess for model parameters
    """
    DATA_LOADER: int = 0
    PARAMS: int = 1


class ErrorNumbers(_BaseEnum):
    """List of all error messages types"""

    # MQTT errors
    FB100: str = "FB100: undetermined messaging server error"
    FB101: str = "FB101: cannot connect to the messaging server"
    FB102: str = "FB102: messaging server does not answer in dedicated time"
    FB103: str = "FB103: messaging call error"
    FB104: str = "FB104: message exchange error"

    # HTTP errors

    FB200: str = "FB200: undetermined repository server error"
    FB201: str = "FB201: server not reachable"
    FB202: str = "FB202: server returns 404 error"
    FB203: str = "FB203: server returns other 4xx or 500 error"

    # application error on node

    FB300: str = "FB300: undetermined node error"
    FB301: str = "FB301: Protocol error"
    FB302: str = "FB302: TrainingPlan class does not load"
    FB303: str = "FB303: TrainingPlan class does not contain expected methods"
    FB304: str = "FB304: TrainingPlan method crashes"
    FB305: str = "FB305: TrainingPlan loops indefinitely"
    FB306: str = "FB306: bad URL (.py) for TrainingPlan"
    FB307: str = "FB307: bad URL (.pt) for training params"
    FB308: str = "FB308: bad training request ().json"
    FB309: str = "FB309: bad model params (.pt)"
    FB310: str = "FB310: bad data format"
    FB311: str = "FB311: receiving a new computation request during a running computation"
    FB312: str = "FB312: Node stopped in SIGTERM signal handler"
    FB313: str = "FB313: no dataset matching request"
    FB314: str = "FB314: Node round error"
    FB315: str = "FB315: Error while loading the data "

    # application error on researcher

    FB400: str = "FB400: undetermined application error"
    FB401: str = "FB401: aggregation crashes or returns an error"
    FB402: str = "FB402: strategy method crashes or sends an error"
    FB403: str = "FB403: bad URL (.pt) for model param"
    FB404: str = "FB404: bad model param (.pt) format for TrainingPlan"
    FB405: str = "FB405: received delayed answer for previous computation round"
    FB406: str = "FB406: list of nodes is empty at data lookup phase"
    FB407: str = "FB407: list of nodes became empty when training (no node has answered)"
    FB408: str = "FB408: node did not answer during training"
    FB409: str = "FB409: node sent Status=Error during training"
    FB410: str = "FB410: bad type or value for experiment argument"
    FB411: str = "FB411: cannot train an experiment that is not fully defined"
    FB412: str = "FB412: cannot do model checking for experiment"
    FB413: str = "FB413: cannot save or load breakpoint for experiment"

    # node problem detected by researcher

    FB500: str = "FB500: undetermined node error, detected by server"
    FB501: str = "FB501: node not reachable"

    # general application errors (common to node/researcher/..)

    FB600: str = "FB600: environ error"
    FB601: str = "FB601: message error"
    FB602: str = "FB602: logger error"
    FB603: str = "FB603: task queue error"
    FB604: str = "FB604: repository error"
    FB605: str = "FB605: training plan error"
    FB606: str = "FB606: model manager error"
    FB607: str = "FB607: data manager error"
    FB608: str = "FB608: torch data manager error"
    FB609: str = "FB609: scikit-learn data manager error"
    FB610: str = "FB610: Torch based tabular dataset creation error"
    FB611: str = "FB611: Error while trying to evaluate using the specified metric"
    FB612: str = "FB612: Torch based NIFTI dataset error"
    FB613: str = "FB613: BIDS dataset error"


    # oops
    FB999: str = "FB999: unknown error code sent by the node"

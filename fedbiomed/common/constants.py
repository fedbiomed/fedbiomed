from enum import Enum


class _BaseEnum(Enum):
    """ Parent class to pass default methods to
        enumeration classes
    """

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class ComponentType(_BaseEnum):
    """
    Enumeration class, used to characterize the type
    of component of the fedbiomed architecture.
    """

    RESEARCHER = 1
    NODE       = 2


class HashingAlgorithms(_BaseEnum):

    """ Enumeration class, used to characterize the hashing
    algorithms
    """

    SHA256      = 'SHA256'
    SHA384      = 'SHA384'
    SHA512      = 'SHA512'
    SHA3_256    = 'SHA3_256'
    SHA3_384    = 'SHA3_384'
    SHA3_512    = 'SHA3_512'
    BLAKE2B     = 'BLAKE2B'
    BLAKE2S     = 'BLAKE2S'


class ModelTypes(_BaseEnum):

    """ Constant values for model type that will be saved into db.
    - `registered` means model saved by a user/hospital/node.
    - `default`    means model is default model provided by Fed-BioMed.
    """

    REGISTERED = 'registered'
    DEFAULT    = 'default'


class ErrorNumbers(_BaseEnum):

    """
    list of all error messages types

    the first value of each enum is used for JSON serializing/deserializing
    the second value is a description string
    """

    # MQTT errors
    FB100 = "FB100: undetermined messaging server error"
    FB101 = "FB101: server does not answer in dedicated time"
    FB102 = "FB102: mqqt call error"
    FB103 = "FB103: message echange error"

    # HTTP errors

    FB200 = "FB200: undetermined repository server error"
    FB201 = "FB201: server not reachable"
    FB202 = "FB202: server return 404 error"
    FB203 = "FB203: server return other 4xx or 500 error"

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

    # application error on researcher

    FB400 = "FB400: undetermined application error"
    FB401 = "FB401: fedaverage method crashes or returns an error"
    FB402 = "FB402: strategy method creashes or sending an error"
    FB403 = "FB403: bad URL (.pt) for model param"
    FB404 = "FB404: bad model param (.pt) format for TrainingPlan"
    FB405 = "FB405: received delayed answer for previous computation round"
    FB406 = "FB406: list of nodes is empty at data lookup phase"
    FB407 = "FB407: list of nodes became empty then training"
    FB408 = "FB408: node did not answer during training"
    FB409 = "FB409: node sent Status=Error during training"
    FB410 = "FB410: Please set training arguments with `.set_training_args()` before setting a `Job`."
    FB411 = "FB410: No Federated Dataset is found. Please use `.set_training_data()` before setting a `Job`."
    FB412 = "FB410: `model_class` is mandatory for setting `Job`.  Please initialize experiment with model class or " \
            "use `.set_model_class()` method of the experiment"
    FB413 = "FB413: Please create job with `.set_job()` before running the experiment"
    FB414 = "FB414: Please set client selection strategy"
    FB415 = "FB415: Please set an aggregator"
    FB416 = "FB416: Please set a training_data before setting `node_selection_strategy`. You can use " \
            "`.set_training_data(tags: list)` to set a training data"
    FB417 = "FB417: Training data is not a type of FederatedDataset or Dict"
    FB418 = "FB418: No tags has been found that are already set and it is not provided with " \
            "`set_training_data` too. Please pass `tags=[<tag>]`  or set `training_data=<FederatedDataset>`"


    # node problem detected by researcher

    FB500 = "FB500: undetermined node error, detected by server"
    FB501 = "FB501: node not reacheable"

    # oops

    FB999 = "FB999: unknown error code sent by the node"

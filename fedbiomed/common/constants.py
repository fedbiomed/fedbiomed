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


class ErrorHandling(_BaseEnum):

    """
    list of all error messages types
    """

    # MQTT errors
    FB100 = "server does not answer in dedicated time"
    FB101 = "mqqt call error"
    FB102 = "message echange error"

    # HTTP errors

    FB200 = "server not reachable"
    FB201 = "server return 404 error"
    FB202 = "server return other 4xx or 500 error"

    # application error on node

    FB300 = "TrainingPlan class does not load"
    FB301 = "TrainingPlan class does not contain expected methods"
    FB302 = "a TrainingPlan method crashes"
    FB303 = "a TrainingPlan loops indefinitely"
    FB304 = "bad URL (.py) for TrainingPlan"
    FB305 = "bad URL (.pt) for training params"
    FB306 = "bad training request ().json"
    FB307 = "bad model params (.pt)"
    FB308 = "bad data format"
    FB309 = "receiving a new computation request during a running computation"

    # application error on researcher
    FB400 = "fedaverage method crashes or returns an error"
    FB401 = "strategy method creashes or sending an error"
    FB402 = "bad URL (.pt) for model param"
    FB403 = "bad model param (.pt) format for TrainingPlan"
    FB404 = "received delayed answer for previous computation round"
    FB405 = "list of nodes is empty at data lookup phase"
    FB406 = "list of nodes became empty then training"

    # node problem detected by researcher
    FB500 = "receiving error message from node"  ### useless ???
    FB501 = "node not reacheable"

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
    FB100 = [ 100, "undetermined messaging server error"]
    FB101 = [ 101, "server does not answer in dedicated time" ]
    FB102 = [ 102, "mqqt call error" ]
    FB103 = [ 103, "message echange error" ]

    # HTTP errors

    FB200 = [ 200, "undetermined repository server error"]
    FB201 = [ 201, "server not reachable"]
    FB202 = [ 202, "server return 404 error"]
    FB203 = [ 203, "server return other 4xx or 500 error"]

    # application error on node

    FB300 = [ 300, "undetermined node error"]
    FB301 = [ 301, "Node killed/stopped by a human"]
    FB302 = [ 302, "TrainingPlan class does not load"]
    FB303 = [ 303, "TrainingPlan class does not contain expected methods"]
    FB304 = [ 304, "a TrainingPlan method crashes"]
    FB305 = [ 305, "a TrainingPlan loops indefinitely"]
    FB306 = [ 306, "bad URL (.py) for TrainingPlan"]
    FB307 = [ 307, "bad URL (.pt) for training params"]
    FB308 = [ 308, "bad training request ().json"]
    FB309 = [ 309, "bad model params (.pt)"]
    FB310 = [ 310, "bad data format"]
    FB311 = [ 311, "receiving a new computation request during a running computation"]

    # application error on researcher

    FB400 = [ 400, "undetermined application error"]
    FB401 = [ 401, "fedaverage method crashes or returns an error"]
    FB402 = [ 402, "strategy method creashes or sending an error"]
    FB403 = [ 403, "bad URL (.pt) for model param"]
    FB404 = [ 404, "bad model param (.pt) format for TrainingPlan"]
    FB405 = [ 405, "received delayed answer for previous computation round"]
    FB406 = [ 406, "list of nodes is empty at data lookup phase"]
    FB407 = [ 407, "list of nodes became empty then training"]

    # node problem detected by researcher

    FB500 = [ 500, "undetermined node error, detected by server" ]
    FB501 = [ 501, "node not reacheable"]

    # oops

    FB999 = [ 999, "unknown error code sent by the node"]

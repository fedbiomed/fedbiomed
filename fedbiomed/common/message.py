from dataclasses import dataclass
from typing import Dict, Any, Union


class Message(object):
    """
    This class defines the structure of a 
    message sent/recieved via MQTT

    """
    def __init__(self):
        """ Constructor of the class
        """        
        pass

    def set_param(self, param: str, param_value: Any):
        """This method allows to modify the value of a given param

        Args:
            param (str): the name of the param to be modified
            param_value: new value of the param
        """        
        setattr(self, param, param_value)

    def get_param(self, param: str):
        """This method allows to get the value of a given param

        Args:
            param (str): name of the param
        """        
        return(getattr(self, param))

    def get_dict(self) -> Dict[str, Any]:
        """Returns pairs (Message class attributes, values)
        as a dictionary
        
        """
        return(self.__dict__)
    
    def validate(self, fields:Dict[str, Any]) -> bool:
        """whether incoming field types match with attributes
            class type. 

        Args:
            fields (Dict[str, Any]): incoming fields

        Returns:
            bool: If validated, ie everything matches,
            returns True, else returns False.
        """
        ret = True
        for field_name, field_def in fields:
            actual_type = type(getattr(self, field_name))
            if actual_type != field_def.type:
                print(f"\t{field_name}: '{actual_type}' instead of '{field_def.type}'")
                ret = False
        return ret


@dataclass
class SearchReply(Message):
    """This class describes a search message sent by the node

    Args:
        Message ([type]): Parent class allows to get and set message params
    """ 
    researcher_id: str
    success: bool
    databases: list
    count: int
    client_id: str
    command: str

    def __post_init__(self):
        if not self.validate(self.__dataclass_fields__.items()):
            raise ValueError('Wrong types')

    


@dataclass
class PingReply(Message):
    """
    This class describes a ping message sent by the node
    """
    researcher_id: str
    client_id: str
    success: bool
    command: str

    def __post_init__(self):
        if not self.validate(self.__dataclass_fields__.items()):
            raise ValueError('Wrong types')

@dataclass
class TrainReply(Message):
    """
    This class describes a train message sent by the node
    """
    researcher_id: str
    job_id: str
    success: bool
    client_id: str
    dataset_id: str
    params_url: str
    timing: dict
    msg: str
    command: str

    def __post_init__(self):
        if not self.validate(self.__dataclass_fields__.items()):
            raise ValueError('Wrong types')

@dataclass
class AddScalarReply(Message):
    """
    This class describes a add_scalar message sent by the node
    """
    researcher_id: str
    client_id: str
    job_id: str
    key: float
    iteration: int
    command: str

    def __post_init__(self):
        if not self.validate(self.__dataclass_fields__.items()):
            raise ValueError('Wrong types')

@dataclass
class ErrorMessage(Message):
    """
    This class describes an error message sent by the node
    """
    researcher_id: str
    success: bool
    client_id: str
    msg: str
    command: str

    def __post_init__(self):
        if not self.validate(self.__dataclass_fields__.items()):
            raise ValueError('Wrong types')


@dataclass
class SearchRequest(Message):
    """
    This class describes a search message sent by the researcher
    """
    researcher_id: str
    tags: list
    command: str

    def __post_init__(self):
        if not self.validate(self.__dataclass_fields__.items()):
            raise ValueError('Wrong types')


@dataclass
class PingRequest(Message):
    """
    This class describes a ping message sent by the researcher
    
    Raises:
        ValueError: triggered if case type validation failed
    """
    researcher_id: str
    command: str

    def __post_init__(self):
        if not self.validate(self.__dataclass_fields__.items()):
            raise ValueError('Wrong types')


@dataclass
class TrainRequest(Message):
    """
    This class describes a train message sent by the researcher
    
    Raises:
        ValueError: triggered if case validation failed
    """
    researcher_id: str
    job_id: str
    params_url: str
    training_args: dict
    training_data: dict
    model_args: dict
    model_url: str
    model_class: str
    command: str

    def __post_init__(self):
        if not self.validate(self.__dataclass_fields__.items()):
            raise ValueError('Wrong types')


class ResearcherMessages():
    """This class allows to create the corresponding class instance from
    a received/ sent message by the researcher
    """    
    @classmethod
    def reply_create(cls, params: Dict[str, Any]):
        """this method is used on message reception.
        It validates:
        - the legacy of the message
        - the structure of the received message

        Raises:
        ValueError: if the message is not allowed te be received by the researcher

        Returns:
        An instance of the corresponding class
        """
        message_type = params['command']
        
        MESSAGE_TYPE_TO_CLASS_MAP = {'train':  TrainReply,
                                    'search': SearchReply,
                                    'ping': PingReply,
                                    'error': ErrorMessage,
                                    'add_scalar':AddScalarReply
        }

        if message_type not in MESSAGE_TYPE_TO_CLASS_MAP:
            raise ValueError('Bad message type {}'.format(message_type))

        return MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)

    @classmethod
    def request_create(cls, params:Dict[str, Any]):

        """This method creates the adequate message:
        It validates:
        - the legagy of the message
        - the structure of the created message

        Args:
        params (dict):
        
        Raises:
            ValueError: if the message is not allowed te be sent by the researcher
            KeyError ?
        Returns:
            An instance of the corresponding class
        """

        message_type = params['command']

        MESSAGE_TYPE_TO_CLASS_MAP = {'train':  TrainRequest,
                                    'search': SearchRequest,
                                    'ping': PingRequest
        }

        if message_type not in MESSAGE_TYPE_TO_CLASS_MAP:
            raise ValueError('Bad message type {}'.format(message_type))

        return MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)


class NodeMessages():
    """This class allows to create the corresponding class instance from
    a received/sent message by the Node
    """
    @classmethod
    def request_create(cls, params: dict) -> Union[TrainRequest, SearchRequest, PingRequest]:
        """
        This method create the adequate message, it maps an instruction (given the key "command" in
        the input dictionary `params`) to a Message object
        
        It validates:
        - the legagy of the message
        - the structure of the created message
        
        Raises:
            ValueError: triggered if the message is not allowed te be sent by the node

        Returns:
            An instance of the corresponding class (TrainRequest, SearchRequest, PingRequest)
        """
        message_type = params['command'] # can be "train", "search", or "ping"
        # mapping message type to an object
        MESSAGE_TYPE_TO_CLASS_MAP = {'train':  TrainRequest,
                                    'search': SearchRequest,
                                    'ping': PingRequest,
        }

        if message_type not in MESSAGE_TYPE_TO_CLASS_MAP:
            raise ValueError('Bad message type {}'.format(message_type))
        #import remote_pdb; remote_pdb.set_trace()
        return MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)


    @classmethod
    def reply_create(cls, params: dict) -> Union[TrainReply,
                                                 SearchReply,
                                                 PingReply,
                                                 ErrorMessage,
                                                 AddScalarReply]:
        """this method is used on message reception.
        It validates:
        - the legacy of the message
        - the structure of the received message

        Raises:
            ValueError: if the message is not allowed te be received by the node

        Returns:
            An instance of the corresponding class
        """
        message_type = params['command']
        MESSAGE_TYPE_TO_CLASS_MAP = {'train':  TrainReply,
                                    'search': SearchReply,
                                    'ping': PingReply,
                                    'error': ErrorMessage,
                                    'add_scalar':AddScalarReply
        }

        if message_type not in MESSAGE_TYPE_TO_CLASS_MAP:
            raise ValueError('Bad message type {}'.format(message_type))

        return MESSAGE_TYPE_TO_CLASS_MAP[message_type](**params)
        

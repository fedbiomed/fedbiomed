import json

from typing import Union


def deserialize_msg(msg: Union[str, bytes]) -> dict:
    """
    Deserializes a JSON string or bytes message as a dictionary.
    :param msg: message in JSON format but encoded as string or bytes
    :return: parsed message as python dictionary.
    """
    return json.loads(msg)


def serialize_msg(msg: dict) -> str:
    """
    Serialize an object as a JSON message (applies for dict-like objects)
    :param msg: dict-like object containing the message to send.
    :return: JSON parsed message ready to transmit.
    """
    return json.dumps(msg)

"""
This module defines message serializer and deserializer
for sending / receiving / parsing messages through Messager.

compared to the usual json module, it deals with some fedbiomed
data types which are not serialized by default (eg: enumerations)
"""

import json

from typing import Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.metrics import MetricTypes


def deserialize_msg(msg: Union[str, bytes]) -> dict:
    """
    Deserializes a JSON string or bytes message as a dictionary.
    :param msg: message in JSON format but encoded as string or bytes
    :return: parsed message as python dictionary.
    """
    decode = json.loads(msg)

    # If message is train request
    decode = _deserialize_test_metric(decode)

    # errnum is present in ErrorMessage and is an Enum
    # which need to be deserialized
    if 'errnum' in decode:
        errnum = decode['errnum']
        found = False
        for e in ErrorNumbers:
            if e.value == errnum:
                found = True
                decode['errnum'] = e
                break
        if not found:
            # error code sent by the node is unknown
            decode['errnum'] = ErrorNumbers.FB999

    return decode


def serialize_msg(msg: dict) -> str:
    """
    Serialize an object as a JSON message (applies for dict-like objects)
    :param msg: dict-like object containing the message to send.
    :return: JSON parsed message ready to transmit.
    """

    # errnum is present in ErrorMessage and is an Enum
    # which need to be serialized

    # If message is train request
    msg = _serialize_test_metric(msg)

    if 'errnum' in msg:
        msg['errnum'] = msg['errnum'].value
    return json.dumps(msg)


def _deserialize_test_metric(msg):
    if msg.get('training_args', False):
        metric = msg['training_args'].get('test_metric', False)
        if metric:
            msg['training_args']['test_metric'] = MetricTypes.get_metric_type_by_name(metric)
    return msg


def _serialize_test_metric(msg):
    if msg.get('training_args', False):
        metric = msg['training_args'].get('test_metric', False)
        if metric and isinstance(metric, MetricTypes):
            msg['training_args']['test_metric'] = metric.name
    return msg

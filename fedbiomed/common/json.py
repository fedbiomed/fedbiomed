# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
This module defines message serializer and deserializer for sending / receiving / parsing messages through Messaging.

Compared to the usual json module, it deals with some fedbiomed data types which are not serialized by
default (eg: enumerations)
"""


import json

from typing import Union

from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.training_args import TrainingArgs


def deserialize_msg(msg: Union[str, bytes]) -> dict:
    """Deserializes a JSON string or bytes message as a dictionary.

    Args:
        msg: message in JSON format but encoded as string or bytes
    Returns:
        Parsed message as python dictionary.
    """
    decode = json.loads(msg)

    # deserialize our own types/classes
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
    """Serialize an object as a JSON message (applies for dict-like objects)

    Args:
        msg: dict-like object containing the message to send.

    Returns:
        JSON parsed message ready to transmit.
    """

    # serialize our own types/classes
    msg = _serialize_training_args(msg)
    msg = _serialize_test_metric(msg)

    # Errnum is present in ErrorMessage and is an Enum
    # which need to be serialized

    if 'errnum' in msg:
        msg['errnum'] = msg['errnum'].value
    return json.dumps(msg)



def _serialize_training_args(msg):
    """TrainingArgs is a class and must be specifically serialized"""
    if 'training_args' in msg:
        if isinstance(msg['training_args'], TrainingArgs):
            msg['training_args'] = msg['training_args'].dict()
    return msg


def _deserialize_test_metric(msg):
    """MetricTypes is an enum and must be specifically deserialized."""
    if 'training_args' in msg:
        metric = msg['training_args'].get('test_metric', False)
        if metric:
            msg['training_args']['test_metric'] = MetricTypes.get_metric_type_by_name(metric)
    return msg


def _serialize_test_metric(msg):
    """MetricTypes is an enum and must be specifically serialized."""
    if 'training_args' in msg:
        metric = msg['training_args'].get('test_metric', False)
        if metric and isinstance(metric, MetricTypes):
            msg['training_args']['test_metric'] = metric.name
    return msg

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class PolicyStatus(Enum):
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    CONTINUE = "CONTINUE"


class RequestStatus(Enum):
    DISCONNECT = "DISCONNECT"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    NO_REPLY_YET = "NO_REPLY_YET"

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

class PolicyStatus:

    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    CONTINUE = "CONTINUE"


class RequestStatus:

    DISCONNECT = "DISCONNECT"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    NO_REPLY_YET = "NO_REPLY_YET"

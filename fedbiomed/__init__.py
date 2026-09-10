# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

__version__ = "v6.4.1"

import os

# gRPC logs every failed TLS handshake at INFO, so a node retrying its connection
# floods the researcher output. The C core reads this once at `import grpc`, which
# declearn triggers well before `fedbiomed.transport` — hence the top level package.
# A preset value wins: raising it to INFO is how an operator sees each rejection.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

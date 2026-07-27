# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Transport module to manage communication between researcher and
node components based on gRPC
"""

import os

# gRPC reports every failed TLS handshake at INFO, and a node retrying its
# connection turns that into unbounded output on the researcher. Read by the C
# core as it initialises, so it is set here, before the submodules `import grpc`.
# A value already in the environment wins: raising it to INFO is how an operator
# sees the individual rejections while diagnosing.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

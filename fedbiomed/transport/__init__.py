# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Transport module to manage communication between researcher and
node components based on gRPC
"""

import os

# Silence gRPC C-core INFO logs (e.g. each failed TLS handshake, reported
# node-side instead); must be set before submodules `import grpc`.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

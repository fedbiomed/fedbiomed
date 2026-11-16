#!/usr/bin/env python3
# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Mints a capability for the mock inference guardian (PoC-only).

Python equivalent of the mock's `capability_generator.sh`
(hasan7n/tmp-policies, `tools/dev/mock_inference_guardian`), which relies on
the GNU-only `base64 -w 0`. The capability authorizes the given script digest, and is
written base64-encoded, the way the mock guardian expects it. Real capabilities
are encrypted; this one is only encoded. The node never parses it either way.

Usage:
    python mint_capability.py sha256:<hex> [-o ./capability.b64]

or from a notebook:

    from mint_capability import mint_capability
    capability = mint_capability("sha256:<hex>")
"""

import argparse
import base64
import binascii
import json
import os
import sys

METHOD_NAME = "do_inference"


def _rand_b64(size: int) -> str:
    """Returns `size` random bytes, base64-encoded."""
    return base64.b64encode(os.urandom(size)).decode("ascii")


def mint_capability(digest: str) -> str:
    """Builds a mock capability authorizing a script digest.

    Args:
        digest: Script digest the capability authorizes, e.g. `sha256:<hex>`.

    Returns:
        The capability, base64-encoded.
    """
    operation = {
        "nonce": _rand_b64(32),
        "request_identifier": _rand_b64(8),
        "method_name": METHOD_NAME,
        "parameters": {"script_digest": digest},
    }
    capability = {
        "minted_identity": _rand_b64(32),
        "operation": {
            "encrypted_session_key": _rand_b64(256),
            "session_key_iv": _rand_b64(16),
            "encrypted_message": base64.b64encode(
                json.dumps(operation).encode("utf-8")
            ).decode("ascii"),
        },
    }
    return base64.b64encode(json.dumps(capability).encode("utf-8")).decode("ascii")


def decode_capability(capability: str) -> dict:
    """Decodes a capability into a readable structure.

    The operation of a mock capability is only base64-encoded, so it is decoded
    too. The operation of a real capability is encrypted: it is then left as is
    and `operation_readable` is False.

    Args:
        capability: The capability, base64-encoded.

    Returns:
        The capability package, with `operation.encrypted_message` replaced by
            the decoded operation when it is readable.

    Raises:
        ValueError: if the capability is not a base64-encoded JSON object.
    """
    try:
        package = json.loads(base64.b64decode(capability, validate=True))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"capability is not a base64-encoded JSON object: {e}") from e

    secret = package.get("operation", {})
    try:
        operation = json.loads(base64.b64decode(secret["encrypted_message"]))
    except (
        KeyError,
        TypeError,
        binascii.Error,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        package["operation_readable"] = False
        return package

    package["operation"] = {**secret, "encrypted_message": operation}
    package["operation_readable"] = True
    return package


def format_capability(capability: str, max_field: int = 24) -> str:
    """Formats a capability for display.

    Args:
        capability: The capability, base64-encoded.
        max_field: Random base64 fields longer than this are shortened.

    Returns:
        A multi-line, human-readable description of the capability.
    """
    package = decode_capability(capability)

    def shorten(value):
        if isinstance(value, dict):
            return {k: shorten(v) for k, v in value.items()}
        if isinstance(value, str) and len(value) > max_field:
            return f"{value[:max_field]}... ({len(value)} chars)"
        return value

    lines = [f"Capability package ({len(capability)} bytes, base64)"]
    operation = package["operation"].get("encrypted_message")
    if package["operation_readable"]:
        parameters = operation.get("parameters", {})
        lines += [
            f"  method:              {operation.get('method_name')}",
            f"  authorized digest:   {parameters.get('script_digest')}",
            f"  request identifier:  {operation.get('request_identifier')}",
        ]
    else:
        lines.append("  operation is encrypted: only the guardian can read it")
    readable = {k: v for k, v in package.items() if k != "operation_readable"}
    lines += ["  full package:"] + [
        "    " + line for line in json.dumps(shorten(readable), indent=2).splitlines()
    ]
    return "\n".join(lines)


def main() -> int:
    """Parses command line arguments and writes the capability file."""
    parser = argparse.ArgumentParser(
        description="Mint a mock capability authorizing a script digest."
    )
    parser.add_argument("digest", help="script digest to authorize, e.g. sha256:<hex>")
    parser.add_argument(
        "-o", "--output", default="./capability.b64", help="output file"
    )
    args = parser.parse_args()

    capability = mint_capability(args.digest)
    with open(args.output, "w") as f:
        f.write(capability)

    print(format_capability(capability) + "\n", file=sys.stderr)
    print(
        f"Capability written to {args.output} ({len(capability)} bytes, base64).\n\n"
        f"Redeem it with the mock's redeem.py:\n\n  redeem.py -f {args.output} -c {args.digest}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

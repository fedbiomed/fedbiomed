#!/usr/bin/env python
# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Development stub of a capability guardian service.

This is NOT a policy authority. It performs no signature verification and no
authorization decision beyond comparing two checksums. It exists so that the
node-side guardian client can be exercised end-to-end while the real service is
being built. Do not deploy it.

Usage:
    python scripts/mock_guardian_server.py [--host HOST] [--port PORT]

Serves `POST /verify`, which expects a JSON body of the form:

    {
      "capabilities": {"training_plan_checksum": "<signed checksum>", ...},
      "training_plan_checksum": "<checksum computed by the node>",
      ...
    }

and answers `{"valid": <bool>, "reason": "<str>"}`.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

MAX_BODY_SIZE = 1024 * 1024


class MockGuardianHandler(BaseHTTPRequestHandler):
    """Handles verification requests for the mock guardian service."""

    def _send_json(self, status_code: int, payload: dict) -> None:
        """Writes a JSON response.

        Args:
            status_code: HTTP status code to answer with.
            payload: Body of the response.
        """
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 (name imposed by BaseHTTPRequestHandler)
        """Handles a `POST /verify` request."""
        if self.path.rstrip("/") != "/verify":
            self._send_json(404, {"reason": f"unknown endpoint {self.path}"})
            return

        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_BODY_SIZE:
            self._send_json(400, {"reason": "missing or oversized request body"})
            return

        try:
            request = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as e:
            self._send_json(400, {"reason": f"body is not valid JSON: {e}"})
            return

        capabilities = request.get("capabilities") or {}
        expected = capabilities.get("training_plan_checksum")
        received = request.get("training_plan_checksum")

        if not expected:
            valid, reason = False, "capability does not carry a training plan checksum"
        elif expected != received:
            valid, reason = (
                False,
                "training plan checksum does not match the one in the capability",
            )
        else:
            valid, reason = True, "training plan checksum matches the capability"

        print(
            f"[mock-guardian] node={request.get('node_id')} "
            f"experiment={request.get('experiment_id')} "
            f"round={request.get('round')} training={request.get('training')} "
            f"valid={valid} reason={reason}",
            flush=True,
        )
        self._send_json(200, {"valid": valid, "reason": reason})


def main() -> None:
    """Parses command line arguments and runs the mock guardian server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="port to bind to")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MockGuardianHandler)
    print(
        f"[mock-guardian] listening on http://{args.host}:{args.port}/verify "
        "(development stub, do not deploy)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[mock-guardian] shutting down", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

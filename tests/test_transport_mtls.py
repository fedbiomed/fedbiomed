# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Mutual-TLS test suite.

Covers the mutual-TLS feature end to end:

* the certificate/config helpers in ``fedbiomed.common.certificate_manager``
  (``certificate_subject_field``, ``is_mtls_enabled``, ``mtls_db_path``),
* the server side wiring (``SSLCredentials.mtls``, ``_peer_node_id``) and the
  node-identity spoofing enforcement in ``ResearcherServicer.GetTaskUnary``,
* a real gRPC handshake matrix validating certificate pinning, required client
  authentication and the target-name override.
"""

import asyncio
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from cryptography import x509

from fedbiomed.common.certificate_manager import (
    CertificateManager,
    certificate_subject_field,
    is_mtls_enabled,
    mtls_db_path,
)
from fedbiomed.common.constants import CONFIG_FOLDER_NAME, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedCertificateError
from fedbiomed.common.message import SearchRequest
from fedbiomed.transport.node_agent import AgentStore
from fedbiomed.transport.protocols.researcher_pb2 import TaskRequest
from fedbiomed.transport.server import (
    ResearcherServicer,
    SSLCredentials,
    _peer_node_id,
)


def _generate(folder, name, org, cn="localhost"):
    """Generates a self-signed cert, returns (key_file, cert_file, key, cert)."""
    key_file, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=folder,
        certificate_name=name,
        component_id=org,
        subject={"CommonName": cn, "OrganizationName": org},
    )
    with open(key_file, "rb") as key, open(pem_file, "rb") as cert:
        return key_file, pem_file, key.read(), cert.read()


@pytest.fixture(scope="module")
def certs():
    """Generates researcher and node certificates for the whole module."""
    with tempfile.TemporaryDirectory() as tmp:
        researcher_key_file, researcher_cert_file, researcher_key, researcher_cert = (
            _generate(tmp, "researcher", "researcher_1", cn="localhost")
        )
        node_key_file, node_cert_file, node_key, node_cert = _generate(
            tmp, "node", "node_1", cn="*"
        )
        yield {
            "researcher_key_file": researcher_key_file,
            "researcher_cert_file": researcher_cert_file,
            "researcher_key": researcher_key,
            "researcher_cert": researcher_cert,
            "node_key_file": node_key_file,
            "node_cert_file": node_cert_file,
            "node_key": node_key,
            "node_cert": node_cert,
        }


# ---------------------------------------------------------------------------
# certificate_subject_field
# ---------------------------------------------------------------------------


def test_subject_field_reads_organization(certs):
    assert (
        certificate_subject_field(
            certs["node_cert"], x509.oid.NameOID.ORGANIZATION_NAME
        )
        == "node_1"
    )


def test_subject_field_reads_common_name(certs):
    assert (
        certificate_subject_field(certs["node_cert"], x509.oid.NameOID.COMMON_NAME)
        == "*"
    )


def test_subject_field_returns_none_for_unparseable_certificate():
    assert (
        certificate_subject_field(b"not a certificate", x509.oid.NameOID.COMMON_NAME)
        is None
    )


def test_subject_field_returns_none_for_absent_oid(certs):
    # LOCALITY is not set in the generated subject
    assert (
        certificate_subject_field(certs["node_cert"], x509.oid.NameOID.LOCALITY_NAME)
        is None
    )


# ---------------------------------------------------------------------------
# is_mtls_enabled / mtls_db_path (config `[mtls]` section)
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal stand-in mirroring `Config.get`/`getbool`/`root` semantics.

    `values` maps `(section, key)` to the stored string; a missing entry
    behaves like an absent `[mtls]` section (uses `fallback` or raises).
    """

    def __init__(self, root, values=None):
        self.root = root
        self._values = values or {}

    def get(self, section, key, **kwargs):
        if (section, key) in self._values:
            return self._values[(section, key)]
        if "fallback" in kwargs:
            return kwargs["fallback"]
        raise KeyError(f"No option {key} in section {section}")

    def getbool(self, section, key, **kwargs):
        return self.get(section, key, **kwargs).lower() in ("true", "1")


def test_is_mtls_enabled_true_when_flag_set():
    config = _FakeConfig("/root", {("mtls", "enabled"): "True"})
    assert is_mtls_enabled(config) is True


def test_is_mtls_enabled_false_when_flag_unset():
    config = _FakeConfig("/root", {("mtls", "enabled"): "False"})
    assert is_mtls_enabled(config) is False


def test_is_mtls_enabled_false_when_section_absent():
    # No `[mtls]` section at all -> legacy workflow, disabled
    assert is_mtls_enabled(_FakeConfig("/root")) is False


def test_mtls_db_path_resolved_relative_to_etc():
    config = _FakeConfig("/root", {("mtls", "db"): os.path.join("certs", "mtls.json")})
    assert mtls_db_path(config) == os.path.join(
        "/root", CONFIG_FOLDER_NAME, "certs", "mtls.json"
    )


def test_mtls_db_path_raises_when_db_not_configured():
    # mTLS enabled but no `db` entry -> configuration error, no synthesized path
    with pytest.raises(FedbiomedCertificateError):
        mtls_db_path(_FakeConfig("/root"))


# ---------------------------------------------------------------------------
# SSLCredentials.mtls
# ---------------------------------------------------------------------------


def test_ssl_credentials_mtls_disabled_without_bundle(certs):
    ssl = SSLCredentials(
        key=certs["researcher_key_file"], cert=certs["researcher_cert_file"]
    )
    assert ssl.mtls is False
    assert ssl.private_key == certs["researcher_key"]
    assert ssl.certificate == certs["researcher_cert"]


def test_ssl_credentials_mtls_enabled_with_bundle(certs):
    ssl = SSLCredentials(
        key=certs["researcher_key_file"],
        cert=certs["researcher_cert_file"],
        trusted_node_certificates=certs["node_cert"],
    )
    assert ssl.mtls is True
    assert ssl.trusted_node_certificates == certs["node_cert"]


# ---------------------------------------------------------------------------
# _peer_node_id
# ---------------------------------------------------------------------------


def _context_with_cert(cert):
    """Builds a servicer context whose peer presents `cert` (None for no cert)."""
    context = MagicMock()
    auth = {"x509_pem_cert": [cert]} if cert is not None else {}
    context.auth_context.return_value = auth
    return context


def test_peer_node_id_from_bytes_certificate(certs):
    assert _peer_node_id(_context_with_cert(certs["node_cert"])) == "node_1"


def test_peer_node_id_from_str_certificate(certs):
    # Some gRPC builds surface the PEM as str rather than bytes
    context = _context_with_cert(certs["node_cert"].decode("utf-8"))
    assert _peer_node_id(context) == "node_1"


def test_peer_node_id_none_when_no_client_certificate():
    assert _peer_node_id(_context_with_cert(None)) is None


# ---------------------------------------------------------------------------
# ResearcherServicer node-identity enforcement (GetTaskUnary)
# ---------------------------------------------------------------------------


class _Aborted(Exception):
    """Stands in for the exception grpc raises out of ``context.abort``."""


def _servicer_with_agent():
    """Returns (servicer, agent_store, node_agent) ready for GetTaskUnary."""
    node_agent = AsyncMock()
    node_agent.task_done = MagicMock()
    node_agent.get_task.return_value = [
        SearchRequest(researcher_id="r-id", tags=["test"]),
        0,
        time.time(),
    ]
    agent_store = MagicMock(spec=AgentStore)
    agent_store.retrieve.return_value = node_agent
    servicer = ResearcherServicer(agent_store=agent_store, on_message=MagicMock())
    return servicer, agent_store, node_agent


@pytest.mark.asyncio
async def test_get_task_aborts_on_node_id_spoofing(certs):
    """Declared node id not matching the client certificate is rejected."""
    servicer, agent_store, _ = _servicer_with_agent()
    # Certificate identity is `node_1`, but the request declares `node-1`
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)

    request = TaskRequest(node="node-1", protocol_version="x")
    with pytest.raises(_Aborted):
        async for _ in servicer.GetTaskUnary(request=request, context=context):
            pass

    context.abort.assert_awaited_once()
    status, message = context.abort.await_args.args
    assert status == grpc.StatusCode.UNAUTHENTICATED
    assert ErrorNumbers.FB628.value in message
    # The task must never be served to a spoofing peer
    agent_store.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_get_task_proceeds_when_identity_matches(certs):
    """Matching declared node id and certificate identity serves the task."""
    servicer, agent_store, _ = _servicer_with_agent()
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)

    request = TaskRequest(node="node_1", protocol_version="x")
    responses = [
        r async for r in servicer.GetTaskUnary(request=request, context=context)
    ]

    context.abort.assert_not_awaited()
    agent_store.retrieve.assert_called_once_with(node_id="node_1")
    assert len(responses) == 1


@pytest.mark.asyncio
async def test_get_task_proceeds_without_client_certificate():
    """With mutual TLS disabled (no client cert) identity is not enforced."""
    servicer, agent_store, _ = _servicer_with_agent()
    context = _context_with_cert(None)
    context.abort = AsyncMock(side_effect=_Aborted)

    request = TaskRequest(node="node-1", protocol_version="x")
    responses = [
        r async for r in servicer.GetTaskUnary(request=request, context=context)
    ]

    context.abort.assert_not_awaited()
    agent_store.retrieve.assert_called_once_with(node_id="node-1")
    assert len(responses) == 1


def _security_info_calls(info_mock):
    return [
        c
        for c in info_mock.call_args_list
        if c.kwargs.get("extra", {}).get("is_security")
    ]


@pytest.mark.asyncio
async def test_get_task_audits_first_authentication_only(certs):
    """The first authenticated poll logs one audit line; later polls stay quiet."""
    servicer, _, _ = _servicer_with_agent()
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)
    request = TaskRequest(node="node_1", protocol_version="x")

    with patch("fedbiomed.transport.server.logger.info") as info:
        for _ in range(2):
            async for _r in servicer.GetTaskUnary(request=request, context=context):
                pass

    audit = _security_info_calls(info)
    assert len(audit) == 1
    assert "node_1" in audit[0].args[0]


@pytest.mark.asyncio
async def test_get_task_no_audit_without_client_certificate():
    """Server-auth-only connections (no client cert) produce no audit line."""
    servicer, _, _ = _servicer_with_agent()
    context = _context_with_cert(None)
    context.abort = AsyncMock(side_effect=_Aborted)
    request = TaskRequest(node="node-1", protocol_version="x")

    with patch("fedbiomed.transport.server.logger.info") as info:
        async for _r in servicer.GetTaskUnary(request=request, context=context):
            pass

    assert _security_info_calls(info) == []


# ---------------------------------------------------------------------------
# End-to-end TLS handshake matrix
# ---------------------------------------------------------------------------


async def _serve(certs, trusted_node_bundle):
    """Starts a mutual-TLS gRPC server and returns (server, port)."""
    server = grpc.aio.server()
    credentials = grpc.ssl_server_credentials(
        ((certs["researcher_key"], certs["researcher_cert"]),),
        root_certificates=trusted_node_bundle,
        require_client_auth=True,
    )
    port = server.add_secure_port("127.0.0.1:0", credentials)
    await server.start()
    return server, port


async def _can_connect(certs, port, present_client_cert, pinned_server_cert):
    """Attempts a TLS handshake, returns True if the channel becomes ready."""
    credentials = grpc.ssl_channel_credentials(
        root_certificates=pinned_server_cert,
        private_key=certs["node_key"] if present_client_cert else None,
        certificate_chain=certs["node_cert"] if present_client_cert else None,
    )
    override = certificate_subject_field(
        pinned_server_cert, x509.oid.NameOID.COMMON_NAME
    )
    channel = grpc.aio.secure_channel(
        f"127.0.0.1:{port}",
        credentials,
        options=[("grpc.ssl_target_name_override", override)] if override else [],
    )
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=4)
        return True
    except (asyncio.TimeoutError, grpc.aio.AioRpcError):
        return False
    finally:
        await channel.close()


@pytest.mark.asyncio
async def test_registered_node_connects(certs):
    server, port = await _serve(certs, certs["node_cert"])
    try:
        assert await _can_connect(certs, port, True, certs["researcher_cert"])
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_node_without_client_cert_is_rejected(certs):
    server, port = await _serve(certs, certs["node_cert"])
    try:
        assert not await _can_connect(certs, port, False, certs["researcher_cert"])
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_unregistered_node_is_rejected(certs):
    # Bundle contains only the researcher cert, so the node cert is untrusted
    server, port = await _serve(certs, certs["researcher_cert"])
    try:
        assert not await _can_connect(certs, port, True, certs["researcher_cert"])
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_wrong_pinned_server_cert_is_rejected(certs):
    # Node pins the wrong certificate (MITM simulation)
    server, port = await _serve(certs, certs["node_cert"])
    try:
        assert not await _can_connect(certs, port, True, certs["node_cert"])
    finally:
        await server.stop(0)

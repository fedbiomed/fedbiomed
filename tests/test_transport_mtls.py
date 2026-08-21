# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Mutual authentication test suite.

Covers the mutual authentication feature end to end:

* the certificate/config helpers in ``fedbiomed.common.certificate_manager``
  (``certificate_subject_field``, ``TrustedCertificateBundle``, ``is_mtls_enabled``),
* the server side wiring (``SSLCredentials.mtls``, ``_verify_peer_identity``),
  the node-identity spoofing enforcement in every ``ResearcherServicer`` RPC
  carrying a node id, and the audit events recorded for accepted and rejected
  peers,
* a real gRPC handshake matrix validating certificate pinning, required client
  authentication and the target-name override.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import grpc
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from fedbiomed.common.certificate_manager import (
    CertificateManager,
    TrustedCertificateBundle,
    certificate_names,
    certificate_subject_field,
    is_mtls_enabled,
)
from fedbiomed.common.constants import ComponentType, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedCertificateError
from fedbiomed.common.message import SearchReply, SearchRequest
from fedbiomed.common.serializer import Serializer
from fedbiomed.transport.client import (
    Channels,
    ResearcherCredentials,
    _researcher_requires_client_auth,
)
from fedbiomed.transport.node_agent import AgentStore
from fedbiomed.transport.protocols.researcher_pb2 import (
    FeedbackMessage,
    TaskRequest,
    TaskResult,
)
from fedbiomed.transport.server import (
    ResearcherServicer,
    SSLCredentials,
    _GrpcAsyncServer,
    _verify_peer_identity,
)

# Component ids as `Config.generate` builds them. The prefix is what restricts a
# generated certificate to a single TLS role, so the shipped certificates are
# single-role and the handshake matrix has to exercise them as such.
NODE_ID = f"NODE_{uuid4()}"
RESEARCHER_ID = f"RESEARCHER_{uuid4()}"
# A party id other than the one the certificate registered under it carries
OTHER_NODE_ID = f"NODE_{uuid4()}"


def _generate(folder, name, party_id, san=None):
    """Generates a self-signed cert, returns (key_file, cert_file, key, cert).

    Names are given for a researcher only, as the shipped configurations do: they
    are what nodes verify the researcher under, and a node certificate is issued
    for no name at all.
    """
    key_file, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=folder,
        certificate_name=name,
        component_id=party_id,
        san=san,
    )
    with open(key_file, "rb") as key, open(pem_file, "rb") as cert:
        return key_file, pem_file, key.read(), cert.read()


def _naming_only(host):
    """A server certificate naming `host` alone, as PEM (key, certificate).

    Fed-BioMed adds the loopback names to every certificate it issues, so one that
    names none of the addresses a test can dial is necessarily issued elsewhere.
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, RESEARCHER_ID)])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(host)]), critical=False
        )
        .sign(private_key=key, algorithm=hashes.SHA256())
    )
    return (
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ),
        certificate.public_bytes(serialization.Encoding.PEM),
    )


@pytest.fixture(scope="module")
def certs():
    """Generates researcher and node certificates for the whole module."""
    with tempfile.TemporaryDirectory() as tmp:
        researcher_key_file, researcher_cert_file, researcher_key, researcher_cert = (
            _generate(tmp, "researcher", RESEARCHER_ID, san=["localhost"])
        )
        node_key_file, node_cert_file, node_key, node_cert = _generate(
            tmp, "node", NODE_ID
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


def test_subject_field_reads_the_party_id(certs):
    assert (
        certificate_subject_field(certs["node_cert"], x509.oid.NameOID.COMMON_NAME)
        == NODE_ID
    )


def test_subject_carries_the_party_id_and_no_host(certs):
    """The subject identifies the component; hosts and addresses live in the SAN."""
    for certificate, party_id in (
        (certs["researcher_cert"], RESEARCHER_ID),
        (certs["node_cert"], NODE_ID),
    ):
        assert (
            x509.load_pem_x509_certificate(certificate).subject.rfc4514_string()
            == f"CN={party_id}"
        )


def test_researcher_certificate_names_its_host_and_the_loopback_names(certs):
    """What a node verifies the researcher under, whichever of them it dials."""
    assert certificate_names(certs["researcher_cert"]) == ["localhost", "127.0.0.1"]


def test_node_certificate_carries_no_name(certs):
    """A node is resolved by fingerprint, so its certificate is issued for no name."""
    assert certificate_names(certs["node_cert"]) == []


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
# is_mtls_enabled (config `[mtls]` section)
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
    # No `[mtls]` section at all -> disabled
    assert is_mtls_enabled(_FakeConfig("/root")) is False


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
        trusted_node_certificates=lambda: certs["node_cert"],
    )
    assert ssl.mtls is True
    assert ssl.trusted_node_certificates() == certs["node_cert"]


# ---------------------------------------------------------------------------
# Peer identity resolution against the certificate registry
# ---------------------------------------------------------------------------


def _context_with_cert(cert, peer="ipv4:127.0.0.1:51234"):
    """Builds a servicer context whose peer presents `cert` (None for no cert)."""
    context = MagicMock()
    auth = {"x509_pem_cert": [cert]} if cert is not None else {}
    context.auth_context.return_value = auth
    context.peer.return_value = peer
    return context


class _Aborted(Exception):
    """Stands in for the exception grpc raises out of ``context.abort``."""


def _events(event_mock, operation):
    """Audit events of one operation recorded by a patched `logger.security_event`."""
    return [
        c for c in event_mock.call_args_list if c.kwargs.get("operation") == operation
    ]


def _registry(path, entries):
    """Writes a certificate registry holding `(party_id, certificate)` entries."""
    manager = CertificateManager(db_path=str(path))
    try:
        for party_id, certificate in entries:
            manager.insert(
                certificate=certificate.decode("utf-8"),
                party_id=party_id,
                component=ComponentType.NODE.name,
            )
    finally:
        manager.close()
    return TrustedCertificateBundle(str(path), ComponentType.NODE.name)


@pytest.fixture
def registry(certs, tmp_path):
    """Registry holding the node certificate under the node's party id."""
    return _registry(tmp_path / "registry.json", [(NODE_ID, certs["node_cert"])])


def test_party_id_resolves_registered_certificate(certs, registry):
    assert registry.party_id(certs["node_cert"]) == NODE_ID
    # Re-encoded PEM (extra whitespace) still resolves: matching is on content
    assert registry.party_id(b"\n" + certs["node_cert"]) == NODE_ID


def test_party_id_returns_none_for_unregistered_certificate(certs, registry):
    # Registered for RESEARCHER, not the NODE bundle this view covers
    assert registry.party_id(certs["researcher_cert"]) is None
    assert registry.party_id(b"not a certificate") is None


def test_party_id_refuses_a_certificate_registered_under_two_parties(certs, tmp_path):
    """One certificate under two party ids authenticates neither of them.

    Registration refuses to create this, so it only arises in a registry written
    before that check or edited by hand.
    """
    ambiguous = _registry(
        tmp_path / "ambiguous.json",
        [(OTHER_NODE_ID, certs["node_cert"]), (NODE_ID, certs["node_cert"])],
    )

    with patch("fedbiomed.common.certificate_manager.logger.security_event") as event:
        assert ambiguous.party_id(certs["node_cert"]) is None

    # The report names every claimant, so the operator knows what to delete
    fields = _events(event, "certificate_ambiguous_identity")[0].kwargs
    assert fields["party_ids"] == sorted([NODE_ID, OTHER_NODE_ID])


def test_party_id_reads_the_registry_once_across_calls(certs, registry):
    """Resolution is cached: an unchanged registry is read only on first use."""
    with patch(
        "fedbiomed.common.certificate_manager.CertificateManager.list",
        side_effect=CertificateManager.list,
        autospec=True,
    ) as read:
        for _ in range(50):
            assert registry.party_id(certs["node_cert"]) == NODE_ID
        # ... and the PEM bundle is served from the same single read
        registry()

    assert read.call_count == 1


def test_party_id_picks_up_a_registration_without_restart(certs, tmp_path, registry):
    """A certificate registered after first use resolves on the next call."""
    assert registry.party_id(certs["researcher_cert"]) is None

    manager = CertificateManager(db_path=registry._db_path)
    try:
        manager.insert(
            certificate=certs["researcher_cert"].decode("utf-8"),
            party_id=OTHER_NODE_ID,
            component=ComponentType.NODE.name,
        )
    finally:
        manager.close()

    assert registry.party_id(certs["researcher_cert"]) == OTHER_NODE_ID


@pytest.mark.asyncio
async def test_verify_peer_identity_skips_without_client_certificate(registry):
    """Server-auth only: there is no peer identity to bind the declared id to."""
    context = _context_with_cert(None)
    assert await _verify_peer_identity(context, "anything", registry) is None


@pytest.mark.asyncio
async def test_verify_peer_identity_prefers_registry_over_certificate_subject(
    certs, tmp_path
):
    """The registered party id is authoritative, not the certificate `CN=` field.

    Registering a certificate under an explicit party id is supported for
    certificates that embed no Fed-BioMed identity, so the identity that counts
    is the one in the registry.
    """
    aliased = _registry(
        tmp_path / "aliased.json", [(OTHER_NODE_ID, certs["node_cert"])]
    )
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)

    assert await _verify_peer_identity(context, OTHER_NODE_ID, aliased) == OTHER_NODE_ID

    # The certificate's CN= value is not what it is registered as
    with pytest.raises(_Aborted):
        await _verify_peer_identity(context, NODE_ID, aliased)


async def _refusal(context, identities):
    """Refuses the peer and returns the recorded `mtls_identity_unresolved` event."""
    with patch("fedbiomed.transport.server.logger.security_event") as event:
        with pytest.raises(_Aborted):
            await _verify_peer_identity(context, NODE_ID, identities)

    unresolved = _events(event, "mtls_identity_unresolved")
    assert len(unresolved) == 1
    return unresolved[0].kwargs


@pytest.mark.asyncio
async def test_refusal_distinguishes_an_unreadable_registry(certs, tmp_path):
    """An unreadable registry is not reported as an unregistered certificate."""
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)
    unreadable = tmp_path / "corrupt.json"
    unreadable.write_text("{ not json", encoding="utf-8")

    fields = await _refusal(
        context, TrustedCertificateBundle(str(unreadable), ComponentType.NODE.name)
    )

    assert fields["reason"] == "registry_unreadable"
    assert "registry could not be read" in fields["detail"]


@pytest.mark.asyncio
async def test_refusal_distinguishes_an_unregistered_certificate(certs, tmp_path):
    """A readable registry the certificate is absent from says exactly that."""
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)

    fields = await _refusal(context, _registry(tmp_path / "empty.json", []))

    assert fields["reason"] == "certificate_not_registered"
    assert "not registered" in fields["detail"]


@pytest.mark.asyncio
async def test_refusal_distinguishes_a_missing_registry(certs):
    """A client certificate with no registry configured is its own failure."""
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)

    fields = await _refusal(context, None)

    assert fields["reason"] == "no_registry_configured"
    assert "no node certificate registry is configured" in fields["detail"]


def test_loaded_reports_whether_the_registry_was_ever_read(certs, tmp_path, registry):
    unreadable = tmp_path / "corrupt.json"
    unreadable.write_text("{ not json", encoding="utf-8")
    broken = TrustedCertificateBundle(str(unreadable), ComponentType.NODE.name)

    assert broken.party_id(certs["node_cert"]) is None
    assert broken.loaded is False

    assert registry.party_id(certs["node_cert"]) == NODE_ID
    assert registry.loaded is True


def test_party_id_keeps_last_read_when_registry_becomes_unreadable(certs, registry):
    """A partially written registry does not drop identities already resolved.

    TinyDB rewrites the file non-atomically, so a read landing mid-write is
    transient; refusing on it would reject healthy nodes at random.
    """
    assert registry.party_id(certs["node_cert"]) == NODE_ID

    with open(registry._db_path, "w", encoding="utf-8") as f:
        f.write("{ partially writ")

    with patch("fedbiomed.common.certificate_manager.logger.security_event") as event:
        assert registry.party_id(certs["node_cert"]) == NODE_ID

    assert len(_events(event, "certificate_store_unreadable")) == 1


# ---------------------------------------------------------------------------
# ResearcherServicer node-identity enforcement (GetTaskUnary)
# ---------------------------------------------------------------------------


def _servicer_with_agent(registry):
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
    servicer = ResearcherServicer(
        agent_store=agent_store, on_message=MagicMock(), identities=registry
    )
    return servicer, agent_store, node_agent


def _feedback_servicer(registry):
    """Returns (servicer, on_message); Feedback never touches the agent store."""
    on_message = MagicMock()
    servicer = ResearcherServicer(
        agent_store=MagicMock(spec=AgentStore),
        on_message=on_message,
        identities=registry,
    )
    return servicer, on_message


@pytest.mark.asyncio
async def test_get_task_aborts_on_node_id_spoofing(certs, registry):
    """Declared node id not matching the registered identity is rejected."""
    servicer, agent_store, _ = _servicer_with_agent(registry)
    # The certificate resolves to the registered node id, the request declares
    # another one
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
async def test_node_id_spoofing_is_registered_as_event(certs, registry):
    """The spoofing rejection is registered as a named audit event."""
    servicer, _, _ = _servicer_with_agent(registry)
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)
    request = TaskRequest(node="node-1", protocol_version="x")

    with patch("fedbiomed.transport.server.logger.security_event") as event:
        with pytest.raises(_Aborted):
            async for _ in servicer.GetTaskUnary(request=request, context=context):
                pass

    audit = _events(event, "mtls_identity_mismatch")
    assert len(audit) == 1
    assert audit[0].kwargs["status"] == "failure"
    assert audit[0].kwargs["node_id"] == NODE_ID
    assert audit[0].kwargs["declared_node_id"] == "node-1"
    # A rejection identifies where it came from and which certificate was used
    assert audit[0].kwargs["source_address"] == "ipv4:127.0.0.1:51234"
    assert audit[0].kwargs["cert_subject"] == f"CN={NODE_ID}"


async def _poll(servicer, certs, peers):
    """Issues one GetTaskUnary per peer address, as the registered node."""
    request = TaskRequest(node=NODE_ID, protocol_version="x")
    for peer in peers:
        context = _context_with_cert(certs["node_cert"], peer=peer)
        async for _ in servicer.GetTaskUnary(request=request, context=context):
            pass


@pytest.mark.asyncio
async def test_authenticated_node_event_identifies_certificate(certs, registry):
    """A successful handshake records the certificate and the peer address."""
    servicer, _, _ = _servicer_with_agent(registry)

    with patch("fedbiomed.transport.server.logger.security_event") as event:
        await _poll(servicer, certs, ("ipv4:127.0.0.1:51234",))

    fields = _events(event, "mtls_node_authenticated")[0].kwargs
    assert fields["status"] == "success"
    assert fields["node_id"] == NODE_ID
    assert fields["source_address"] == "ipv4:127.0.0.1:51234"
    assert fields["destination_service"] == "researcher.ResearcherService"
    assert fields["cert_subject"] == f"CN={NODE_ID}"
    assert {"cert_issuer", "cert_serial", "cert_not_after"} <= fields.keys()


@pytest.mark.asyncio
async def test_authenticated_node_event_repeats_on_new_origin(certs, registry):
    """Reconnecting is audited again, keeping only the node's current identity.

    gRPC reports the peer's ephemeral source port, so every reconnection yields
    a new address even from the same host. Returning to an address used before
    is one such reconnection, so it is audited rather than treated as already
    seen; the bookkeeping stays at one entry per node throughout.
    """
    servicer, _, _ = _servicer_with_agent(registry)
    addresses = ("ipv4:127.0.0.1:1", "ipv4:127.0.0.1:2", "ipv4:127.0.0.1:1")

    with patch("fedbiomed.transport.server.logger.security_event") as event:
        await _poll(servicer, certs, addresses)

    audit = _events(event, "mtls_node_authenticated")
    assert [c.kwargs["source_address"] for c in audit] == list(addresses)
    assert list(servicer._peer_identity) == [NODE_ID]


@pytest.mark.asyncio
async def test_get_task_proceeds_when_identity_matches(certs, registry):
    """Matching declared node id and registered identity serves the task."""
    servicer, agent_store, _ = _servicer_with_agent(registry)
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)

    request = TaskRequest(node=NODE_ID, protocol_version="x")
    responses = [
        r async for r in servicer.GetTaskUnary(request=request, context=context)
    ]

    context.abort.assert_not_awaited()
    agent_store.retrieve.assert_called_once_with(node_id=NODE_ID)
    assert len(responses) == 1


@pytest.mark.asyncio
async def test_get_task_proceeds_without_client_certificate(registry):
    """With mutual authentication disabled (no client cert) identity is not enforced."""
    servicer, agent_store, _ = _servicer_with_agent(registry)
    context = _context_with_cert(None)
    context.abort = AsyncMock(side_effect=_Aborted)

    request = TaskRequest(node="node-1", protocol_version="x")
    responses = [
        r async for r in servicer.GetTaskUnary(request=request, context=context)
    ]

    context.abort.assert_not_awaited()
    agent_store.retrieve.assert_called_once_with(node_id="node-1")
    assert len(responses) == 1


@pytest.mark.asyncio
async def test_get_task_audits_first_authentication_only(certs, registry):
    """The first authenticated poll logs one audit event; later polls stay quiet."""
    servicer, _, _ = _servicer_with_agent(registry)
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)
    request = TaskRequest(node=NODE_ID, protocol_version="x")

    with patch("fedbiomed.transport.server.logger.security_event") as event:
        for _ in range(2):
            async for _r in servicer.GetTaskUnary(request=request, context=context):
                pass

    audit = _events(event, "mtls_node_authenticated")
    assert len(audit) == 1
    assert audit[0].kwargs["status"] == "success"
    assert audit[0].kwargs["node_id"] == NODE_ID


@pytest.mark.asyncio
async def test_get_task_no_audit_without_client_certificate(registry):
    """Server-auth-only connections (no client cert) produce no audit event."""
    servicer, _, _ = _servicer_with_agent(registry)
    context = _context_with_cert(None)
    context.abort = AsyncMock(side_effect=_Aborted)
    request = TaskRequest(node="node-1", protocol_version="x")

    with patch("fedbiomed.transport.server.logger.security_event") as event:
        async for _r in servicer.GetTaskUnary(request=request, context=context):
            pass

    assert _events(event, "mtls_node_authenticated") == []


# ---------------------------------------------------------------------------
# ResearcherServicer node-identity enforcement (ReplyTask, Feedback)
# ---------------------------------------------------------------------------


async def _reply_stream(node_id):
    """Single-chunk ReplyTask stream carrying a reply declared by `node_id`."""
    payload = Serializer.dumps(
        SearchReply(
            researcher_id="r-id",
            node_id=node_id,
            node_name="node-name",
            databases=[],
            count=0,
        ).to_dict()
    )
    yield TaskResult(size=1, iteration=1, bytes_=payload)


@pytest.mark.asyncio
async def test_reply_task_aborts_on_node_id_spoofing(certs, registry):
    """A reply declaring another node's id is refused, not handed to its agent."""
    servicer, agent_store, _ = _servicer_with_agent(registry)
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)

    with pytest.raises(_Aborted):
        await servicer.ReplyTask(
            request_iterator=_reply_stream("node-2"), context=context
        )

    status, message = context.abort.await_args.args
    assert status == grpc.StatusCode.UNAUTHENTICATED
    assert ErrorNumbers.FB628.value in message
    agent_store.get.assert_not_called()


@pytest.mark.asyncio
async def test_reply_task_proceeds_when_identity_matches(certs, registry):
    """A reply declaring the peer's own registered id reaches its agent."""
    servicer, agent_store, _ = _servicer_with_agent(registry)
    node_agent = AsyncMock()
    agent_store.get.return_value = node_agent
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)

    await servicer.ReplyTask(request_iterator=_reply_stream(NODE_ID), context=context)

    context.abort.assert_not_awaited()
    node_agent.on_reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_feedback_aborts_on_node_id_spoofing(certs, registry):
    """Feedback attributed to another node is refused before dispatch."""
    servicer, on_message = _feedback_servicer(registry)
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)
    request = FeedbackMessage(
        researcher_id="r-id",
        log=FeedbackMessage.Log(node_id="node-2", level="DEBUG", msg="spoofed"),
    )

    with pytest.raises(_Aborted):
        await servicer.Feedback(request=request, context=context)

    status, _message = context.abort.await_args.args
    assert status == grpc.StatusCode.UNAUTHENTICATED
    on_message.assert_not_called()


@pytest.mark.asyncio
async def test_feedback_proceeds_when_identity_matches(certs, registry):
    """Feedback attributed to the peer's own registered id is dispatched."""
    servicer, on_message = _feedback_servicer(registry)
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)
    request = FeedbackMessage(
        researcher_id="r-id",
        log=FeedbackMessage.Log(node_id=NODE_ID, level="DEBUG", msg="genuine"),
    )

    await servicer.Feedback(request=request, context=context)

    context.abort.assert_not_awaited()
    on_message.assert_called_once()


@pytest.mark.asyncio
async def test_unregistered_certificate_is_refused_on_every_rpc(certs, tmp_path):
    """A certificate trusted at handshake but absent from the registry is refused.

    Covers the window after `certificate delete`, where the running trust bundle
    and the registry disagree.
    """
    empty = _registry(tmp_path / "empty.json", [])
    servicer, _, _ = _servicer_with_agent(empty)
    context = _context_with_cert(certs["node_cert"])
    context.abort = AsyncMock(side_effect=_Aborted)

    with patch("fedbiomed.transport.server.logger.security_event") as event:
        with pytest.raises(_Aborted):
            async for _ in servicer.GetTaskUnary(
                request=TaskRequest(node=NODE_ID, protocol_version="x"),
                context=context,
            ):
                pass
        with pytest.raises(_Aborted):
            await servicer.ReplyTask(
                request_iterator=_reply_stream(NODE_ID), context=context
            )

    assert len(_events(event, "mtls_identity_unresolved")) == 2


# ---------------------------------------------------------------------------
# End-to-end TLS handshake matrix
# ---------------------------------------------------------------------------


def _credentials(certs, trusted_node_bundle):
    """Builds server credentials through the shipped `_GrpcAsyncServer` path.

    `trusted_node_bundle` is a zero-argument callable returning the current
    bundle, or None for a server-auth-only server.
    """
    ssl = SSLCredentials(
        key=certs["researcher_key_file"],
        cert=certs["researcher_cert_file"],
        trusted_node_certificates=trusted_node_bundle,
    )
    server = _GrpcAsyncServer(
        host="127.0.0.1",
        port="0",
        on_message=MagicMock(),
        config=MagicMock(),
        ssl=ssl,
    )
    return server._server_credentials()


async def _serve(certs, trusted_node_bundle):
    """Starts a mutually authenticated gRPC server and returns (server, port).

    Credentials come from the shipped code path, so the handshake matrix
    exercises the dynamic, per-handshake trust bundle rather than a hand-rolled
    static one.
    """
    server = grpc.aio.server()
    port = server.add_secure_port(
        "127.0.0.1:0", _credentials(certs, trusted_node_bundle)
    )
    await server.start()
    return server, port


async def _can_connect(certs, port, present_client_cert, pinned_server_cert):
    """Attempts a TLS handshake, returns True if the channel becomes ready."""
    credentials = grpc.ssl_channel_credentials(
        root_certificates=pinned_server_cert,
        private_key=certs["node_key"] if present_client_cert else None,
        certificate_chain=certs["node_cert"] if present_client_cert else None,
    )
    # As `Channels._create` picks it: none where the certificate names the address.
    names = certificate_names(pinned_server_cert)
    override = None if "127.0.0.1" in names else next(iter(names), None)
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
    server, port = await _serve(certs, lambda: certs["node_cert"])
    try:
        assert await _can_connect(certs, port, True, certs["researcher_cert"])
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_node_without_client_cert_is_rejected(certs):
    server, port = await _serve(certs, lambda: certs["node_cert"])
    try:
        assert not await _can_connect(certs, port, False, certs["researcher_cert"])
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_unregistered_node_is_rejected(certs):
    # Bundle contains only the researcher cert, so the node cert is untrusted
    server, port = await _serve(certs, lambda: certs["researcher_cert"])
    try:
        assert not await _can_connect(certs, port, True, certs["researcher_cert"])
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_wrong_pinned_server_cert_is_rejected(certs):
    # Node pins the wrong certificate (MITM simulation)
    server, port = await _serve(certs, lambda: certs["node_cert"])
    try:
        assert not await _can_connect(certs, port, True, certs["node_cert"])
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_node_registered_after_startup_connects_without_restart(certs):
    """A certificate registered mid-session is trusted on the next handshake."""
    bundle = {"pem": certs["researcher_cert"]}
    server, port = await _serve(certs, lambda: bundle["pem"])
    try:
        assert not await _can_connect(certs, port, True, certs["researcher_cert"])

        bundle["pem"] = certs["researcher_cert"] + b"\n" + certs["node_cert"]

        assert await _can_connect(certs, port, True, certs["researcher_cert"])
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_node_revoked_after_startup_is_rejected_without_restart(certs):
    """Dropping a certificate from the bundle stops trusting it."""
    bundle = {"pem": certs["node_cert"]}
    server, port = await _serve(certs, lambda: bundle["pem"])
    try:
        assert await _can_connect(certs, port, True, certs["researcher_cert"])

        bundle["pem"] = certs["researcher_cert"]

        assert not await _can_connect(certs, port, True, certs["researcher_cert"])
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_channel_verifies_the_dialled_address_when_the_certificate_names_it(
    certs,
):
    """A researcher certificate names the loopback addresses, so a node on the same
    machine has its dialled address verified rather than overridden. Server-auth
    only, since that is the path where the dialled address is verified.
    """
    server, port = await _serve(certs, None)
    channel = Channels(
        ResearcherCredentials(
            host="127.0.0.1", port=str(port), certificate=certs["researcher_cert"]
        )
    )._create()
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=4)
    finally:
        await channel.close()
        await server.stop(0)


@pytest.mark.asyncio
async def test_channel_connects_on_an_address_the_certificate_omits():
    """A certificate naming none of the addresses a node dials is still verified,
    under the name it does carry: a certificate issued outside Fed-BioMed names
    the hosts its issuer knew, which need not be how nodes reach the researcher.
    """
    key, cert = _naming_only("fbm-researcher")
    server = grpc.aio.server()
    port = server.add_secure_port(
        "127.0.0.1:0", grpc.ssl_server_credentials([(key, cert)])
    )
    await server.start()

    # Certificate names `fbm-researcher` alone, node dials 127.0.0.1
    channel = Channels(
        ResearcherCredentials(host="127.0.0.1", port=str(port), certificate=cert)
    )._create()
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=4)
    finally:
        await channel.close()
        await server.stop(0)


def test_empty_trust_bundle_is_reported_before_binding(certs):
    """An empty bundle fails with the cause, not an opaque port-binding error."""
    with pytest.raises(FedbiomedCertificateError, match="no node certificate"):
        _credentials(certs, lambda: b"")


def _in_fresh_interpreter(code, preset=None):
    """Runs `code` in a new interpreter, with GRPC_VERBOSITY unset or preset."""
    environment = {k: v for k, v in os.environ.items() if k != "GRPC_VERBOSITY"}
    if preset is not None:
        environment["GRPC_VERBOSITY"] = preset

    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=True,
    ).stdout.strip()


def test_grpc_verbosity_default_is_set_before_grpc_is_imported():
    """The default must be in place when the gRPC core reads it, at `import grpc`.

    Importing the serializer loads grpc through declearn, so setting the default
    anywhere below the top level package is too late. Run in a fresh interpreter
    because the core reads the variable only once.
    """
    code = (
        "import os, sys, fedbiomed.common.serializer;"
        "print(os.environ['GRPC_VERBOSITY'], 'grpc' in sys.modules)"
    )

    assert _in_fresh_interpreter(code) == "ERROR True"


def test_grpc_verbosity_preset_by_an_operator_wins():
    """The default is lowered noise, not a ceiling on what an operator can see.

    A rejected handshake is reported by gRPC itself and nowhere else, so raising
    the variable is the sole way to observe rejections on the researcher.
    """
    code = "import os, fedbiomed; print(os.environ['GRPC_VERBOSITY'])"

    assert _in_fresh_interpreter(code, "INFO") == "INFO"


async def _probe(port):
    """Runs the blocking client-auth probe off the server's event loop."""
    return await asyncio.get_running_loop().run_in_executor(
        None, _researcher_requires_client_auth, "127.0.0.1", str(port)
    )


@pytest.mark.asyncio
async def test_probe_detects_enforced_client_auth(certs):
    """A researcher requiring client certificates is reported as enforcing.

    Under TLS 1.3 the anonymous handshake itself completes, so this only holds
    because the probe reads the server's reply.
    """
    server, port = await _serve(certs, lambda: certs["node_cert"])
    try:
        assert await _probe(port) is True
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_probe_detects_server_auth_only(certs):
    """A server-auth-only researcher is reported as not enforcing."""
    server, port = await _serve(certs, None)
    try:
        assert await _probe(port) is False
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_probe_reports_unknown_when_server_unreachable(certs):
    """An unreachable server yields no verdict, in either direction.

    It must not read as "not enforced", which would wrongly reassure a node that
    its identity goes unchecked, nor as "enforced", which previously let an
    unreachable researcher look like a mutual authentication configuration mismatch.
    """
    server, port = await _serve(certs, lambda: certs["node_cert"])
    await server.stop(0)
    assert await _probe(port) is None

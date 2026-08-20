# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Certificates and researcher connection state of the node the GUI serves."""

import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from flask import request

from fedbiomed.common.certificate_manager import (
    CERTIFICATE_EXPIRY_WARNING_DAYS,
    CertificateManager,
    certificate_audit_fields,
    certificate_expiry,
    certificate_fingerprint,
    certificate_names,
    is_mtls_enabled,
)
from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.node.node_pm import NodeConnectionStateManager

from ..config import config
from ..helpers.auth_helpers import admin_required
from ..utils import error, response
from .api import api
from .node_management import node_process_manager


def _certificate_manager() -> CertificateManager:
    """Open the node's certificate registry. The caller closes it."""
    return CertificateManager(db_path=config.node_config.getpath("default", "db"))


def _certificate_summary(certificate: str) -> Dict[str, Any]:
    """Describe a certificate for display, without exposing the certificate itself.

    Uses the audit fields the node records elsewhere, so both read the same way.
    """
    expiry = certificate_expiry(certificate)
    fingerprint = certificate_fingerprint(certificate)

    summary: Dict[str, Any] = {
        **certificate_audit_fields(certificate),
        "san": certificate_names(certificate),
        "fingerprint": fingerprint.hex() if fingerprint else None,
        "expires_in_days": None,
        "expiring_soon": False,
    }

    if expiry:
        days = (expiry - datetime.now(timezone.utc)).days
        summary["expires_in_days"] = days
        summary["expiring_soon"] = days <= CERTIFICATE_EXPIRY_WARNING_DAYS

    return summary


def _registered_certificates() -> List[Dict[str, Any]]:
    """Summaries of the certificates this node has registered."""
    certificate_manager = _certificate_manager()
    try:
        return [
            {
                "party_id": document["party_id"],
                "component": document["component"],
                **_certificate_summary(document["certificate"]),
            }
            for document in certificate_manager.list()
        ]
    finally:
        certificate_manager.close()


def _own_certificate() -> Dict[str, Any]:
    """Summary of this node's own certificate, with the error when unreadable."""
    path = config.node_config.getpath("certificate", "public_key")
    try:
        with open(path) as file:
            certificate = file.read()
    except OSError as exp:
        return {"path": path, "error": f"Could not read the node certificate: {exp}"}

    return {
        "party_id": config.node_config.get("default", "id"),
        "path": path,
        **_certificate_summary(certificate),
    }


def _startup_check(registered: List[Dict[str, Any]], mtls_enabled: bool) -> List[str]:
    """Reasons the node would refuse to start, in the state it is configured in.

    Reproduces what `Node` verifies when mutual TLS is on, so the GUI can report
    it before a start attempt rather than after it.
    """
    if not mtls_enabled:
        return []

    problems = []
    researchers = [
        certificate
        for certificate in registered
        if certificate["component"] == ComponentType.RESEARCHER.name
    ]
    if not researchers:
        problems.append(
            "Mutual TLS is enabled but no researcher certificate is registered, "
            "so the node cannot start. Register the researcher certificate."
        )
    elif len(researchers) > 1:
        problems.append(
            f"Mutual TLS is enabled and {len(researchers)} researcher certificates "
            "are registered, so the certificate to pin is ambiguous and the node "
            "cannot start. Delete all but the researcher this node connects to."
        )

    private_key = config.node_config.getpath("certificate", "private_key")
    if not os.path.isfile(private_key):
        problems.append(
            f"The node private key is missing from {private_key}, so the node "
            "cannot present its identity under mutual TLS."
        )

    return problems


def _registry_warnings(registered: List[Dict[str, Any]]) -> List[str]:
    """Registrations that break the rules a node registry must satisfy.

    Registration refuses these, so an entry breaking one predates the checks or
    was written by hand.
    """
    warnings = []
    own_kind = [
        certificate["party_id"]
        for certificate in registered
        if certificate["component"] == ComponentType.NODE.name
    ]
    if own_kind:
        warnings.append(
            f"Certificate(s) of this component's own type are registered: "
            f"{', '.join(own_kind)}. Parties register each other's certificates, "
            "never their own type."
        )

    if len(registered) > 1:
        warnings.append(
            "A node registers at most one certificate - its researcher's - but "
            f"{len(registered)} are registered. Delete the extra entries."
        )

    expiring = [
        certificate["party_id"]
        for certificate in registered
        if certificate["expiring_soon"]
    ]
    if expiring:
        warnings.append(
            f"Certificate(s) expiring within {CERTIFICATE_EXPIRY_WARNING_DAYS} days: "
            f"{', '.join(expiring)}. Ask the party to renew and register the new one."
        )

    researcher_host = config.node_config.get("researcher", "ip")
    for certificate in registered:
        if certificate["component"] != ComponentType.RESEARCHER.name:
            continue
        names = certificate["san"]
        if names and researcher_host not in names:
            warnings.append(
                f"The researcher certificate is issued for {', '.join(names)}, which "
                f"does not include the configured researcher host {researcher_host}. "
                "The node verifies it under the first name it carries."
            )

    return warnings


def _restart_required() -> bool:
    """Whether the node runs, so that a change here takes effect only on restart."""
    return node_process_manager.get_status().value == "running"


def _status() -> Dict[str, Any]:
    """The node's mutual-TLS posture: its certificate and what it expects."""
    mtls_enabled = is_mtls_enabled(config.node_config)
    registered = _registered_certificates()

    return {
        "mtls_enabled": mtls_enabled,
        "node_id": config.node_config.get("default", "id"),
        "researcher": {
            "host": config.node_config.get("researcher", "ip"),
            "port": config.node_config.get("researcher", "port"),
        },
        "certificate": _own_certificate(),
        "registered": registered,
        "startup_problems": _startup_check(registered, mtls_enabled),
        "warnings": _registry_warnings(registered),
        "node_state": node_process_manager.get_status().value,
    }


def _register(certificate: str, party_id: Optional[str], upsert: bool) -> str:
    """Register a certificate, returning the party it was registered for."""
    certificate_manager = _certificate_manager()
    # `register_certificate` reads the certificate from a file, as the CLI hands it one
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "certificate.pem")
        try:
            with open(path, "w") as file:
                file.write(certificate)

            return certificate_manager.register_certificate(
                certificate_path=path,
                party_id=party_id,
                upsert=upsert,
                registering_component=ComponentType.NODE.name,
            )
        finally:
            certificate_manager.close()


@api.route("/certificates/status", methods=["GET"])
@admin_required
def certificates_status():
    """Return the node's mutual-TLS posture: its certificate and what it expects."""
    try:
        return response(_status()), 200
    except FedbiomedError as exp:
        return error(f"Could not read the certificate status: {exp}"), 500


@api.route("/certificates", methods=["GET"])
@admin_required
def list_certificates():
    """Return the certificates the node has registered, without their contents."""
    try:
        return response({"certificates": _registered_certificates()}), 200
    except FedbiomedError as exp:
        return error(f"Could not list registered certificates: {exp}"), 500


@api.route("/certificates", methods=["POST"])
@admin_required
def register_certificate():
    """Register a certificate the node received from another party.

    The certificate is sent as text, whether the user pasted it or picked a file.
    `upsert` replaces an existing registration of the same party, which the user
    confirms after the conflict is reported.
    """
    payload = request.get_json(silent=True) or {}
    certificate = payload.get("certificate")
    if not isinstance(certificate, str) or not certificate.strip():
        return error("A certificate in PEM format is required"), 400

    party_id = payload.get("party_id") or None
    if party_id is not None and not isinstance(party_id, str):
        return error("'party_id' must be a string"), 400

    try:
        registered_party_id = _register(
            certificate, party_id, bool(payload.get("upsert", False))
        )
    except FedbiomedError as exp:
        return error(str(exp)), 400
    except OSError as exp:
        return error(f"Could not read the certificate: {exp}"), 500

    return response(
        {
            "party_id": registered_party_id,
            "requires_restart": _restart_required(),
        },
        f"Certificate registered for party {registered_party_id}.",
    ), 200


@api.route("/certificates/<party_id>", methods=["DELETE"])
@admin_required
def delete_certificate(party_id: str):
    """Remove a party's certificate from the node's registry."""
    certificate_manager = _certificate_manager()
    try:
        if not certificate_manager.get(party_id=party_id):
            return error(f"No certificate is registered for party {party_id}"), 404

        certificate_manager.delete(party_id=party_id)
    except FedbiomedError as exp:
        return error(f"Could not delete the certificate: {exp}"), 400
    finally:
        certificate_manager.close()

    return response(
        {"party_id": party_id, "requires_restart": _restart_required()},
        f"Certificate of party {party_id} has been deleted.",
    ), 200


@api.route("/certificates/export", methods=["GET"])
@admin_required
def export_certificate():
    """Return this node's certificate, to be shared with the other parties.

    The public certificate only; the private key never leaves the node.
    """
    path = config.node_config.getpath("certificate", "public_key")
    try:
        with open(path) as file:
            certificate = file.read()
    except OSError as exp:
        return error(f"Could not read the node certificate: {exp}"), 500

    return response(
        {
            "party_id": config.node_config.get("default", "id"),
            "filename": os.path.basename(path),
            "certificate": certificate,
        }
    ), 200


@api.route("/certificates/connection", methods=["GET"])
@admin_required
def connection_state():
    """Return the connection state the node recorded, and its recent history.

    The node writes this as it observes its channel; a state recorded while the
    node was running is reported stale once it is not.
    """
    try:
        manager = NodeConnectionStateManager(config.node_config)
        current = manager.get_connection_state()
        history = manager.get_connection_history()
    except FedbiomedError as exp:
        return error(f"Could not read the connection state: {exp}"), 500

    node_state = node_process_manager.get_status().value

    return response(
        {
            "state": current.to_dict() if current else None,
            "history": [entry.to_dict() for entry in history],
            "node_state": node_state,
            # The node writes nothing while stopped, so its last state is only
            # what was true when it stopped.
            "stale": current is not None and node_state != "running",
        }
    ), 200

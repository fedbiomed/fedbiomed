# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Certificates and researcher connection state of the node the GUI serves."""

import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from cryptography import x509
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from flask import request

from fedbiomed.common.certificate_manager import (
    CERTIFICATE_EXPIRY_WARNING_DAYS,
    CertificateManager,
    certificate_audit_fields,
    certificate_expiry,
    certificate_fingerprint,
    certificate_san_names,
)
from fedbiomed.common.cli import COMPONENT_PURPOSE
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
        "san": certificate_san_names(certificate),
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
                "component_id": document["component_id"],
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
        "component_id": config.node_config.get("default", "id"),
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
    # A node registers at most one certificate - its researcher's - so what is
    # registered says on its own whether the node can pin one.
    if not registered:
        problems.append(
            "Mutual TLS is enabled but no researcher certificate is registered, "
            "so the node cannot start. Register the researcher certificate."
        )
    elif len(registered) > 1:
        problems.append(
            f"Mutual TLS is enabled and {len(registered)} certificates are "
            "registered, so the certificate to pin is ambiguous and the node "
            "cannot start. Delete all but the researcher this node connects to."
        )

    # The node verifies the researcher under a name read from its certificate, and
    # refuses to start on one that states none.
    for certificate in registered:
        if not certificate["san"]:
            problems.append(
                f"The certificate registered for {certificate['component_id']} states "
                "no host, so the node cannot verify the researcher and will not "
                "start. Request the researcher to reissue it for the hosts nodes "
                "reach it at."
            )

    # The node reads both to build the identity it presents, so either one missing
    # stops it.
    for label, key in (("private key", "private_key"), ("certificate", "public_key")):
        path = config.node_config.getpath("certificate", key)
        if not os.path.isfile(path):
            problems.append(
                f"The node {label} is missing from {path}, so the node cannot "
                "present its identity under mutual TLS."
            )

    return problems


def _registry_warnings(registered: List[Dict[str, Any]]) -> List[str]:
    """Registrations that break the rules a node registry must satisfy.

    Registration refuses these, so an entry breaking one predates the checks or
    was written by hand.
    """
    warnings = []
    if len(registered) > 1:
        warnings.append(
            "A node registers at most one certificate - its researcher's - but "
            f"{len(registered)} are registered. Delete the extra entries."
        )

    expiring = [
        certificate["component_id"]
        for certificate in registered
        if certificate["expiring_soon"]
    ]
    if expiring:
        warnings.append(
            f"Certificate(s) expiring within {CERTIFICATE_EXPIRY_WARNING_DAYS} days: "
            f"{', '.join(expiring)}. Ask the component to renew and register the new "
            "one."
        )

    researcher_host = config.node_config.get("researcher", "ip")
    for certificate in registered:
        names = certificate["san"]
        if not names:
            warnings.append(
                "The researcher certificate carries no host name, so the node "
                f"verifies it under {researcher_host} and the connection fails. "
                "Ask the researcher for a certificate naming the host it serves."
            )
        elif researcher_host not in names:
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
    mtls_enabled = config.node_config.getbool(
        "authentication", "mutual_authentication", fallback="False"
    )
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


def _configured_certificate_paths() -> Tuple[str, str]:
    """The certificate and private key paths the node's config points at.

    Both replacing and regenerating write here rather than anywhere of their own,
    so the node keeps reading the files its configuration already names.
    """
    return (
        config.node_config.getpath("certificate", "public_key"),
        config.node_config.getpath("certificate", "private_key"),
    )


def _public_key_bytes(key: Any) -> bytes:
    """A public key in the one encoding, so two of them compare by value."""
    return key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _validate_certificate_pair(certificate: str, private_key: str) -> None:
    """Check a supplied pair before it displaces the node's own.

    Everything the node would only discover at startup or mid-handshake is
    checked here instead, while the current pair is still in place.

    Raises:
        ValueError: If either file is unreadable, the key does not belong to the
            certificate, or the certificate is outside its validity window. The
            message is what the user is shown.
    """
    try:
        parsed = x509.load_pem_x509_certificate(certificate.encode("utf-8"))
    except (TypeError, ValueError) as exp:
        raise ValueError(f"The certificate is not readable PEM: {exp}") from exp

    try:
        parsed_key = serialization.load_pem_private_key(
            private_key.encode("utf-8"), password=None
        )
    except (TypeError, ValueError, UnsupportedAlgorithm) as exp:
        # The underlying error is an OpenSSL dump that tells the user nothing.
        raise ValueError(
            "The private key could not be read. It has to be an unencrypted "
            "private key in PEM format."
        ) from exp

    if _public_key_bytes(parsed_key.public_key()) != _public_key_bytes(
        parsed.public_key()
    ):
        raise ValueError(
            "The private key does not match the certificate, so this pair could "
            "never complete a handshake."
        )

    now = datetime.now(timezone.utc)
    if parsed.not_valid_after_utc < now:
        raise ValueError(
            f"The certificate expired on {parsed.not_valid_after_utc:%Y-%m-%d}, so "
            "every connection made with it would fail."
        )

    if parsed.not_valid_before_utc > now:
        raise ValueError(
            f"The certificate is not valid before "
            f"{parsed.not_valid_before_utc:%Y-%m-%d %H:%M} UTC, so every connection "
            "made with it would fail until then."
        )


def _back_up(path: str) -> Optional[str]:
    """Copy a file aside under a timestamped name, returning where it went.

    A copy rather than a move: the file stays readable at its configured path
    until the new one overwrites it, so a failure midway leaves the node with a
    pair it can still serve.

    Returns:
        Path of the backup, or None when there was no file to back up.
    """
    if not os.path.isfile(path):
        return None

    backup = f"{path}.bak-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    shutil.copy2(path, backup)

    return backup


def _write_certificate_pair(
    certificate: str, private_key: str
) -> Dict[str, Optional[str]]:
    """Write a validated pair to the paths the node's config points at.

    Returns:
        Where each displaced file was backed up, by role, with None for a role
        that had no file in place.
    """
    certificate_path, private_key_path = _configured_certificate_paths()
    backups = {
        "certificate": _back_up(certificate_path),
        "private_key": _back_up(private_key_path),
    }

    for path in (certificate_path, private_key_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(certificate_path, "w") as file:
        file.write(certificate)

    with open(private_key_path, "w") as file:
        file.write(private_key)
    # The key arrived over HTTP; it is not left readable to anyone but the node.
    os.chmod(private_key_path, 0o600)

    return backups


def _register(certificate: str, component_id: Optional[str], upsert: bool) -> str:
    """Register a certificate, returning the component it was registered for."""
    certificate_manager = _certificate_manager()
    # `register_certificate` reads the certificate from a file, as the CLI hands it one
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "certificate.pem")
        try:
            with open(path, "w") as file:
                file.write(certificate)

            return certificate_manager.register_certificate(
                certificate_path=path,
                component_id=component_id,
                upsert=upsert,
                registering_purpose=COMPONENT_PURPOSE[config.node_config.COMPONENT_TYPE],
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
    """Register a certificate the node received from another component.

    The certificate is sent as text, whether the user pasted it or picked a file.
    `upsert` replaces an existing registration of the same component, which the user
    confirms after the conflict is reported. `component_id` is required only for a
    certificate that carries no component id of its own in `CN=`.
    """
    payload = request.get_json(silent=True) or {}
    certificate = payload.get("certificate")
    if not isinstance(certificate, str) or not certificate.strip():
        return error("A certificate in PEM format is required"), 400

    component_id = payload.get("component_id") or None
    if component_id is not None and not isinstance(component_id, str):
        return error("'component_id' must be a string"), 400

    try:
        registered_component_id = _register(
            certificate, component_id, bool(payload.get("upsert", False))
        )
    except FedbiomedError as exp:
        return error(str(exp)), 400
    except OSError as exp:
        return error(f"Could not read the certificate: {exp}"), 500

    return response(
        {
            "component_id": registered_component_id,
            "requires_restart": _restart_required(),
        },
        f"Certificate of {registered_component_id} has been registered.",
    ), 200


@api.route("/certificates/<component_id>", methods=["DELETE"])
@admin_required
def delete_certificate(component_id: str):
    """Remove a component's certificate from the node's registry."""
    certificate_manager = _certificate_manager()
    try:
        if not certificate_manager.get(component_id=component_id):
            return error(f"No certificate is registered for {component_id}"), 404

        certificate_manager.delete(component_id=component_id)
    except FedbiomedError as exp:
        return error(f"Could not delete the certificate: {exp}"), 400
    finally:
        certificate_manager.close()

    return response(
        {"component_id": component_id, "requires_restart": _restart_required()},
        f"Certificate of {component_id} has been deleted.",
    ), 200


@api.route("/certificates/export", methods=["GET"])
@admin_required
def export_certificate():
    """Return this node's certificate, to be shared with the other components.

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
            "component_id": config.node_config.get("default", "id"),
            "filename": os.path.basename(path),
            "certificate": certificate,
        }
    ), 200


@api.route("/certificates/generate", methods=["POST"])
@admin_required
def generate_own_certificate():
    """Issue this node a fresh certificate and private key.

    The pair is generated aside and then written over the configured paths, so
    a configuration naming any file gets its own file back. The displaced pair
    is kept as a timestamped backup. The previous key stops being the node's
    identity: every component holding the old certificate has to register the
    new one.
    """
    try:
        with tempfile.TemporaryDirectory() as directory:
            # A node certificate is resolved by fingerprint and never verified by
            # name, so it is issued for no host.
            key_file, pem_file = (
                CertificateManager.generate_self_signed_ssl_certificate(
                    certificate_folder=directory,
                    certificate_name="certificate",
                    component_id=config.node_config.get("default", "id"),
                    purpose=COMPONENT_PURPOSE[config.node_config.COMPONENT_TYPE],
                )
            )
            with open(pem_file) as file:
                certificate = file.read()
            with open(key_file) as file:
                private_key = file.read()

        backups = _write_certificate_pair(certificate, private_key)
    except FedbiomedError as exp:
        return error(f"Could not generate the certificate: {exp}"), 400
    except OSError as exp:
        return error(f"Could not write the new certificate: {exp}"), 500

    return response(
        {
            "certificate": _own_certificate(),
            "backups": backups,
            "requires_restart": _restart_required(),
        },
        "A new certificate has been generated. Send it to the researcher, which "
        "has to register it in place of the previous one.",
    ), 200


@api.route("/certificates/replace", methods=["POST"])
@admin_required
def replace_own_certificate():
    """Replace this node's certificate and private key with a supplied pair.

    Both parts are required and are validated together before anything on disk
    is touched, so a pair the node could not serve is refused while the current
    one still stands. The pair is written over the configured paths and the
    displaced one is kept as a timestamped backup.
    """
    payload = request.get_json(silent=True) or {}
    certificate = payload.get("certificate")
    private_key = payload.get("private_key")

    if not isinstance(certificate, str) or not certificate.strip():
        return error("A certificate in PEM format is required"), 400

    if not isinstance(private_key, str) or not private_key.strip():
        return error("The matching private key in PEM format is required"), 400

    try:
        _validate_certificate_pair(certificate, private_key)
    except ValueError as exp:
        return error(str(exp)), 400

    try:
        backups = _write_certificate_pair(certificate, private_key)
    except OSError as exp:
        return error(f"Could not write the new certificate: {exp}"), 500

    return response(
        {
            "certificate": _own_certificate(),
            "backups": backups,
            "requires_restart": _restart_required(),
        },
        "The certificate has been replaced. Send it to the researcher, which has "
        "to register it in place of the previous one.",
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

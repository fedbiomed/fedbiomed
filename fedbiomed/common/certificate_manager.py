# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import copy
import ipaddress
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from tabulate import tabulate
from tinydb import Query, TinyDB
from tinydb.table import Document, Table

from fedbiomed.common.constants import (
    CERTS_FOLDER_NAME,
    ComponentType,
    ErrorNumbers,
)
from fedbiomed.common.db import DBTable
from fedbiomed.common.exceptions import FedbiomedCertificateError, FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import read_file

# Subject organization marking a certificate as issued by Fed-BioMed.
CERT_ORGANIZATION = "Fed-BioMed"
# TLS role a generated certificate's Extended Key Usage declares it for.
CERT_PURPOSE_SERVER = "server"
CERT_PURPOSE_CLIENT = "client"
# TLS role each component acts in: a node dials the researcher, which serves.
_COMPONENT_PURPOSE = {
    ComponentType.NODE.name: CERT_PURPOSE_CLIENT,
    ComponentType.RESEARCHER.name: CERT_PURPOSE_SERVER,
}
CERTIFICATE_EXPIRY_WARNING_DAYS = 30


def _validated_component_type(component_type: str) -> str:
    """Returns the component type, which is `NODE` or `RESEARCHER` and nothing else.

    Both what a component's certificate declares and the rules its registrations
    are held to follow from this value, so anything else is refused rather than
    falling through to the researcher's, which are the permissive ones.

    Raises:
        FedbiomedCertificateError: the value is not one of the two component types.
    """
    if component_type not in ComponentType.__members__:
        raise FedbiomedCertificateError(
            f"{ErrorNumbers.FB619.value}: `{component_type}` is not a component "
            f"type. It is one of {', '.join(ComponentType.__members__)}."
        )
    return component_type


def certificate_subject_field(
    certificate: bytes, oid: x509.ObjectIdentifier
) -> Optional[str]:
    """Extracts a subject field (e.g. `CN=` or `O=`) from a PEM certificate.

    Args:
        certificate: PEM encoded certificate.
        oid: Subject attribute OID, e.g. `x509.oid.NameOID.COMMON_NAME`.

    Returns:
        The field value, or None if absent or unparsable.
    """
    try:
        return (
            x509.load_pem_x509_certificate(certificate)
            .subject.get_attributes_for_oid(oid)[0]
            .value
        )
    except (IndexError, AttributeError, TypeError, ValueError):
        return None


def certificate_component_id(certificate: Union[bytes, str]) -> Optional[str]:
    """Component id of a PEM certificate, or None if Fed-BioMed did not issue it.

    The CommonName counts as a component id only on a certificate whose organization is
    Fed-BioMed; any other issuer puts what it likes there. Self-asserted, so it says how
    to read the field, not that the certificate is trusted.
    """
    if isinstance(certificate, str):
        certificate = certificate.encode("utf-8")

    organization = certificate_subject_field(certificate, NameOID.ORGANIZATION_NAME)
    if organization != CERT_ORGANIZATION:
        return None

    return certificate_subject_field(certificate, NameOID.COMMON_NAME)


def is_loopback_name(name: str) -> bool:
    """Whether a host name or address is one of the forms of the local machine."""
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return name.lower() == "localhost"

    # `is_loopback` follows the mapping itself only from CPython 3.11.10 on.
    mapped = getattr(address, "ipv4_mapped", None)
    return (address if mapped is None else mapped).is_loopback


def san_entry(name: str) -> x509.GeneralName:
    """The SAN entry a host name or address belongs in.

    TLS never matches an address against a `dNSName` entry, so an IP literal is
    issued as an `iPAddress` and everything else as a name.
    """
    try:
        return x509.IPAddress(ipaddress.ip_address(name))
    except ValueError:
        return x509.DNSName(name)


def certificate_san_names(certificate: Union[bytes, str]) -> List[str]:
    """Hosts and addresses a certificate is valid for.

    Read from the Subject Alternative Name, the only place a certificate states
    them: the Common Name is free text and carries no host.

    Only `dNSName` and `iPAddress` entries, the two TLS matches a server name
    against; an e-mail address, a URI or a blank entry names no server.

    Args:
        certificate: PEM encoded certificate.

    Returns:
        The host names and addresses the certificate is valid for, empty when it
        declares none or cannot be read.
    """
    if isinstance(certificate, str):
        certificate = certificate.encode("utf-8")

    try:
        extensions = x509.load_pem_x509_certificate(certificate).extensions
        return [
            str(entry.value)
            for entry in extensions.get_extension_for_class(
                x509.SubjectAlternativeName
            ).value
            if isinstance(entry, (x509.DNSName, x509.IPAddress))
            # Kept as written, since that is the name TLS matches
            and str(entry.value).strip()
        ]
    except (x509.ExtensionNotFound, TypeError, ValueError, AttributeError):
        return []


def certificate_audit_fields(certificate: Union[bytes, str]) -> Dict[str, str]:
    """Identifying fields of a peer certificate, for security audit events.

    Reports the certificate a peer authenticated with precisely enough to trace
    it back to an issued credential, without emitting the certificate itself.
    Called while logging a connection, so a certificate that cannot be described
    yields an empty dict instead of raising into the connection path.
    """
    if isinstance(certificate, str):
        certificate = certificate.encode("utf-8")
    try:
        cert = x509.load_pem_x509_certificate(certificate)
        fields = {
            "cert_subject": cert.subject.rfc4514_string(),
            "cert_issuer": cert.issuer.rfc4514_string(),
            # Hex, as serials exceed what JSON consumers hold exactly as integers.
            "cert_serial": f"{cert.serial_number:x}",
            "cert_not_after": f"{cert.not_valid_after_utc:%Y-%m-%dT%H:%M:%SZ}",
        }
    except (TypeError, ValueError, AttributeError):
        return {}

    try:
        extension = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        san_names = [str(entry.value) for entry in extension.value]
    except (x509.ExtensionNotFound, TypeError, ValueError, AttributeError):
        san_names = []
    if san_names:
        fields["cert_san"] = ",".join(san_names)

    return fields


def certificate_expiry(certificate: Union[bytes, str]) -> Optional[datetime]:
    """Expiry date (`notAfter`, UTC) of a PEM certificate, or None if unparsable."""
    if isinstance(certificate, str):
        certificate = certificate.encode("utf-8")
    try:
        return x509.load_pem_x509_certificate(certificate).not_valid_after_utc
    except (TypeError, ValueError):
        return None


def certificate_fingerprint(certificate: Union[bytes, str]) -> Optional[bytes]:
    """SHA-256 fingerprint of a PEM certificate, or None if unparsable.

    Identifies a certificate by what it contains rather than by its PEM text,
    which TLS layers re-encode: the certificate a peer presents is compared to
    the registered one on this value.
    """
    if isinstance(certificate, str):
        certificate = certificate.encode("utf-8")
    try:
        return x509.load_pem_x509_certificate(certificate).fingerprint(hashes.SHA256())
    except (TypeError, ValueError):
        return None


class TrustedCertificateBundle:
    """Registered certificates, for mutual authentication.

    Answers the two questions the researcher asks of its registry: which
    certificates to trust (the PEM bundle, as the trusted-certificate source of
    the gRPC server) and, for a certificate a peer presented, which component it is
    registered as. Both are served from one re-read of the certificate database,
    performed only when its file changes, so they cannot disagree and so
    certificates registered after startup are picked up without a restart.
    Thread-safe: called from gRPC handshake threads.

    Certificates expiring within `CERTIFICATE_EXPIRY_WARNING_DAYS` are reported
    whenever the database is re-read. Note that a re-read only happens when the
    database changes, so a certificate crossing the threshold while the database
    sits untouched is not reported until the next registration.
    """

    def __init__(self, db_path: str):
        """
        Args:
            db_path: Path of the certificate database.
        """
        self._db_path = db_path
        self._lock = threading.Lock()
        self._bundle: bytes = b""
        self._component_ids: Dict[bytes, str] = {}
        self._warned: set = set()

    def __call__(self) -> bytes:
        """Current PEM bundle, refreshed when the database file has changed."""
        with self._lock:
            self._refresh()
            return self._bundle

    def component_id(self, certificate: Union[bytes, str]) -> Optional[str]:
        """Component id the given certificate is registered under.

        The authoritative identity of a peer: every registered certificate has a
        component id, taken from its own `CN=` field or supplied at registration, so
        this also identifies certificates embedding no Fed-BioMed identity.

        Args:
            certificate: PEM encoded certificate, as presented by the peer.

        Returns:
            The registered component id, or None if the certificate is unparsable,
            not registered, or registered under more than one component id.
        """
        fingerprint = certificate_fingerprint(certificate)
        if fingerprint is None:
            return None

        with self._lock:
            self._refresh()
            return self._component_ids.get(fingerprint)

    def _refresh(self) -> None:
        """Re-reads the database if its file changed. The caller holds the lock.

        The database is written non-atomically by other processes registering
        certificates, so a read may land on a partially written file. What was
        last read is kept in that case and the read retried on the next call.
        """
        try:
            certificate_manager = CertificateManager(
                db_path=self._db_path,
                component_type=ComponentType.RESEARCHER.name,
            )
            try:
                documents = certificate_manager.list()
                expiring = certificate_manager.expiring_certificates(
                    CERTIFICATE_EXPIRY_WARNING_DAYS
                )
            finally:
                certificate_manager.close()

            registrations: Dict[bytes, List[str]] = {}
            for doc in documents:
                fingerprint = certificate_fingerprint(doc["certificate"])
                if fingerprint is not None:
                    registrations.setdefault(fingerprint, []).append(
                        doc["component_id"]
                    )

            component_ids = {}
            for fingerprint, registered in registrations.items():
                registered_ids = sorted(set(registered))
                # A certificate under several component ids identifies none of them:
                # binding a peer to an arbitrary one of them would let each act
                # under the others' identity. Left unmapped, so peers presenting
                # it are refused until the registry is corrected.
                if len(registered_ids) > 1:
                    msg = (
                        "The same certificate is registered "
                        f"under {', '.join(f'`{p}`' for p in registered_ids)} in "
                        f"{self._db_path}; "
                        "none of them can be authenticated with it. Delete the "
                        "duplicate registrations."
                    )
                    logger.warning(msg)
                    logger.security_event(
                        operation="certificate_ambiguous_identity",
                        status="warning",
                        db_path=self._db_path,
                        component_ids=registered_ids,
                        detail=msg,
                    )
                    continue
                component_ids[fingerprint] = registered_ids[0]

            self._bundle = "\n".join(d["certificate"] for d in documents).encode(
                "utf-8"
            )
            self._component_ids = component_ids
            self._warn_expiring(expiring)
        except (OSError, FedbiomedError) as e:
            msg = (
                f"Could not read certificate database {self._db_path}: {e}. "
                "Keeping the previously loaded certificates."
            )
            logger.warning(msg)
            logger.security_event(
                operation="certificate_store_unreadable",
                status="warning",
                db_path=self._db_path,
                detail=msg,
            )

    def _warn_expiring(self, expiring: List[Tuple[str, datetime]]) -> None:
        """Reports certificates expiring soon, once per certificate.

        A renewed certificate has a new expiry date, so it is reported again while
        it remains within the warning window.

        Args:
            expiring: `(component_id, expiry)` of certificates expiring soon.
        """
        for component_id, expiry in expiring:
            if (component_id, expiry) not in self._warned:
                msg = (
                    f"Certificate `{component_id}` expires on "
                    f"{expiry:%Y-%m-%d}; register an updated certificate to avoid "
                    "connection failures."
                )
                logger.warning(msg)
                logger.security_event(
                    operation="certificate_expiring",
                    status="warning",
                    component_id=component_id,
                    expires_on=f"{expiry:%Y-%m-%d}",
                    detail=msg,
                )
        self._warned = set(expiring)


class CertificateManager:
    """Certificate manager to manage certificates of parties

    A manager is one component's, and the rules a registration has to satisfy are
    that component's. Its type is therefore held here rather than passed at each
    call, so that no caller registers under another component's rules.

    Attrs:
        _db: TinyDB database to store certificates
        _component_type: Type of the component the manager belongs to
    """

    def __init__(self, db_path: str, component_type: str):
        """Opens a component's certificate database.

        Args:
            db_path: The name of the DB file to connect through TinyDB
            component_type: Type of the component the manager belongs to, `NODE`
                or `RESEARCHER`.

        Raises:
            FedbiomedCertificateError: the component type is neither `NODE` nor
                `RESEARCHER`.
        """
        self._component_type: str = _validated_component_type(component_type)
        self._query: Query = Query()

        db = TinyDB(db_path)
        db.table_class = DBTable
        self._tinydb: Optional[TinyDB] = db
        self._db: Optional[Table] = db.table("Certificates")

    @property
    def _table(self) -> Table:
        if self._db is None:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: The certificate database is closed."
            )
        return self._db

    def close(self) -> None:
        """Closes the underlying TinyDB handle and releases the open file."""
        if self._tinydb is not None:
            self._tinydb.close()
            self._tinydb = None
            self._db = None

    def _insert(
        self,
        certificate: str,
        component_id: str,
        upsert: bool = False,
    ) -> Union[int, list[int]]:
        """Writes a certificate into the table, granting it trust.

        The only writer: private so that every registration goes through `register`,
        which holds the rules a stored certificate must satisfy.

        Args:
            certificate: Public-key for the FL parties
            component_id: ID of the component
            upsert: Update document with new data if it is existing

        Returns:
            Document ID of inserted certificate

        Raises:
            FedbiomedCertificateError: component is already registered
        """
        certificate_ = self.get(component_id=component_id)
        if not certificate_:
            result = self._table.insert(
                {
                    "certificate": certificate,
                    "component_id": component_id,
                }
            )
        elif upsert:
            result = self._table.upsert(
                {
                    "certificate": certificate,
                    "component_id": component_id,
                },
                self._query.component_id == component_id,
            )
        else:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Component {component_id} already "
                "registered. Please use `upsert=True` or '--upsert' option through CLI"
            )

        # Audited here rather than at the callers, so every path that grants trust
        # is recorded. `replaced` marks an existing pin being overwritten.
        logger.security_event(
            operation="certificate_registered",
            status="success",
            component_id=component_id,
            replaced=bool(certificate_),
            **certificate_audit_fields(certificate),
        )

        return result

    def get(self, component_id: str) -> Union[dict, None]:
        """Gets certificate/public key  of given component

        Args:
            component_id: ID of the component which certificate will be retrieved
                from DB

        Returns:
            Certificate, dict like TinyDB document
        """

        v = self._table.get(self._query.component_id == component_id)
        return v

    def delete(self, component_id) -> List[int]:
        """Deletes given component from table

        Args:
            component_id: Component id

        Returns:
            The document IDs of deleted certificates
        """

        # Read before removing: revoking trust is auditable only while the entry
        # being removed can still be described.
        document = self.get(component_id=component_id) or {}
        removed = self._table.remove(self._query.component_id == component_id)

        if removed:
            logger.security_event(
                operation="certificate_deleted",
                status="success",
                component_id=component_id,
                **certificate_audit_fields(document.get("certificate", "")),
            )

        return removed

    def list(self, verbose: bool = False) -> List[Document]:
        """Lists registered certificates.

        Args:
            verbose: Prints list of registered certificates in tabular format

        Returns:
            List of certificate objects registered in DB
        """
        certificates = self._table.all()

        if verbose:
            to_print = copy.deepcopy(certificates)
            for doc in to_print:
                expiry = certificate_expiry(doc.pop("certificate"))
                doc["expires"] = expiry.strftime("%Y-%m-%d") if expiry else "unknown"

            print(tabulate(to_print, headers="keys"))

        return certificates

    def expiring_certificates(self, within_days: int) -> List[Tuple[str, datetime]]:
        """`(component_id, expiry)` for certs expiring within `within_days` (or
        expired)."""
        threshold = datetime.now(timezone.utc) + timedelta(days=within_days)
        expiring = []
        for doc in self._table.all():
            expiry = certificate_expiry(doc["certificate"])
            if expiry is not None and expiry <= threshold:
                expiring.append((doc["component_id"], expiry))
        return sorted(expiring, key=lambda item: item[1])

    def register_certificate(
        self,
        certificate_path: str,
        component_id: Optional[str] = None,
        upsert: bool = False,
    ) -> str:
        """Registers the certificate stored at the given path.

        Reads the file; `register` holds the rules, the meaning of every other
        argument and the returned component id.

        Raises:
            FedbiomedCertificateError: the path is not a file, or `register` rejects
                the certificate.
        """
        if not os.path.isfile(certificate_path):
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Certificate path does not represents a file."
            )

        return self.register(
            certificate=read_file(certificate_path),
            component_id=component_id,
            upsert=upsert,
        )

    def register(
        self,
        certificate: str,
        component_id: Optional[str] = None,
        upsert: bool = False,
    ) -> str:
        """Registers a certificate, applying every rule a stored certificate must meet.

        The single entry point for granting trust: it validates, then writes. Callers
        holding a file use `register_certificate` instead, and no caller writes to the
        table directly. Which rules apply follows from the component this manager was
        opened for: a node holds one certificate, its researcher's, and requires it to
        state a host.

        The component id may be recovered from the certificate's `CN=`, but only on a
        certificate Fed-BioMed issued (`O=Fed-BioMed`); any other issuer's `CN=` is
        free text:

        - certificate carries an identity, `component_id` omitted: recovered;
        - certificate carries none: `component_id` is required;
        - both present: they must be the same.

        Args:
            certificate: PEM encoded certificate to register.
            component_id: ID of the component the certificate belongs to. Optional
                when the certificate embeds an identity in `CN=`, required otherwise.
            upsert: Replaces the certificate registered for that component; without
                it, registering a component twice raises.

        Returns:
            The component id the certificate was registered under, which the caller
            may not have supplied: it is recovered from the certificate when
            `component_id` is omitted.

        Raises:
            FedbiomedCertificateError: If the certificate cannot be read or has
                expired; if `component_id` is neither given nor recoverable from the
                certificate; if a given `component_id` conflicts with the certificate
                identity; if the certificate is already registered under another
                component id; or, on a node, if it already holds a certificate for
                another component or is given one stating no host.
        """
        # Every rule below passes on a certificate it cannot read, so reject it first.
        fingerprint = certificate_fingerprint(certificate)
        if fingerprint is None:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: The certificate could not be read: it is "
                "not a PEM encoded certificate. Register the `.pem` file the component "
                "serves."
            )

        # An expired certificate completes no handshake, so registering it would only
        # defer the failure to the connection, where it is reported as a dropped
        # handshake naming no certificate. The date is there to read: the PEM the
        # fingerprint was taken from is the one it is read from.
        expiry = certificate_expiry(certificate)
        now = datetime.now(timezone.utc)
        if expiry <= now:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: The certificate expired on "
                f"{expiry:%Y-%m-%d}, so no connection can be established with it. "
                "Request the party it belongs to reissue its certificate, and "
                "register the one it serves then."
            )

        certificate_id = certificate_component_id(certificate)

        if certificate_id is not None:
            if component_id is not None and component_id != certificate_id:
                raise FedbiomedCertificateError(
                    f"{ErrorNumbers.FB619.value}: Given component id `{component_id}` "
                    f"does not match the certificate identity `{certificate_id}`."
                )
            component_id = certificate_id
        elif component_id is None:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: The certificate does not embed a "
                "Fed-BioMed identity, so `component_id` must be provided to "
                "register it."
            )

        registering_on_node = self._component_type == ComponentType.NODE.name

        # gRPC falls back to the Common Name, which holds the component id, not a host
        if registering_on_node and not certificate_san_names(certificate):
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: The certificate states no host: its "
                "Subject Alternative Name carries no host name and no address, so "
                "nothing in it says which server it is valid for. Request the "
                "researcher to reissue its certificate for the hosts nodes reach "
                "it at."
            )

        others = [d for d in self.list() if d["component_id"] != component_id]

        # A node communicates with a single researcher, so it holds one certificate
        if registering_on_node and others:
            registered = ", ".join(f"`{d['component_id']}`" for d in others)
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: A node registers at most one "
                f"certificate. Cannot register `{component_id}` while other "
                f"certificates are registered: {registered}. Delete them first."
            )

        # A researcher registers one certificate per node, so the same one
        # under two component ids would identify neither.
        duplicates = [
            d["component_id"]
            for d in others
            if certificate_fingerprint(d["certificate"]) == fingerprint
        ]
        if duplicates:
            registered = ", ".join(f"`{d}`" for d in duplicates)
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: This certificate is already "
                f"registered under {registered}, and a certificate identifies a "
                f"single component. Delete that registration, or register a "
                f"certificate of its own for `{component_id}`."
            )

        self._insert(
            certificate=certificate,
            component_id=component_id,
            upsert=upsert,
        )

        # Reported here because a node hears about the researcher certificate it pins
        # nowhere else: only the researcher watches its registry for expiries.
        if expiry <= now + timedelta(days=CERTIFICATE_EXPIRY_WARNING_DAYS):
            logger.warning(
                f"Certificate `{component_id}` expires on {expiry:%Y-%m-%d}; "
                "register an updated certificate to avoid connection failures."
            )

        return component_id

    @staticmethod
    def generate_self_signed_ssl_certificate(
        certificate_folder,
        component_id: str,
        purpose: str,
        certificate_name: str = "FBM_",
        san: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """Creates self-signed certificates

        The subject states who the component is, never where it is reached:
        `CN=<component_id>`, `O=` the organization marking it as Fed-BioMed's. The
        Extended Key Usage declares `purpose`, the single TLS role the component acts
        in.

        Args:
            certificate_folder: The path where certificate files `.pem` and `.key`
                will be saved. Path should be absolute.
            component_id: ID of the component, which the certificate identifies.
            purpose: TLS role the certificate declares, `CERT_PURPOSE_CLIENT` or
                `CERT_PURPOSE_SERVER`.
            certificate_name: Name of the certificate file.
            san: Hosts and IP addresses the component is reached at, which peers
                verify it under. Required for a certificate peers verify by name.

        Returns:
            private_key: Private key file
            public_key: Certificate file

        Raises:
            FedbiomedCertificateError: If the purpose is unknown, if certificate
                directory is invalid, or an error occurs while writing certificate
                files in given path.

        !!! info "Certificate files"
                Certificate files will be saved in the given directory as
                `certificates.key` for private key `certificate.pem` for public key.
        """
        if purpose not in (CERT_PURPOSE_CLIENT, CERT_PURPOSE_SERVER):
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Unknown certificate purpose `{purpose}`; "
                f"expected `{CERT_PURPOSE_CLIENT}` or `{CERT_PURPOSE_SERVER}`."
            )

        if not os.path.isabs(certificate_folder):
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Certificate path should be absolute: "
                f"{certificate_folder}"
            )

        if not os.path.isdir(certificate_folder):
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Certificate path is not valid: {certificate_folder}"
            )

        pkey = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # The names given and no others. One issued for none — a node's, resolved by
        # fingerprint — carries no SubjectAlternativeName at all.
        san_names = list(dict.fromkeys(entry for entry in san or [] if entry))

        # A peer on this machine dials it by whichever loopback form it holds, and
        # verifies only what the certificate carries: naming one names all three.
        if any(is_loopback_name(name) for name in san_names):
            san_names = list(
                dict.fromkeys([*san_names, "localhost", "127.0.0.1", "::1"])
            )

        # Who the component is, not where: its id, under the organization that marks
        # the certificate as issued by Fed-BioMed.
        name = x509.Name(
            [
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, CERT_ORGANIZATION),
                x509.NameAttribute(NameOID.COMMON_NAME, component_id),
            ]
        )

        builder = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(pkey.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=5 * 365))
        )

        if purpose == CERT_PURPOSE_SERVER:
            extended_key_usages = [ExtendedKeyUsageOID.SERVER_AUTH]
            key_encipherment = True
        else:
            extended_key_usages = [ExtendedKeyUsageOID.CLIENT_AUTH]
            key_encipherment = False

        builder = (
            builder.add_extension(
                x509.BasicConstraints(ca=False, path_length=None), critical=True
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=key_encipherment,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(x509.ExtendedKeyUsage(extended_key_usages), critical=False)
        )

        if san_names:
            builder = builder.add_extension(
                x509.SubjectAlternativeName([san_entry(name) for name in san_names]),
                critical=False,
            )

        certificate = builder.sign(private_key=pkey, algorithm=hashes.SHA256())

        # Certificate names
        key_file = os.path.join(certificate_folder, f"{certificate_name}.key")
        pem_file = os.path.join(certificate_folder, f"{certificate_name}.pem")

        try:
            with open(key_file, "wb") as f:
                f.write(
                    pkey.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                        encryption_algorithm=serialization.NoEncryption(),
                    )
                )
        except Exception as e:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Can not write private key: {e}"
            ) from e

        try:
            with open(pem_file, "wb") as f:
                f.write(certificate.public_bytes(serialization.Encoding.PEM))
        except Exception as e:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Can not write certificate: {e}"
            ) from e

        return key_file, pem_file


def generate_component_certificate(
    component_type: str,
    component_id: str,
    folder: str,
    name: str,
    host: Optional[str] = None,
    extra_san: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Issues the certificate a component serves, as its type calls for.

    Both the TLS role and the hosts the certificate is issued for follow from the
    component type alone, so every caller issuing on a component's behalf — its
    creation, the CLI, the GUI — produces the same certificate: a researcher
    serves and is verified by name, so it is issued for the host it is reached at;
    a node dials and is resolved by fingerprint, so it is issued for no host.

    Args:
        component_type: Type of the component the certificate identifies, `NODE`
            or `RESEARCHER`.
        component_id: ID of the component, which the certificate identifies.
        folder: Absolute path the `.key` and `.pem` files are written in.
        name: Base name shared by both files.
        host: Host the component is reached at, required for a researcher.
        extra_san: Further hosts and addresses a researcher is reached at.

    Returns:
        key_file: The path where private key file is saved
        pem_file: The path where public key file is saved

    Raises:
        FedbiomedCertificateError: the component type is neither `NODE` nor
            `RESEARCHER`; a host is given for a component whose certificate names
            none, or missing for one whose certificate needs it; or the files
            cannot be written.
    """
    component_type = _validated_component_type(component_type)

    if component_type == ComponentType.NODE.name:
        if host or extra_san:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: A node is never verified by name, so "
                "its certificate is issued for no host and takes none."
            )
        san = None
    else:
        if not host:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: A researcher is verified by name, so "
                "its certificate is issued for the host nodes reach it at, which "
                "has to be given."
            )
        san = [host, *(extra_san or [])]

    return CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=folder,
        certificate_name=name,
        component_id=component_id,
        purpose=_COMPONENT_PURPOSE[component_type],
        san=san,
    )


def generate_certificate(
    root,
    component_id,
    component_type: str,
    prefix: Optional[str] = None,
    host: Optional[str] = None,
) -> Tuple[str, str]:
    """Issues a component its certificate, under its own root, at creation.

    Refuses to replace one already there: the private key that goes with it is the
    component's identity, and the parties it talks to registered the certificate.

    Args:
        root: Root directory of the component.
        component_id: ID of the component for which the certificate will be generated
        component_type: Type of that component, which decides what the certificate
            declares and which hosts it is issued for.
        prefix: Base name shared by the certificate files.
        host: Host the component is reached at, required for a researcher.

    Returns:
        key_file: The path where private key file is saved
        pem_file: The path where public key file is saved

    Raises:
        FedbiomedCertificateError: If certificate directory for the component has already
            `certificate.pem` or `certificate.key` files generated; if the component
            type is neither `NODE` nor `RESEARCHER`; or if a host is given for a
            component whose certificate names none, or missing for one that needs it.
    """

    certificate_path = os.path.join(root, CERTS_FOLDER_NAME)

    if os.path.isdir(certificate_path) and (
        os.path.isfile(os.path.join(certificate_path, "certificate.key"))
        or os.path.isfile(os.path.join(certificate_path, "certificate.pem"))
    ):
        raise FedbiomedCertificateError(
            f"{ErrorNumbers.FB619.value}: Certificate generation is aborted. Directory "
            f"{certificate_path} has already certificates. Please remove those files to "
            "regenerate"
        )

    os.makedirs(certificate_path, exist_ok=True)

    return generate_component_certificate(
        component_type=component_type,
        component_id=component_id,
        folder=certificate_path,
        name=prefix if prefix else "",
        host=host,
    )

# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import copy
import ipaddress
import os
import threading
import uuid
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

CERTIFICATE_EXPIRY_WARNING_DAYS = 30

# TLS role a certificate is restricted to via Extended Key Usage.
CERT_PURPOSE_SERVER = "server"
CERT_PURPOSE_CLIENT = "client"


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


def certificate_names(certificate: Union[bytes, str]) -> List[str]:
    """Hosts and addresses a certificate is valid for.

    Read from the Subject Alternative Name, the only place a certificate states
    them: the Common Name is free text and carries no host.

    Args:
        certificate: PEM encoded certificate.

    Returns:
        The certificate's SAN entries, empty when it declares none or cannot be
        read.
    """
    if isinstance(certificate, str):
        certificate = certificate.encode("utf-8")

    try:
        san = x509.load_pem_x509_certificate(certificate).extensions
        return [
            str(name.value)
            for name in san.get_extension_for_class(x509.SubjectAlternativeName).value
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
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        names = [str(name.value) for name in san.value]
    except (x509.ExtensionNotFound, TypeError, ValueError, AttributeError):
        names = []
    if names:
        fields["cert_san"] = ",".join(names)

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


def certificate_expected_component_type(certificate: bytes) -> Optional[str]:
    """Component type a certificate's Extended Key Usage restricts it to.

    A client-only certificate is a node's, a server-only one a researcher's.

    Returns:
        `ComponentType.NODE.name` or `ComponentType.RESEARCHER.name`, or None
        when the certificate is unparsable or its EKU does not restrict it to a
        single role (third-party certificates need not carry one).
    """
    try:
        usages = (
            x509.load_pem_x509_certificate(certificate)
            .extensions.get_extension_for_class(x509.ExtendedKeyUsage)
            .value
        )
    except (ValueError, x509.ExtensionNotFound):
        return None
    client = ExtendedKeyUsageOID.CLIENT_AUTH in usages
    server = ExtendedKeyUsageOID.SERVER_AUTH in usages
    if client == server:  # both roles or neither: component type unconstrained
        return None
    return ComponentType.NODE.name if client else ComponentType.RESEARCHER.name


def validate_registering_component(
    certificate: str, component_type: str, registering_component_type: str
) -> None:
    """Checks a certificate is not of the registering component's own kind.

    Components register the certificates of the components they talk to, never
    of their own type: a node registers researcher certificates and a researcher
    node ones. Two independent checks, each restrictive only for information
    actually present:

    - the component type the certificate is registered as must differ from the
      registering component's type;
    - when the Extended Key Usage restricts the certificate to a single TLS
      role, that role must not be the registering component's own (a node acts
      as a TLS client, a researcher as a TLS server). An absent or dual-role
      EKU constrains nothing.

    Raises:
        FedbiomedCertificateError: the certificate is identified as belonging
            to the registering component's own type.
    """
    if component_type == registering_component_type:
        raise FedbiomedCertificateError(
            f"{ErrorNumbers.FB619.value}: Cannot register a {component_type} "
            f"certificate on a {registering_component_type}: a component registers "
            "the certificates of the components it communicates with, not of its "
            "own type."
        )
    eku_component_type = certificate_expected_component_type(
        certificate.encode("utf-8")
    )
    if eku_component_type == registering_component_type:
        role = (
            CERT_PURPOSE_CLIENT
            if eku_component_type == ComponentType.NODE.name
            else CERT_PURPOSE_SERVER
        )
        raise FedbiomedCertificateError(
            f"{ErrorNumbers.FB619.value}: The certificate's Extended Key Usage "
            f"restricts it to a TLS {role} certificate — the "
            f"{registering_component_type}'s own role, not that of the components "
            "it registers."
        )


def _certificate_purpose(component_id: str) -> Optional[str]:
    """TLS role to restrict a certificate to, from the component id prefix.

    Inferred as in `register`. Ids matching neither prefix yield None, allowing
    both roles so the handshake never breaks for such ids.
    """
    if component_id.upper().startswith(f"{ComponentType.NODE.name}_"):
        return CERT_PURPOSE_CLIENT
    if component_id.upper().startswith(f"{ComponentType.RESEARCHER.name}_"):
        return CERT_PURPOSE_SERVER
    return None


def _component_type(component_id: str) -> Optional[str]:
    """Component type encoded in a component id, or None if it does not follow
    the pattern.

    Component ids follow `<COMPONENT_TYPE>_<uuid4>` (see `Config.generate`); the
    prefix is matched case-insensitively.
    """
    prefix, _, identifier = component_id.partition("_")
    component_type = prefix.upper()
    try:
        uuid.UUID(identifier)
    except ValueError:
        return None
    if component_type not in (ComponentType.NODE.name, ComponentType.RESEARCHER.name):
        return None
    return component_type


class TrustedCertificateBundle:
    """Certificates registered for a component type, for mutual authentication.

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

    def __init__(self, db_path: str, component_type: str):
        """
        Args:
            db_path: Path of the certificate database.
            component_type: Component type name, e.g. `ComponentType.NODE.name`.
        """
        self._db_path = db_path
        self._component_type = component_type
        self._lock = threading.Lock()
        self._state: Optional[Tuple[int, int]] = None
        self._bundle: bytes = b""
        self._component_ids: Dict[bytes, str] = {}
        self._warned: set = set()

    def __call__(self) -> bytes:
        """Current PEM bundle, refreshed when the database file has changed."""
        with self._lock:
            self._refresh()
            return self._bundle

    @property
    def loaded(self) -> bool:
        """Whether the database has been read once, so that a caller can tell a
        certificate that is not registered from one it could not look up."""
        with self._lock:
            return self._state is not None

    def component_id(self, certificate: Union[bytes, str]) -> Optional[str]:
        """Component id the given certificate is registered under.

        The authoritative identity of a peer: every registered certificate has a
        component id, taken from its own `CN=` field or supplied at registration, so
        this also identifies certificates embedding no Fed-BioMed identity.

        Args:
            certificate: PEM encoded certificate, as presented by the peer.

        Returns:
            The registered component id, or None if the certificate is unparsable,
            not registered for this component type, or registered under more than
            one component id.
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
            stat = os.stat(self._db_path)
            state = (stat.st_mtime_ns, stat.st_size)
            if state == self._state:
                return

            certificate_manager = CertificateManager(db_path=self._db_path)
            try:
                documents = [
                    doc
                    for doc in certificate_manager.list()
                    if doc.get("component_type") == self._component_type
                ]
                expiring = certificate_manager.expiring_certificates(
                    CERTIFICATE_EXPIRY_WARNING_DAYS, self._component_type
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
                        f"The same {self._component_type} certificate is registered "
                        f"under {', '.join(f'`{p}`' for p in registered_ids)} in "
                        f"{self._db_path}; "
                        "none of them can be authenticated with it. Delete the "
                        "duplicate registrations."
                    )
                    logger.warning(msg)
                    logger.security_event(
                        operation="certificate_ambiguous_identity",
                        status="warning",
                        component_type=self._component_type,
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
            self._state = state
            self._warn_expiring(expiring)
        except (OSError, FedbiomedError) as e:
            if self._state is None:
                msg = (
                    f"Could not read certificate database {self._db_path}: {e}. "
                    f"No {self._component_type} certificate is available."
                )
            else:
                msg = (
                    f"Could not read certificate database {self._db_path}: {e}. "
                    "Keeping the previously loaded "
                    f"{self._component_type} certificates."
                )
            logger.warning(msg)
            logger.security_event(
                operation="certificate_store_unreadable",
                status="warning",
                component_type=self._component_type,
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
                    f"{self._component_type} certificate `{component_id}` expires on "
                    f"{expiry:%Y-%m-%d}; register an updated certificate to avoid "
                    "connection failures."
                )
                logger.warning(msg)
                logger.security_event(
                    operation="certificate_expiring",
                    status="warning",
                    component_id=component_id,
                    component_type=self._component_type,
                    expires_on=f"{expiry:%Y-%m-%d}",
                    detail=msg,
                )
        self._warned = set(expiring)


class CertificateManager:
    """Certificate manager to manage certificates of parties

    Attrs:
        _db: TinyDB database to store certificates
    """

    def __init__(self, db_path: Optional[str] = None):
        """Constructs certificate manager

        Args:
            db: The name of the DB file to connect through TinyDB
        """

        self._tinydb: Optional[TinyDB] = None
        self._db: Optional[Table] = None
        self._query: Query = Query()

        if db_path is not None:
            self.set_db(db_path)

    @property
    def _table(self) -> Table:
        if self._db is None:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Database not initialized. Call set_db() first."
            )
        return self._db

    def set_db(self, db_path: str) -> None:
        """Sets database

        Args:
            db_path: The path of DB file where `Certificates` table are stored
        """
        self.close()
        db = TinyDB(db_path)
        db.table_class = DBTable
        self._tinydb = db
        self._db = db.table("Certificates")

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
        component_type: str,
        upsert: bool = False,
    ) -> Union[int, list[int]]:
        """Writes a certificate into the table, granting it trust.

        The only writer: private so that every registration goes through `register`,
        which holds the rules a stored certificate must satisfy.

        Args:
            certificate: Public-key for the FL parties
            component_id: ID of the component
            component_type: Node or researcher,
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
                    "component_type": component_type,
                }
            )
        elif upsert:
            result = self._table.upsert(
                {
                    "certificate": certificate,
                    "component_type": component_type,
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
            component_type=component_type,
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

    def get_by_component_type(self, component_type: str) -> List[str]:
        """Gets certificates of all components of a given type.

        Args:
            component_type: Component type name, e.g. `ComponentType.NODE.name`.

        Returns:
            List of certificate contents (PEM strings).
        """
        return [
            doc["certificate"]
            for doc in self._table.search(self._query.component_type == component_type)
        ]

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
                component_type=document.get("component_type"),
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

    def expiring_certificates(
        self, within_days: int, component_type: Optional[str] = None
    ) -> List[Tuple[str, datetime]]:
        """`(component_id, expiry)` for certs expiring within `within_days` (or
        expired), optionally restricted to a `component_type`."""
        threshold = datetime.now(timezone.utc) + timedelta(days=within_days)
        expiring = []
        for doc in self._table.all():
            if (
                component_type is not None
                and doc.get("component_type") != component_type
            ):
                continue
            expiry = certificate_expiry(doc["certificate"])
            if expiry is not None and expiry <= threshold:
                expiring.append((doc["component_id"], expiry))
        return sorted(expiring, key=lambda item: item[1])

    def register_certificate(
        self,
        certificate_path: str,
        component_id: Optional[str] = None,
        upsert: bool = False,
        registering_component_type: Optional[str] = None,
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
            registering_component_type=registering_component_type,
        )

    def register(
        self,
        certificate: str,
        component_id: Optional[str] = None,
        upsert: bool = False,
        registering_component_type: Optional[str] = None,
    ) -> str:
        """Registers a certificate, applying every rule a stored certificate must meet.

        The single entry point for granting trust: it validates, then writes. Callers
        holding a file use `register_certificate` instead, and no caller writes to the
        table directly.

        The component id may be recovered from the certificate's `CN=` (CommonName)
        field. A certificate identity counts only when it is itself a valid
        component id (`<NODE|RESEARCHER>_<uuid>`):

        - certificate carries a valid identity, `component_id` omitted: recovered;
        - certificate carries no valid identity: `component_id` is required and must
          follow the pattern;
        - both present: they must be the same.

        Args:
            certificate: PEM encoded certificate to register.
            component_id: ID of the component. Optional when the certificate embeds a
                valid identity in `CN=`; required otherwise.
            upsert: If `True` overwrites existing certificate for specified component.
                If `False` and the certificate for the specified component already
                existing it raises error.
            registering_component_type: Component type performing the registration
                (`ComponentType.NODE.name` or `ComponentType.RESEARCHER.name`).
                When given, certificates identified as that component's own kind
                are rejected: components register each other's certificates, never
                their own type. A node additionally keeps a single registered
                certificate — its researcher's. None skips these checks.

        Returns:
            The component id the certificate was registered under, which the caller
            may not have supplied: it is recovered from the certificate when
            `component_id` is omitted.

        Raises:
            FedbiomedCertificateError: If `component_id` is neither given nor
                recoverable from the certificate; if a given `component_id` conflicts
                with the certificate identity; if a given `component_id` does not
                follow `<NODE|RESEARCHER>_<uuid>`; if the certificate is already
                registered under another component id; or, when
                `registering_component_type` is given, if the certificate is
                identified — by the component id it is registered under or by a
                single-role EKU — as one of the registering component's own kind,
                or if a node already holds a certificate for another component.
        """
        certificate_id = certificate_subject_field(
            certificate.encode("utf-8"), NameOID.COMMON_NAME
        )
        # A certificate identity is usable only if it is itself a valid component id.
        certificate_component_type = (
            _component_type(certificate_id) if certificate_id is not None else None
        )

        # A provided component id must always follow the expected pattern, whether it
        # is later used as-is or reconciled against the certificate identity.
        given_component_type = (
            _component_type(component_id) if component_id is not None else None
        )
        if component_id is not None and given_component_type is None:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Component id `{component_id}` is "
                "invalid; it must follow `<NODE|RESEARCHER>_<uuid>`."
            )

        if certificate_component_type is not None:
            if component_id is not None and component_id != certificate_id:
                raise FedbiomedCertificateError(
                    f"{ErrorNumbers.FB619.value}: Given component id `{component_id}` "
                    f"does not match the certificate identity `{certificate_id}`."
                )
            component_id, component_type = certificate_id, certificate_component_type
        elif given_component_type is not None:
            # Third-party certificate: an arbitrary `CN=`, registered under the
            # given `component_id`.
            component_type = given_component_type
        else:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: The certificate does not embed a valid "
                "Fed-BioMed identity, so `component_id` must be provided to "
                "register it."
            )

        if registering_component_type is not None:
            validate_registering_component(
                certificate, component_type, registering_component_type
            )

        # A node communicates with a single researcher, so its database holds at most one certificate
        if registering_component_type == ComponentType.NODE.name:
            others = [
                d["component_id"]
                for d in self.list()
                if d["component_id"] != component_id
            ]
            if others:
                registered = ", ".join(f"`{o}`" for o in others)
                raise FedbiomedCertificateError(
                    f"{ErrorNumbers.FB619.value}: A node registers at most one "
                    f"certificate. Cannot register `{component_id}` while other "
                    f"certificates are registered: {registered}. Delete them first."
                )

        # A researcher registers one certificate per node, so the same one
        # under two component ids would identify neither.
        fingerprint = certificate_fingerprint(certificate)
        duplicates = [
            d["component_id"]
            for d in self.list()
            if d["component_id"] != component_id
            and certificate_fingerprint(d["certificate"]) == fingerprint
        ]
        if fingerprint is not None and duplicates:
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
            component_type=component_type,
            upsert=upsert,
        )

        return component_id

    @staticmethod
    def _write_certificate_file(path: str, certificate: str) -> None:
        """Writes certificate file

        Args:
            path: Filesystem path the file will be written
            certificate: Certificate that will be written

        Raises:
            FedbiomedCertificateError: If certificate can not be written into given path
        """
        try:
            with open(path, "w", encoding="UTF-8") as file:
                file.write(certificate)
        except Exception as e:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Can not write certificate file {path}. "
                f"Aborting the operation. Please check raised exception: {e}"
            ) from e

    @staticmethod
    def generate_self_signed_ssl_certificate(
        certificate_folder,
        certificate_name: str = "FBM_",
        component_id: str = "unknown",
        subject: Optional[Dict[str, str]] = None,
        san: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """Creates self-signed certificates

        The Extended Key Usage restricting the certificate to a single TLS role is
        inferred from `component_id`: a node acts as a TLS client, a researcher as
        a TLS server.

        Args:
            certificate_folder: The path where certificate files `.pem` and `.key`
                will be saved. Path should be absolute.
            certificate_name: Name of the certificate file.
            component_id: ID of the component
            subject: Subject fields, `CommonName` being the component id peers register
                the certificate under. The subject states who a component is, never
                where it is reached.
            san: Hosts and IP addresses the component is reached at, which peers
                verify it under. Required for a certificate peers verify by name.

        Returns:
            private_key: Private key file
            public_key: Certificate file

        Raises:
            FedbiomedCertificateError: If certificate directory is invalid or an error
                occurs while writing certificate files in given path.

        !!! info "Certificate files"
                Certificate files will be saved in the given directory as
                `certificates.key` for private key `certificate.pem` for public key.
        """
        subject = subject or {}

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

        # The names given, plus the loopback names a party on the same machine dials
        # it by. A certificate issued for none — a node's, resolved by fingerprint —
        # carries no SubjectAlternativeName at all.
        names = [entry for entry in san or [] if entry]
        if names:
            names = list(dict.fromkeys([*names, "localhost", "127.0.0.1"]))

        # The subject holds the component id alone: who the component is, not where.
        name = x509.Name(
            [
                x509.NameAttribute(
                    NameOID.COMMON_NAME, subject.get("CommonName", component_id)
                )
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

        resolved_purpose = _certificate_purpose(component_id)
        if resolved_purpose == CERT_PURPOSE_SERVER:
            extended_key_usages = [ExtendedKeyUsageOID.SERVER_AUTH]
            key_encipherment = True
        elif resolved_purpose == CERT_PURPOSE_CLIENT:
            extended_key_usages = [ExtendedKeyUsageOID.CLIENT_AUTH]
            key_encipherment = False
        else:
            extended_key_usages = [
                ExtendedKeyUsageOID.SERVER_AUTH,
                ExtendedKeyUsageOID.CLIENT_AUTH,
            ]
            key_encipherment = True

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

        if names:
            entries: List[x509.GeneralName] = []
            for entry in names:
                try:
                    entries.append(x509.IPAddress(ipaddress.ip_address(entry)))
                except ValueError:
                    entries.append(x509.DNSName(entry))
            builder = builder.add_extension(
                x509.SubjectAlternativeName(entries), critical=False
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
                f"{ErrorNumbers.FB619.value}: Can not write public key: {e}"
            ) from e

        try:
            with open(pem_file, "wb") as f:
                f.write(certificate.public_bytes(serialization.Encoding.PEM))
        except Exception as e:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Can not write certificate: {e}"
            ) from e

        return key_file, pem_file


def generate_certificate(
    root,
    component_id,
    prefix: Optional[str] = None,
    subject: Optional[Dict[str, str]] = None,
    san: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Generates certificates

    Args:
        component_id: ID of the component for which the certificate will be generated
        subject: Subject fields of the certificate, stating who the component is.
        san: Hosts and IP addresses the component is reached at, which peers verify
            it under.

    Returns:
        key_file: The path where private key file is saved
        pem_file: The path where public key file is saved

    Raises:
        FedbiomedCertificateError: If certificate directory for the component has already
            `certificate.pem` or `certificate.key` files generated.
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

    key_file, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=certificate_path,
        certificate_name=prefix if prefix else "",
        component_id=component_id,
        subject=subject,
        san=san,
    )

    return key_file, pem_file

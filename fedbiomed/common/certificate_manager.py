# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import copy
import ipaddress
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from tabulate import tabulate
from tinydb import Query, TinyDB
from tinydb.table import Document, Table

from fedbiomed.common.constants import (
    CERTS_FOLDER_NAME,
    NODE_PREFIX,
    ComponentType,
    ErrorNumbers,
)
from fedbiomed.common.db import DBTable
from fedbiomed.common.exceptions import FedbiomedCertificateError
from fedbiomed.common.utils import read_file


def certificate_subject_field(
    certificate: bytes, oid: x509.ObjectIdentifier
) -> Optional[str]:
    """Extracts a subject field (e.g. `O=` or `CN=`) from a PEM certificate.

    Args:
        certificate: PEM encoded certificate.
        oid: Subject attribute OID, e.g. `x509.oid.NameOID.ORGANIZATION_NAME`.

    Returns:
        The field value, or None if absent or unparsable.
    """
    try:
        return (
            x509.load_pem_x509_certificate(certificate)
            .subject.get_attributes_for_oid(oid)[0]
            .value
        )
    except (IndexError, AttributeError, ValueError):
        return None


def certificate_expiry(certificate: Union[bytes, str]) -> Optional[datetime]:
    """Expiry date (`notAfter`, UTC) of a PEM certificate, or None if unparsable."""
    if isinstance(certificate, str):
        certificate = certificate.encode("utf-8")
    try:
        return x509.load_pem_x509_certificate(certificate).not_valid_after_utc
    except (TypeError, ValueError):
        return None


def is_mtls_enabled(config) -> bool:
    """Whether mutual TLS is enabled in the `[mtls]` config section.

    A missing section or `enabled` entry means disabled (legacy workflow).

    Args:
        config: Component configuration object.

    Returns:
        True if mutual TLS with certificate pinning is enabled.
    """
    return config.getbool("mtls", "enabled", fallback="False")


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
        db = TinyDB(db_path)
        db.table_class = DBTable
        self._db = db.table("Certificates")

    def insert(
        self,
        certificate: str,
        party_id: str,
        component: str,
        upsert: bool = False,
    ) -> Union[int, list[int]]:
        """Inserts new certificate

        Args:
            certificate: Public-key for the FL parties
            party_id: ID of the party
            component: Node or researcher,
            ip: IP of the component which the certificate will be registered
            port: Port of the component which the certificate will be registered
            upsert: Update document with new data if it is existing

        Returns:
            Document ID of inserted certificate

        Raises:
            FedbiomedCertificateError: party is already registered
        """
        certificate_ = self.get(party_id=party_id)
        if not certificate_:
            return self._table.insert(
                {
                    "certificate": certificate,
                    "party_id": party_id,
                    "component": component,
                }
            )

        if upsert:
            return self._table.upsert(
                {
                    "certificate": certificate,
                    "component": component,
                    "party_id": party_id,
                },
                self._query.party_id == party_id,
            )
        raise FedbiomedCertificateError(
            f"{ErrorNumbers.FB619.value}: Party {party_id} already registered. "
            "Please use `upsert=True` or '--upsert' option through CLI"
        )

    def get(self, party_id: str) -> Union[dict, None]:
        """Gets certificate/public key  of given party

        Args:
            party_id: ID of the party which certificate will be retrieved from DB

        Returns:
            Certificate, dict like TinyDB document
        """

        v = self._table.get(self._query.party_id == party_id)
        return v

    def get_by_component(self, component: str) -> List[str]:
        """Gets certificates of all parties of a given component type.

        Args:
            component: Component type name, e.g. `ComponentType.NODE.name`.

        Returns:
            List of certificate contents (PEM strings).
        """
        return [
            doc["certificate"]
            for doc in self._table.search(self._query.component == component)
        ]

    def delete(self, party_id) -> List[int]:
        """Deletes given party from table

        Args:
            party_id: Party id

        Returns:
            The document IDs of deleted certificates
        """

        return self._table.remove(self._query.party_id == party_id)

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
        self, within_days: int, component: Optional[str] = None
    ) -> List[Tuple[str, datetime]]:
        """`(party_id, expiry)` for certs expiring within `within_days` (or expired),
        optionally restricted to a `component` type."""
        threshold = datetime.now(timezone.utc) + timedelta(days=within_days)
        expiring = []
        for doc in self._table.all():
            if component is not None and doc.get("component") != component:
                continue
            expiry = certificate_expiry(doc["certificate"])
            if expiry is not None and expiry <= threshold:
                expiring.append((doc["party_id"], expiry))
        return sorted(expiring, key=lambda item: item[1])

    def register_certificate(
        self,
        certificate_path: str,
        party_id: str,
        upsert: bool = False,
    ) -> Union[int, List[int]]:
        """Registers certificate

        Args:
            certificate_path: Path where certificate/key file stored
            party_id: ID of the FL party which the certificate will be registered
            upsert: If `True` overwrites existing certificate for specified party.  If `False`
                and the certificate for the specified party already existing it raises error.

        Returns:
            The document ID of registered certificated.

        Raises:
            FedbiomedCertificateError: If certificate file is not existing in file system
        """

        if not os.path.isfile(certificate_path):
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Certificate path does not represents a file."
            )

        # Read certificate content
        certificate_content = read_file(certificate_path)

        # Save certificate in database
        component = (
            ComponentType.NODE.name
            if party_id.startswith(NODE_PREFIX)
            else ComponentType.RESEARCHER.name
        )

        return self.insert(
            certificate=certificate_content,
            party_id=party_id,
            component=component,
            upsert=upsert,
        )

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
    ) -> Tuple[str, str]:
        """Creates self-signed certificates

        Args:
            certificate_folder: The path where certificate files `.pem` and `.key`
                will be saved. Path should be absolute.
            certificate_name: Name of the certificate file.
            component_id: ID of the component

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

        if not os.path.abspath(certificate_folder):
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Certificate path should be absolute: "
                f"{certificate_folder}"
            )

        if not os.path.isdir(certificate_folder):
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Certificate path is not valid: {certificate_folder}"
            )

        pkey = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        cn = subject.get("CommonName", "*")
        on = subject.get("OrganizationName", component_id)

        name = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, cn),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, on),
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

        san: Optional[x509.GeneralName]
        try:
            san = x509.IPAddress(ipaddress.ip_address(cn))
        except ValueError:
            san = x509.DNSName(cn) if cn and cn != "*" else None
        if san is not None:
            builder = builder.add_extension(
                x509.SubjectAlternativeName([san]), critical=False
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
) -> Tuple[str, str]:
    """Generates certificates

    Args:
        component_id: ID of the component for which the certificate will be generated

    Returns:
        key_file: The path where private key file is saved
        pem_file: The path where public key file is saved

    Raises:
        FedbiomedEnvironError: If certificate directory for the component has already
            `certificate.pem` or `certificate.key` files generated.
    """

    certificate_path = os.path.join(root, CERTS_FOLDER_NAME)

    if os.path.isdir(certificate_path) and (
        os.path.isfile(os.path.join(certificate_path, "certificate.key"))
        or os.path.isfile(os.path.join(certificate_path, "certificate.pem"))
    ):
        raise ValueError(
            f"Certificate generation is aborted. Directory {certificate_path} has already "
            f"certificates. Please remove those files to regenerate"
        )

    os.makedirs(certificate_path, exist_ok=True)

    key_file, pem_file = CertificateManager.generate_self_signed_ssl_certificate(
        certificate_folder=certificate_path,
        certificate_name=prefix if prefix else "",
        component_id=component_id,
        subject=subject,
    )

    return key_file, pem_file

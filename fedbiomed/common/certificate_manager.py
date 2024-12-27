# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import copy
import ipaddress
import os
import random
from typing import Dict, List, Optional, Tuple, Union

from OpenSSL import crypto
from tabulate import tabulate
from tinydb import Query, TinyDB
from tinydb.table import Table

from fedbiomed.common.constants import (
    CERTS_FOLDER_NAME,
    NODE_PREFIX,
    ComponentType,
    ErrorNumbers,
)
from fedbiomed.common.db import DBTable
from fedbiomed.common.exceptions import FedbiomedCertificateError
from fedbiomed.common.utils import read_file


class CertificateManager:
    """Certificate manager to manage certificates of parties

    Attrs:
        _db: TinyDB database to store certificates
    """

    def __init__(self, db_path: str = None):
        """Constructs certificate manager

        Args:
            db: The name of the DB file to connect through TinyDB
        """

        self._db: Union[Table, None] = None
        self._query: Query = Query()

        if db_path is not None:
            self.set_db(db_path)

    def set_db(self, db_path: str) -> None:
        """Sets database

        Args:
            db_path: The path of DB file where `Certificates` table are stored
        """
        db = TinyDB(db_path)
        db.table_class = DBTable
        self._db: Table = db.table("Certificates")

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
            return self._db.insert(
                {
                    "certificate": certificate,
                    "party_id": party_id,
                    "component": component,
                }
            )

        if upsert:
            return self._db.upsert(
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

        v = self._db.get(self._query.party_id == party_id)
        return v

    def delete(self, party_id) -> List[int]:
        """Deletes given party from table

        Args:
            party_id: Party id

        Returns:
            The document IDs of deleted certificates
        """

        return self._db.remove(self._query.party_id == party_id)

    def list(self, verbose: bool = False) -> List[dict]:
        """Lists registered certificates.

        Args:
            verbose: Prints list of registered certificates in tabular format

        Returns:
            List of certificate objects registered in DB
        """
        certificates = self._db.all()

        if verbose:
            to_print = copy.deepcopy(certificates)
            for doc in to_print:
                doc.pop("certificate")

            print(tabulate(to_print, headers="keys"))

        return certificates

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
            path: Filesystem path the file will be writen
            certificate: Certificate that will be writen

        Raises:
            FedbiomedCertificateError: If certificate can not be writen into given path
        """
        try:
            with open(path, "w", encoding="UTF-8") as file:
                file.write(certificate)
                file.close()
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

        pkey = crypto.PKey()
        pkey.generate_key(crypto.TYPE_RSA, 2048)

        cn = subject.get("CommonName", "*")
        on = subject.get("OrganizationName", component_id)

        x509 = crypto.X509()
        subject = x509.get_subject()
        subject.commonName = cn
        subject.organizationName = on
        x509.set_issuer(subject)

        extensions = []
        try:
            if ipaddress.ip_address(cn):
                extensions.append(
                    # TODO: X509Extension is deprecated, update with newer version
                    crypto.X509Extension(
                        type_name=b"subjectAltName",
                        critical=False,
                        value=f"IP:{cn}".encode("ASCII"),
                    )
                )
        except ValueError:
            pass
        if extensions:
            x509.add_extensions(extensions)

        x509.gmtime_adj_notBefore(0)
        x509.gmtime_adj_notAfter(5 * 365 * 24 * 60 * 60)
        x509.set_pubkey(pkey)
        x509.set_serial_number(random.randrange(100000))
        x509.set_version(2)
        x509.sign(pkey, "SHA256")

        # Certificate names
        key_file = os.path.join(certificate_folder, f"{certificate_name}.key")
        pem_file = os.path.join(certificate_folder, f"{certificate_name}.pem")

        try:
            with open(key_file, "wb") as f:
                f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, pkey))
                f.close()
        except Exception as e:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Can not write public key: {e}"
            ) from e

        try:
            with open(pem_file, "wb") as f:
                f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, x509))
                f.close()
        except Exception as e:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Can not write public key: {e}"
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

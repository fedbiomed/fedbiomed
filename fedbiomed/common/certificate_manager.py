# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import random

from OpenSSL import crypto
from typing import List, Union, Tuple
from tinydb import TinyDB, Query
from tinydb.table import Table
from tabulate import tabulate

from fedbiomed.common.constants import ComponentType, MPSPDZ_certificate_prefix, NODE_PREFIX, ErrorNumbers
from fedbiomed.common.exceptions import FedbiomedError, FedbiomedCertificateError
from fedbiomed.common.utils import read_file


class CertificateManager:
    """ Certificate manager to manage certificates of parties

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
            self._db: Table = TinyDB(db_path).table("Certificates")

    def set_db(self, db_path: str) -> None:
        """Sets database

        Args:
            db_path: The path of DB file where `Certificates` table are stored
        """
        self._db = TinyDB(db_path).table("Certificates")

    def insert(
            self,
            certificate: str,
            party_id: str,
            component: str,
            ip: str,
            port: str,
            upsert: bool = False
    ) -> Union[int, list[int]]:
        """ Inserts new certificate

        Args:
            certificate: Public-key for the FL parties
            party_id: ID of the party
            component: Node or researcher,
            ip: MP-SPDZ IP of the component which the certificate will be registered
            port: MP-SPDZ Port of the component which the certificate will be registered
            upsert: Update document with new data if it is existing

        Returns:
            Document ID of inserted certificate

        Raises:
            FedbiomedCertificateError: party is already registered
        """
        certificate_ = self.get(party_id=party_id)
        if not certificate_:
            return self._db.insert(dict(
                certificate=certificate,
                party_id=party_id,
                component=component,
                ip=ip,
                port=port
            ))

        elif upsert:
            return self._db.upsert(
                dict(
                    certificate=certificate,
                    component=component,
                    party_id=party_id,
                    ip=ip,
                    port=port
                ),
                self._query.party_id == party_id
            )
        else:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Party {party_id} already registered. Please use `upsert=True` or "
                f"'--upsert' option through CLI")

    def get(
            self,
            party_id: str
    ) -> Union[dict, None]:
        """Gets certificate/public key  of given party

        Args:
            party_id: ID of the party which certificate will be retrieved from DB

        Returns:
            Certificate, dict like TinyDB document
        """

        v = self._db.get(self._query.party_id == party_id)
        return dict(v) if v else None

    def delete(
            self,
            party_id
    ) -> List[int]:
        """Deletes given party from table

        Args:
            party_id: Party id

        Returns:
            The document IDs of deleted certificates
        """

        return self._db.remove(self._query.party_id == party_id)

    def list(self, verbose: bool = False) -> List[dict]:
        """ Lists registered certificates.

        Args:
            verbose: Prints list of registered certificates in tabular format

        Returns:
            List of certificate objects registered in DB
        """
        certificates = [ dict(c) for c in self._db.all() ]

        if verbose:
            to_print = copy.deepcopy(certificates)
            for doc in to_print:
                doc.pop('certificate')

            print(tabulate(to_print, headers='keys'))

        return certificates

    def register_certificate(
            self,
            certificate_path: str,
            party_id: str,
            ip: str,
            port: int,
            upsert: bool = False
    ) -> Union[int, List[int]]:
        """ Registers certificate

        Args:
            certificate_path: Path where certificate/key file stored
            party_id: ID of the FL party which the certificate will be registered
            ip:  MP-SPDZ IP address of the party where MP-SPDZ create communication
            port: MP-SPDZ port of the party
            upsert: If `True` overwrites existing certificate for specified party. If `False` and the certificate for
                the specified party already existing it raises error.

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
        with open(certificate_path) as file:
            certificate_content = file.read()
            file.close()

        # Save certificate in database
        component = ComponentType.NODE.name if party_id.startswith(NODE_PREFIX) else ComponentType.RESEARCHER.name

        return self.insert(
            certificate=certificate_content,
            party_id=party_id,
            ip=ip,
            port=port,
            component=component,
            upsert=upsert,
        )

    def write_mpc_certificates_for_experiment(
            self,
            parties: List[str],
            path: str,
            self_id: str,
            self_ip: str,
            self_port: int,
            self_private_key: str,
            self_public_key: str
    ) -> List[str]:
        """ Writes certificates into given directory respecting the order

        !!! info "Certificate Naming Convention"
                MP-SPDZ requires saving certificates respecting the naming convention `P<PARTY_ID>.pem`. Party ID should
                be integer in the order of [0,1, ...].  Therefore, the order of parties are critical in the sense of
                naming files in given folder path. Files will be named as `P[ORDER].pem` to make it compatible with
                MP-SPDZ.

        Args:
            parties: ID of the parties (nodes/researchers) will join FL experiment.y
            path: The path where certificate files will be writen
            self_id: ID of the component that will launch MP-SPDZ protocol
            self_ip: IP of the component that will launch MP-SPDZ protocol
            self_port: Port of the component that will launch MP-SPDZ protocol
            self_private_key: Path to MPSPDZ public key
            self_public_key: Path to MPSDPZ private key

        Returns:
            List of written certificates files (paths).

        Raises:
            FedbiomedCertificateError: - If certificate for given party is not existing in the database
                - If given path is not a directory
        """

        if not os.path.isdir(path):
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Specified `path` argument should be a directory. `path` is not a "
                f"directory or it is not existing."
            )

        path = os.path.abspath(path)
        self_private_key = os.path.abspath(self_private_key)
        self_public_key = os.path.abspath(self_public_key)

        ip_addresses = os.path.join(path, "ip_addresses")
        # Files already writen into directory
        writen_certificates = []

        # Function remove writen files in case of error
        def remove_writen_files():
            for wf in writen_certificates:
                os.remove(wf)
            if os.path.isfile(ip_addresses):
                os.remove(ip_addresses)

        if os.path.isfile(ip_addresses):
            os.remove(ip_addresses)

        # Get certificate for each party
        try:
            for index, party in enumerate(parties):

                # Self certificate requires to
                if party == self_id:
                    self_certificate_key = read_file(self_private_key)
                    self_certificate_pem = read_file(self_public_key)

                    key = os.path.join(path, f"P{index}.key")
                    pem = os.path.join(path, f"P{index}.pem")

                    self._write_certificate_file(key, self_certificate_key)
                    self._write_certificate_file(pem, self_certificate_pem)
                    self._append_new_ip_address(ip_addresses, self_ip, self_port, self_id)
                    writen_certificates.extend([key, pem])

                    continue

                # Remote parties
                party_object = self.get(party)
                if not party_object:
                    remove_writen_files()
                    raise FedbiomedCertificateError(
                        f"{ErrorNumbers.FB619.value}: Certificate for {party} is not existing. Aborting setup."
                    )

                path_ = os.path.join(path, f"P{index}.pem")
                self._write_certificate_file(path_, party_object["certificate"])
                writen_certificates.append(path_)

                self._append_new_ip_address(
                    ip_addresses,
                    party_object["ip"],
                    party_object["port"],
                    party_object["party_id"]
                )

        except FedbiomedCertificateError as e:
            # Remove all writen file in case of an error
            remove_writen_files()
            raise FedbiomedCertificateError(e)

        except FedbiomedError as e:
            remove_writen_files()
            raise FedbiomedCertificateError(f"{ErrorNumbers.FB619.value}: {e}")

        except Exception as e:
            remove_writen_files()
            raise FedbiomedCertificateError(f"{ErrorNumbers.FB619.value}: Undetermined error: {e}")

        return writen_certificates

    @staticmethod
    def _write_certificate_file(path: str, certificate: str) -> None:
        """ Writes certificate file

        Args:
            path: Filesystem path the file will be writen
            certificate: Certificate that will be writen

        Raises:
            FedbiomedCertificateError: If certificate can not be writen into given path
        """
        try:
            with open(path, 'w') as file:
                file.write(certificate)
                file.close()
        except Exception as e:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Can not write certificate file {path}. Aborting the operation. "
                f"Please check raised exception: {e}"
            )

    @staticmethod
    def _append_new_ip_address(path: str, ip: str, port: int, party_id: str):
        """Append new address to the file where the ip address of the parties are kept

        Args:
            path: Path to IP addresses file
            ip: IP address
            port: Port number
            party_id: ID of the party corresponds to given IP and port

        Raises:
            FedbiomedCertificateError: If new ip address can not be appended into the given file path (file).
        """
        try:
            with open(path, 'a') as file:
                file.write(f"{ip}:{port}\n")
                file.close()

        except Exception as e:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Can not write ip address of component: {party_id}. Aborting the "
                f"operation. Please check raised exception: {e}"
            )

    @staticmethod
    def generate_self_signed_ssl_certificate(
            certificate_folder,
            certificate_name: str = MPSPDZ_certificate_prefix,
            component_id: str = "unknown",
    ) -> Tuple[str, str]:
        """Creates self-signed certificates

        Args:
            certificate_folder: The path where certificate files `.pem` and `.key` will be saved. Path should be
                absolute.
            certificate_name: Name of the certificate file.
            component_id: ID of the component

        Returns:
            private_key: Private key file
            public_key: Certificate file

        Raises:
            FedbiomedCertificateError: If certificate directory is invalid or an error occurs while writing certificate
                files in given path.

        !!! info "Certificate files"
                Certificate files will be saved in the given directory as `certificates.key` for private key
                `certificate.pem` for public key.
        """

        if not os.path.abspath(certificate_folder):
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Certificate path should be absolute: {certificate_folder}"
            )

        if not os.path.isdir(certificate_folder):
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Certificate path is not valid: {certificate_folder}"
            )

        pkey = crypto.PKey()
        pkey.generate_key(crypto.TYPE_RSA, 2048)

        x509 = crypto.X509()
        subject = x509.get_subject()
        subject.commonName = '*'
        subject.organizationName = component_id
        x509.set_issuer(subject)
        x509.gmtime_adj_notBefore(0)
        x509.gmtime_adj_notAfter(5 * 365 * 24 * 60 * 60)
        x509.set_pubkey(pkey)
        x509.set_serial_number(random.randrange(100000))
        x509.set_version(2)
        x509.sign(pkey, 'SHA256')

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
            )

        try:
            with open(pem_file, "wb") as f:
                f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, x509))
                f.close()
        except Exception as e:
            raise FedbiomedCertificateError(
                f"{ErrorNumbers.FB619.value}: Can not write public key: {e}"
            )

        return key_file, pem_file

import os

from typing import Dict, List, Union, Tuple
from tinydb import TinyDB, Query
from tinydb.table import Document
from tabulate import tabulate

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography import x509
from cryptography.x509.oid import NameOID

from fedbiomed.common.validator import SchemeValidator, ValidateError
from fedbiomed.common.constants import ComponentType
from fedbiomed.common.exceptions import FedbiomedError

CertificateDataValidator = SchemeValidator({
    "email": {"rules": [str], "required": False, "default": "fed@biomed"},
    "country": {"rules": [str], "required": False, "default": "FR"},
    "organization": {"rules": [str], "required": False, "default": "Fed-BioMed"},
    "validity": {"rules": [int], "required": False, "default": 10 * 365 * 24 * 60 * 60}
})

"""Validator object for certificate data"""


class CertificateManager:
    """ Certificate manager to manage certificates of parties

    Attrs:
        _db: TinyDB database to store certificates
    """

    def __init__(self, db: str = None):
        """Constructs certificate manager

        Args:
            db: The name of the DB file to connect through TinyDB
        """

        self._db: Union[TinyDB, None] = None
        self._query: Query = Query()

        if db is not None:
            self._db: TinyDB.table = TinyDB(db).table("Certificates")

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
            upsert: bool = False
    ) -> List[int]:
        """ Inserts new certificate

        Args:
            certificate: Public-key for the FL parties
            party_id: ID of the party
            component: Node or researcher,
            upsert:

        Returns:
            Document ID of inserted certificate
        """
        certificate_ = self.get(party_id=party_id)

        if not certificate_:
            return self._db.insert(dict(
                certificate=certificate,
                party_id=party_id,
                component=component
            ))

        elif upsert:
            return self._db.upsert(
                dict(certificate=certificate, component=component, party_id=party_id),
                self._query.party_id == party_id
            )
        else:
            raise FedbiomedError(f"Party {party_id} already registered. Please use `upsert=True` or '--upsert' "
                                 f"option through CLI")

    def get(
            self,
            party_id: str
    ) -> Document:
        """Gets certificate/public key  of given party

        Args:
            party_id: ID of the party which certificate will be retrieved from DB

        Returns:
            Certificate, dict like TinyDB document
        """

        return self._db.get(self._query.party_id == party_id)

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

    def list(self, verbose: bool = False) -> List[Document]:
        """ Lists registered certificates.

        Args:
            verbose: Prints list of registered certificates in tabular format

        Returns:
            List of certificate objects registered in DB
        """
        certificates = self._db.all()

        if verbose:
            for doc in certificates:
                doc.pop('certificate')
            print(tabulate(certificates, headers='keys'))

        return certificates

    def register_certificate(
            self,
            certificate_path: str,
            party_id: str,
            upsert: bool = False
    ) -> Union[int, List[int]]:
        """ Registers certificate

        Args:
            certificate_path: Path where certificate/key file stored
            party_id: ID of the FL party which the certificate will be registered
            upsert: If `True` overwrites existing certificate for specified party. If `False` and the certificate for
                the specified party already existing it raises error.

        Raises:
            FedbiomedCertificateError: - If `upsert` is `False` and the certificate is already existing.
                - If certificate file is not existing in file system

        Returns:
            The document ID of registered certificated.
        """

        if not os.path.isfile(certificate_path):
            raise FedbiomedError(f"Certificate path does not represents a file.")

        # Read certificate content
        with open(certificate_path) as file:
            certificate_content = file.read()
            file.close()

        # Save certificate in database
        component = ComponentType.NODE.name if party_id.startswith("node") else ComponentType.RESEARCHER.name

        return self.insert(
                certificate=certificate_content,
                party_id=party_id,
                component=component,
                upsert=upsert
            )

    def write_certificates_for_experiment(
            self,
            parties: List[str],
            path: str
    ) -> List[str]:
        """ Writes certificates into given directory respecting the order

        !!! info "Certificate Naming Convention"
                MP-SPDZ requires saving certificates respecting the naming convention `P<PARTY_ID>.pem`. Party ID should
                be integer in the order of [0,1, ...].  Therefore, the order of parties are critical in the sense of
                naming files in given folder path. Files will be named as `P[ORDER].pem` to make it compatible with MP-SPDZ.

        Args:
            parties: ID of the parties (nodes/researchers) will join FL experiment.y
            path: The path where certificate files will be writen

        Raises:
            FedbiomedCertificateError: - If certificate for given party is not existing in the database
                - If certificate files can not be writen in given path.
                - If given path is not a directory

        Returns:
            List of writen certificates files (paths).
        """

        if not os.path.isdir(path):
            raise FedbiomedError(
                "Specified `path` argument should be a directory. `path` is not a directory or it is not existing."
            )

        # Files already writen into directory
        writen_certificates = []

        # Function remove writen files in case of error
        def remove_writen_files():
            for wf in writen_certificates:
                os.remove(wf)

        # Get certificate for each party
        for index, party in enumerate(parties):
            party = self.get(party)
            if not party:
                remove_writen_files()
                raise FedbiomedError(
                    f"Certificate for {party} is not existing. Aborting setup."
                )

            # Write certificate
            try:
                with open(os.path.join(path, f"P{index}.pem")) as file:
                    file.write(party.certificate)
                    file.close()
            except Exception as e:
                remove_writen_files()
                raise FedbiomedError(
                    f"Can not write certificate file for {party}. Aborting the operation. Please check raised "
                    f"exception: {e}"
                )

        return writen_certificates

    @staticmethod
    def generate_certificate(
            certificate_path,
            certificate_data: Dict = {},
    ) -> Tuple[str, str]:
        """Creates self-signed certificates

        Args:
            certificate_path: The path where certificate files `.pem` and `.key` will be saved. Path should be
                absolute.
            certificate_data: Data for certificates to declare, `email`, `country`, `organization`, `validity`.
                Certificate data should be dict where `email`, `country`, `organization` is string type and `validity`
                boolean

        Raises:
            FedbiomedCertificateError: If certificate directory is invalid or an error occurs while writing certificate
                files in given path.

        Returns:
            Status of the certificate creation.


        !!! info "Certificate files"
                Certificate files will be saved in the given directory as `certificates.key` for private key
                `certificate.pem` for public key.
        """

        if not os.path.abspath(certificate_path):
            raise FedbiomedError(f"Certificate path should be absolute: {certificate_path}")

        if not os.path.isdir(certificate_path):
            raise FedbiomedError(f"Certificate path is not valid: {certificate_path}")

        try:
            CertificateDataValidator.validate(certificate_data)
        except ValidateError as e:
            raise FedbiomedError(f"Certificate data is not valid: {e}")

        certificate_data = CertificateDataValidator.populate_with_defaults(
            certificate_data, only_required=False
        )

        # Generate our key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Generate a CSR
        csr = x509.CertificateSigningRequestBuilder() \
            .subject_name(
            x509.Name([
                # Provide various details about who we are.
                x509.NameAttribute(NameOID.COUNTRY_NAME, certificate_data["country"]),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, certificate_data["organization"]),
                x509.NameAttribute(NameOID.EMAIL_ADDRESS, certificate_data["email"]),
            ])).sign(private_key, hashes.SHA256())

        # Certificate names
        key_file = os.path.join(certificate_path, "certificate.key")
        pem_file = os.path.join(certificate_path, "certificate.pem")

        try:
            with open(key_file, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.BestAvailableEncryption(b"passphrase"),
                ))
                f.close()
        except Exception as e:
            raise FedbiomedError(f"Can not write public key: {e}")

        try:
            with open(pem_file, "wb") as f:
                f.write(csr.public_bytes(serialization.Encoding.PEM))
                f.close()
        except Exception as e:
            raise FedbiomedError(f"Can not write public key: {e}")

        return key_file, pem_file

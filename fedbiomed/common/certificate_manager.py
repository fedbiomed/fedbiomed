import os

from functools import wraps
from OpenSSL import crypto
from typing import Dict, List, Union
from tinydb import TinyDB, Query

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
            self._db: TinyDB = TinyDB(db, 'Certificates')

    def set_db(self, db_path: str) -> None:
        """Sets database """
        self._db = TinyDB(db_path, "Certificates")

    @classmethod
    def check_db(cls):


        pass


    def insert(
            self,
            certificate:
            str,
            party_id: str,
            component: str
    ) -> int:
        """ Inserts new certificate

        Args:
            certificate: Public-key for the FL parties
            party_id: ID of the party
            component: Node or researcher
        """

        return self._db.insert(dict(certificate=certificate,
                                    party=party_id,
                                    component=component
                                    )
                               )

    def get(
            self,
            party_id
    ) -> int:

        """Gets certificate/public key  of given party

        Args:
            party_id: Party id
        """

        return self._db.get(self._query.party == party_id)

    def delete(
            self,
            party_id
    ) -> Dict:
        """Deletes given party from table

        Args:
            party_id: Party id

        Returns:

        """

        return self._db.remove(self._query.party_id == party_id)

    def register_certificate(
            self,
            certificate_path,
            party_id
    ) -> int:

        if not os.path.isfile(certificate_path):
            raise FedbiomedError(f"Certificate path doe snot represents a file.")

        # Read certificate content
        with open(certificate_path) as file:
            certificate = file.read()
            file.close()

        # Save certificate in database
        component = ComponentType.NODE if party_id.startswith("node") else ComponentType.RESEARCHER
        register = self.insert(
            certificate=certificate,
            party_id=party_id,
            component=component
        )

        return register

    def write_certificates_for_experiment(
            self,
            parties: List[str],
            path: str
    ) -> List[str]:
        """ Writes certificates into given directory respecting the order

        Args:
            parties:
            path:

        """

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
    ) -> bool:
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

        # Certificate names
        key_file = os.path.join(certificate_path, "certificate.key")
        pem_file = os.path.join(certificate_path, "certificate.pem")

        private_key = crypto.PKey()
        private_key.generate_key(crypto.TYPE_RSA, 2048)

        # create a self-signed cert
        certificate = crypto.X509()
        certificate.get_subject().C = certificate_data["country"]
        certificate.get_subject().O = certificate_data["organization"]
        certificate.get_subject().emailAddress = certificate_data["email"]
        certificate.set_serial_number(0)
        certificate.gmtime_adj_notBefore(0)
        certificate.gmtime_adj_notAfter(certificate_data["validity"])
        certificate.set_issuer(certificate.get_subject())
        certificate.set_pubkey(private_key)

        # Sign certificate SHA512
        certificate.sign(private_key, 'sha512')

        try:
            with open(pem_file, "w") as f:
                f.write(
                    str(crypto.dump_certificate(
                        crypto.FILETYPE_PEM,
                        certificate.decode("utf-8")
                    ))
                )
        except Exception as e:
            raise FedbiomedError(f"Can not write public key: {e}")
        else:
            try:
                with open(key_file, "w") as f:
                    f.write(
                        str(crypto.dump_privatekey(
                            crypto.FILETYPE_PEM,
                            private_key).decode("utf-8"))
                    )
            except Exception as e:
                raise FedbiomedError(f"Can not write public key: {e}")
        finally:
            return key_file, pem_file

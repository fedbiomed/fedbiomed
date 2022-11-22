import os

from OpenSSL import crypto
from typing import Dict
from validator import SchemeValidator, ValidateError
from fedbiomed.common.exceptions import FedbiomedError


CertificateDataValidator = SchemeValidator({
    "email": {"rules": [str], "required": False, "default": "fed@biomed"},
    "country": {"rules": [str], "required": False, "default": "FR"},
    "organization": {"rules": [str], "required": False, "default": "Fed-BioMed"},
    "validity": {"rules": [int], "required": False, "default": 10 * 365 * 24 * 60 * 60}
})
"""Validator object for certificate data"""


class CertificateManager:

    # TODO: Add database CRUD operation here

    def __init__(self):
        pass

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
            return True

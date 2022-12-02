"""Common CLI Modules

This module includes common CLI methods and parser extension

"""

import argparse
import os
import sys
from typing import Dict
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.validator import SchemeValidator, ValidateError
from fedbiomed.common.certificate_manager import CertificateManager
from fedbiomed.common.logger import logger

# Create certificate dict validator
CertificateDataValidator = SchemeValidator({
    'DB_PATH': {"rules": [str], "required": True},
    'CERT_DIR': {"rules": [str], "required": True},
    'COMPONENT_ID': {"rules": [str], "required": True},
    'CERTIFICATE_DIR': {"rules": [str], "required": True}
})

RED = '\033[1;31m'  # red
YLW = '\033[1;33m'  # yellow
GRN = '\033[1;32m'  # green
NC = '\033[0m'  # no color
BOLD = '\033[1m'


class CommonCLI:

    def __init__(self):
        self._parser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog='fedbiomed_run [ node | researcher | gui ] config [CONFIG_NAME] ',
            formatter_class=argparse.RawTextHelpFormatter
        )

        self._subparsers = self._parser.add_subparsers()
        self._certificate_manager: CertificateManager = CertificateManager()
        self._environ = None
        self._description: str = ''
        self._args = None

    @property
    def parser(self):
        """Gets parser for CLI"""
        return self._parser

    @property
    def description(self):
        """Gets description of CLI"""
        return self._description

    @property
    def arguments(self):
        return self._args

    @description.setter
    def description(self, value) -> str:
        """Sets description for parser """
        self._description = value
        self._parser.description = value

        return self._description

    def set_environ(self, environ):
        """Sets envrion object"""
        self._environ = environ

    @staticmethod
    def error(message: str):
        """Prints given error message

        Args:
            message:
        """
        print(f"{RED}ERROR:{NC}")
        print(f"{BOLD}{message}{NC}")

    @staticmethod
    def success(message):
        """

        """
        print(f"{GRN}Operation successful! {NC}")
        print(f"{BOLD}{message}{NC}")

    def create_configuration(self):
        """"""

        configuration = self._subparsers.add_parser('configuration', help='a help')

        # Create sub parser under `configuration` command
        configuration_sub_parsers = configuration.add_subparsers(
            help='Certificate management commands. Please run [command] -h to see details of the commands'
        )

        recreate = configuration_sub_parsers.add_parser(
            'create',
            help="Recreates configuration file for the specified component"
        )

        recreate.set_defaults(func=self._create_component_configuration)

    def initialize_certificate_parser(self):
        """Common arguments """

        # Add certificate sub parser (sub-command)
        certificate_parser = self._subparsers.add_parser(
            'certificate',
            prog="fedbiomed_run [ node | researcher ] [config [CONFIG_FILE]] "
        )

        # Create sub parser under `certificate` command
        certificate_sub_parsers = certificate_parser.add_subparsers(
            help='Certificate management commands. Please run [command] -h to see details of the commands.'
        )

        register_parser = certificate_sub_parsers.add_parser(
            'register',
            help="Register certificate of specified party. Please run 'fedbiomed_run [COMPONENT SPECIFICATION] "
                 "certificate register --help'"
        )  # command register
        list_parser = certificate_sub_parsers.add_parser(
            'list',
            help="Lists registered certificates"
        )  # command list
        delete_parser = certificate_sub_parsers.add_parser(
            'delete',
            help="Deletes specified certificate from database")  # command delete

        # Command `certificate generate`
        generate = certificate_sub_parsers.add_parser(
            'generate',
            help="Generates certificate for given component/party. Overwrites exisintg certificate ff '--path' option "
                 "isn't specified ")

        # Command `certificate generate`
        prepare = certificate_sub_parsers.add_parser(
            'prepare-my-certificate-for-email',
            help="Prepare certificates of current component to send other FL participant through trusted channel.")

        register_parser.set_defaults(func=self._register_certificate)
        list_parser.set_defaults(func=self._list_certificates)
        delete_parser.set_defaults(func=self._delete_certificate)
        generate.set_defaults(func=self._generate_certificate)
        prepare.set_defaults(func=self._prepare_my_certificate_for_email)

        # Add arguments
        register_parser.add_argument(
            '-pk',
            '--public-key',
            metavar='PUBLIC_KEY',
            type=str,
            nargs='?',
            required=True,
            help='Certificate/key that will be registered')

        register_parser.add_argument(
            '-pi',
            '--party-id',
            metavar='PUBLIC_ID',
            type=str,
            nargs='?',
            required=True,
            help="ID of the party to which the certificate is to be registered (component ID).")

        register_parser.add_argument(
            '--upsert',
            action="store_true",
            help="Updates if certificate of given party id is already existing.")

        generate.add_argument(
            '--path',
            type=str,
            nargs='?',
            default=os.path.join(self._environ["CERT_DIR"], f"cert_{self._environ['ID']}"),
            help="The path where certificates will be saved. By default it will overwrite existing certificate.")

        generate.add_argument(
            '--organization',
            type=str,
            nargs='?',
            default="Fed-BioMed",
            help="Organization name for CSR")

        generate.add_argument(
            '--email',
            type=str,
            nargs='?',
            default="fed@biomed",
            help="E-mail address for CSR")

        generate.add_argument(
            '--country',
            type=str,
            nargs='?',
            default="FR",
            help="Country for CSR")

        # Set db path that certificate manager will be using to store certificates
        self._certificate_manager.set_db(db_path=self._environ["DB_PATH"])

    def _create_component_configuration(self, args):
        """CLI Handler for creating configuration file for given component

        TODO: This method doesn't do specific action for creating configuration file for
            given component. Since, `environ` will be imported through component CLI, configuration
            file will be automatically created. In future, it might be useful to generate configuration
            files.
        """
        pass

    @staticmethod
    def _generate_certificate(args):
        """Generates certificate using Certificate Manager

        Args:
            args: Arguments that are passed after `certificate generate` command

        """

        try:
            CertificateManager.generate_certificate(
                certificate_path=args.path,
                certificate_data={
                    "organization": args.organization,
                    "country": args.country,
                    "email": args.email
                })
        except FedbiomedError as e:
            CommonCLI.error(f"Can not generate certificate. Please see: {e}")
            sys.exit(101)

        else:
            CommonCLI.success(f"Certificate has been successfully generated in : {args.path} \n")

            print(f"Add following lines into {os.getenv('CONFIG_FILE', 'component')} configuration file: \n\n"
                  f"; -------------------------------------------------------------------------------------\n"
                  f"; - SSL Configuration \n"
                  f"; -------------------------------------------------------------------------------------\n"
                  f"[ssl]\n"
                  f"private_key = {args.path}/certificate.key \n"
                  f"public_key = {args.path}/certificate.pem \n\n")

        pass

    def _register_certificate(self, args):
        """ Registers certificate with given parameters"""

        try:
            t = self._certificate_manager.register_certificate(
                certificate_path=args.public_key,
                party_id=args.party_id,
                upsert=args.upsert
            )
            print(t)
        except FedbiomedError as exp:
            print(exp)
            sys.exit(101)
        else:
            print(f"{GRN}Success!{NC}")
            print(f"{BOLD}Certificate has been successfully created for party: {args.party_id}.{NC}")

    def _list_certificates(self, args):
        """ Lists saved certificates """

        self._certificate_manager.list(verbose=True)

    def _delete_certificate(self, args):

        certificates = self._certificate_manager.list(verbose=False)
        options = [d['party_id'] for d in certificates]
        msg = "Select the certificate to delete:\n"
        msg += "\n".join([f'{i}) {d}' for i, d in enumerate(options, 1)])
        msg += "\nSelect: "

        while True:
            try:
                opt_idx = int(input(msg)) - 1
                assert opt_idx in range(len(certificates))

                party_id = certificates[opt_idx]['party_id']
                self._certificate_manager.delete(party_id=party_id)
                print(f"{GRN}Success!{NC}")
                print(f"{BOLD}Certificate for '{party_id}' has been successfully removed {NC}")
                return
            except (ValueError, IndexError, AssertionError):
                logger.error('Invalid option. Please, try again.')

    def _prepare_my_certificate_for_email(self, args):

        try:
            with open(self._environ["CERTIFICATE_PEM"], 'r') as file:
                certificate = file.read()
                file.close()
        except Exception as e:
            CommonCLI.error(f"Error while reading certificate: {e}")

        else:
            print(f"Hi There! \n\n")
            print("Please find following certificate to register \n")
            print(certificate)

            print(f"{BOLD}Please follow the instructions below to register this certificate:{NC}\n\n")

            print(" 1- Copy certificate content into a file e.g 'Hospital1.pem'")
            print(" 2- Change your directory to 'fedbiomed' root")
            print(f" 2- Run: \"scripts/fedbiomed_run [node | researcher] certificate register -pk [PATH WHERE "
                  f"CERTIFICATE IS SAVED] -pi {self._environ['ID']} \" ")

        pass

    def parse_args(self):
        """Parse arguments after adding the arguments

        !!! warning "Attention"
                Please make sure this method is called after all necessary arguments are set

        """
        self._args = self._parser.parse_args()

        if hasattr(self._args, 'func'):
            self._args.func(self._args)


if __name__ == '__main__':
    print("ERROR:")
    print("This is a submodule. You can not execute directly. Please import and extend CLI parser")
    exit(2)

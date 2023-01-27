# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Common CLI Modules

This module includes common CLI methods and parser extension

"""

import argparse
import os
import sys
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.certificate_manager import CertificateManager
from fedbiomed.common.constants import DB_FOLDER_NAME, MPSPDZ_certificate_prefix
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import get_existing_component_db_names, \
    get_all_existing_certificates, \
    get_method_spec, \
    get_fedbiomed_root

RED = '\033[1;31m'  # red
YLW = '\033[1;33m'  # yellow
GRN = '\033[1;32m'  # green
NC = '\033[0m'  # no color
BOLD = '\033[1m'


class CommonCLI:

    def __init__(self) -> None:
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
    def parser(self) -> argparse.ArgumentParser:
        """Gets parser for CLI

        Returns:
            Main argument parser object
        """
        return self._parser

    @property
    def description(self) -> str:
        """Gets description of CLI

        Returns:
            Description (Intro) for the CLI
        """
        return self._description

    @property
    def arguments(self) -> argparse.Namespace:
        """Gets global parser arguments

        Returns:
            Parser arguments
        """
        return self._args

    @description.setter
    def description(self, value: str) -> str:
        """Sets description for parser

        Args:
            value: Description or Intro for the CLI

        Returns:
            The description set
        """
        self._description = value
        self._parser.description = value

        return self._description

    def set_environ(self, environ):
        """Sets environ object"""
        self._environ = environ

    @staticmethod
    def error(message: str) -> None:
        """Prints given error message

        Args:
            message: Error message
        """
        print(f"{RED}ERROR:{NC}")
        print(f"{BOLD}{message}{NC}")
        logger.critical(message)
        sys.exit(1)

    @staticmethod
    def success(message: str) -> None:
        """Prints given message with success tag

        Args:
            message: Message to print as successful operation
        """
        print(f"{GRN}Operation successful! {NC}")
        print(f"{BOLD}{message}{NC}")

    def initialize_magic_dev_environment_parsers(self) -> None:
        """Initializes argument parser for the option to create development environment."""
        magic = self._subparsers.add_parser(
            'certificate-dev-setup',
            description="Prepares development environment by registering certificates of each component created in a "
                        "single clone of Fed-BioMed. Parses configuration files ends with '.ini' that are created "
                        "in 'etc' directory. This setup requires to have one 'researcher' and at least 2 nodes.",
            help="Prepares development environment by registering certificates of each component created in a single "
                 "clone of Fed-BioMed."
        )
        magic.set_defaults(func=self._create_magic_dev_environment)

    def initialize_create_configuration(self) -> None:
        """Initializes argument parser for creating configuration file."""

        configuration = self._subparsers.add_parser('configuration', help='Configuration')

        # Create sub parser under `configuration` command
        configuration_sub_parsers = configuration.add_subparsers()

        create = configuration_sub_parsers.add_parser(
            'create',
            help="Creates configuration file for the specified component if it does not exist. "
                 "If the configuration file exists, leave it unchanged"
        )

        create.set_defaults(func=self._create_component_configuration)

    def initialize_certificate_parser(self):
        """Common arguments """

        # Add certificate sub parser (sub-command)
        certificate_parser = self._subparsers.add_parser(
            'certificate',
            prog="fedbiomed_run [ node | researcher ] [config [CONFIG_FILE]] certificate",

        )

        # Create sub parser under `certificate` command
        certificate_sub_parsers = certificate_parser.add_subparsers(
            description="Commands that can be used with the option `certificate`",
            title="Subcommands"
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
            help="Generates certificate for given component/party if files don't exist yet. "
                 "Overwrites existing certificate file if '--force' option is given. "
                 "Uses an alternate directory if '--path DIRECTORY' is given")

        # Command `certificate generate`
        prepare = certificate_sub_parsers.add_parser(
            'registration-instructions',
            help="Prepares certificate of current component to send other FL participant through trusted channel.")

        register_parser.set_defaults(func=self._register_certificate)
        list_parser.set_defaults(func=self._list_certificates)
        delete_parser.set_defaults(func=self._delete_certificate)
        generate.set_defaults(func=self._generate_certificate)
        prepare.set_defaults(func=self._prepare_certificate_for_registration)

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
            '--ip',
            metavar='IP_ADDRESS',
            type=str,
            nargs='?',
            required=True,
            help="IP address of the component which the certificate will be saved.")

        register_parser.add_argument(
            '--port',
            metavar='PORT',
            type=int,
            nargs='?',
            required=True,
            help="Port number of the component which the certificate will be saved.")

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
            '-f',
            '--force',
            action="store_true",
            help="Forces to overwrite certificate files"
        )

        # Set db path that certificate manager will be using to store certificates
        self._certificate_manager.set_db(db_path=self._environ["DB_PATH"])

    def _create_magic_dev_environment(self):
        """Creates development environment for MPSDPZ"""

        db_names = get_existing_component_db_names()
        certificates = get_all_existing_certificates()

        if len(certificates) <= 2:
            print(f"\n{RED}Warning!{NC}")
            print(f"{BOLD}There is {len(certificates)} Fed-BioMed component created.For 'certificate-dev-setup' "
                  f"you should have at least 2 components created{NC}\n")
            return

        for id_, db_name in db_names.items():
            print(f"Registering certificates for component {id_} ------------------")
            # Sets DB
            self._certificate_manager.set_db(
                os.path.join(
                    get_fedbiomed_root(),
                    DB_FOLDER_NAME,
                    f"{db_name}.json"
                )
            )

            for certificate in certificates:

                if certificate["party_id"] == id_:
                    continue
                try:
                    self._certificate_manager.insert(**certificate, upsert=True)
                except FedbiomedError as e:
                    CommonCLI.error(f"Can not register certificate for {certificate['party_id']}: {e}")

                print(f"Certificate of {certificate['party_id']} has been registered.")

    def _create_component_configuration(self, args):
        """CLI Handler for creating configuration file for given component

        TODO: This method doesn't do specific action for creating configuration file for
            given component. Since, `environ` will be imported through component CLI, configuration
            file will be automatically created. In future, it might be useful to generate configuration
            files.
        """

        print(f"{GRN}Configuration already existed or was created for component {self._environ['ID']}{NC}")

        pass

    def _generate_certificate(self, args: argparse.Namespace):
        """Generates certificate using Certificate Manager

        Args:
            args: Arguments that are passed after `certificate generate` command
        """

        if not args.force and (os.path.isfile(f"{args.path}/{MPSPDZ_certificate_prefix}.key") or
                               os.path.isfile(f"{args.path}/{MPSPDZ_certificate_prefix}.key")):

            CommonCLI.error(f"Certificate is already existing in {MPSPDZ_certificate_prefix}. \n "
                            f"Please use -f | --force option to overwrite existing certificate.")

        try:
            CertificateManager.generate_self_signed_ssl_certificate(
                certificate_folder=args.path,
                certificate_name=MPSPDZ_certificate_prefix,
                component_id=self._environ["ID"]
            )
        except FedbiomedError as e:
            CommonCLI.error(f"Can not generate certificate. Please see: {e}")

        CommonCLI.success(f"Certificate has been successfully generated in : {args.path} \n")

        print(f"Please make sure in {os.getenv('CONFIG_FILE', 'component')}, the section [mpspdz] `public_key` "
              f"and `private_key` has new generated certificate files : \n\n"
              f"{BOLD}Certificates are saved in {NC}\n"
              f"{args.path}/{MPSPDZ_certificate_prefix}.key \n"
              f"{args.path}/{MPSPDZ_certificate_prefix}.pem \n\n"
              f"{YLW}IMPORTANT:{NC}\n"
              f"{BOLD}Since the certificate is renewed please ask other parties "
              f"to register your new certificate.{NC}\n")

        pass

    def _register_certificate(self, args: argparse.Namespace):
        """ Registers certificate with given parameters

        Args:
            args: Parser arguments
        """

        try:
            self._certificate_manager.register_certificate(
                certificate_path=args.public_key,
                party_id=args.party_id,
                upsert=args.upsert,
                ip=args.ip,
                port=args.port
            )
        except FedbiomedError as exp:
            print(exp)
            sys.exit(1)
        else:
            print(f"{GRN}Success!{NC}")
            print(f"{BOLD}Certificate has been successfully created for party: {args.party_id}.{NC}")

    def _list_certificates(self, args: argparse.Namespace):
        """ Lists saved certificates """

        self._certificate_manager.list(verbose=True)

    def _delete_certificate(self, args: argparse.Namespace):

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
                CommonCLI.success(f"Certificate for '{party_id}' has been successfully removed")
                return
            except (ValueError, IndexError, AssertionError):
                CommonCLI.error('Invalid option. Please, try again.')

    def _prepare_certificate_for_registration(self, args: argparse.Namespace):
        """Prints instruction to registration of the certificate by the other parties """

        try:
            with open(self._environ["MPSPDZ_CERTIFICATE_PEM"], 'r') as file:
                certificate = file.read()
                file.close()
        except Exception as e:
            CommonCLI.error(f"Error while reading certificate: {e}")

        else:
            print("Hi There! \n\n")
            print("Please find following certificate to register \n")
            print(certificate)

            print(f"{BOLD}Please follow the instructions below to register this certificate:{NC}\n\n")

            print(" 1- Copy certificate content into a file e.g 'Hospital1.pem'")
            print(" 2- Change your directory to 'fedbiomed' root")
            print(f" 2- Run: \"scripts/fedbiomed_run [node | researcher] certificate register -pk [PATH WHERE "
                  f"CERTIFICATE IS SAVED] -pi {self._environ['ID']}  --ip {self._environ['MPSPDZ_IP']} "
                  f"--port {self._environ['MPSPDZ_PORT']}\" ")

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
    cli = CommonCLI()
    # Initialize only development magic parser
    cli.initialize_magic_dev_environment_parsers()
    cli.parse_args()

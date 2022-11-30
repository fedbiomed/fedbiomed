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

# Create certificate dict validator
CertificateDataValidator = SchemeValidator({
    'DB_PATH': {"rules": [str], "required": True}
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
        self._certificate_manager: CertificateManager = CertificateManager()
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

    def initialize_certificate_parser(self, data: Dict):
        """Common arguments """

        """ Validate data """
        try:
            CertificateDataValidator.validate(data)
        except ValidateError as e:
            raise FedbiomedError(
                f"Inconvenient 'data' value. Certificate CLI manager can not be initialized. Error: {e}"
            )

        subparsers = self._parser.add_subparsers(help='Certificate sub-commands', dest='certificate')

        certificate_parser = subparsers.add_parser('certificate', help='a help')
        certificate_sub_parsers = certificate_parser.add_subparsers(help='Holala')

        register_parser = certificate_sub_parsers.add_parser('register')
        list_parser = certificate_sub_parsers.add_parser('list')

        register_parser.set_defaults(func=self._register_certificate)
        list_parser.set_defaults(func=self._list_certificates)

        register_parser.add_argument('-pk',
                                     '--public-key',
                                     metavar='PUBLIC_KEY',
                                     type=str,
                                     nargs='?',
                                     help='Certificate/key that will be registered')

        register_parser.add_argument('-pi',
                                     '--party-id',
                                     metavar='PUBLIC_ID',
                                     type=str,
                                     nargs='?',
                                     help="ID of the party to which the certificate is to be registered (component"
                                          " ID)")

        register_parser.add_argument('--upsert',
                                     action="store_true",
                                     help="Updates if certificate of given party id is already existing ")

        # Set db path that certificate manager will be using to store certificates
        self._certificate_manager.set_db(db_path=data["DB_PATH"])

    def _register_certificate(self, args):
        """ Registers certificate with given parameters"""

        if not os.path.isfile(args.public_key):
            print("'-pk | --public-key' should be a valid file.")
            sys.exit(101)

        if not args.party_id:
            print("'-pi | --party-id' is required.")
            sys.exit(101)

        try:
            self._certificate_manager.register_certificate(
                certificate_path=args.public_key,
                party_id=args.party_id,
                upsert=args.upsert
            )
        except FedbiomedError as exp:
            print(exp)
            sys.exit(101)
        else:
            print(f"{GRN}Success!{NC}")
            print(f"{BOLD}Certificate has been successfully created for party: {args.party_id}.{NC}")

    def _list_certificates(self, args):
        """ Lists saved certificates """

        self._certificate_manager.list(verbose=True)

    def parse_args(self):
        """"""
        self._args = self._parser.parse_args()

        if hasattr(self._args, 'func'):
            self._args.func(self._args)


if __name__ == '__main__':
    print("ERROR:")
    print("This is a submodule. You can not execute directly. Please import and extend CLI parser")
    exit(2)

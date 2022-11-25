"""Common CLI Modules

This module includes common CLI methods and parser extension

"""

import argparse
from typing import Dict
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.validator import SchemeValidator, ValidateError
from fedbiomed.common.certificate_manager import CertificateManager

# Create certificate dict validator
CertificateDataValidator = SchemeValidator({
    'DP_PATH': {"rules": [str], "required": True}
})


class CommonCLI:

    def __init__(self):
        self._parser: argparse.ArgumentParser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
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

        self._parser.add_argument('-r',
                                  '--register',
                                  action='store_true')

        self._parser.add_argument('-c',
                                  '--certificate',
                                  metavar='CERTIFICATE',
                                  type=str,
                                  nargs='?',
                                  help='Certificate path or certificate string')


        # Set db path that certificate manager will be using to store certificates
        self._certificate_manager.set_db(db_path=data["DP_PATH"])

        # args = self.parser.parse_args()
        #
        # if(args.help and args.register):
        #     self._parser.print_help()
        #
        # print(args.register)
        # print(args.certificate)

    def _register_certificate(self, certificate: str):

        pass


    def parse_args(self):
        """"""
        self._args = self._parser.parse_args()

        if self._args.register:
             print('Tests')


if __name__ == '__main__':
    print("ERROR:")
    print("This is a submodule. You can not execute directly. Please import and extend CLI parser")
    exit(2)

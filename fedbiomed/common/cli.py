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
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        self.certificate_manager = CertificateManager()

    def initialize_certificate_parser(self, data: Dict):
        """Common arguments """

        """ Validate data """
        try:
            CertificateDataValidator.validate(data)
        except ValidateError as e:
            raise FedbiomedError(
                f"Inconvenient 'data' value. Certificate CLI manager can not be initialized. Error: {e}"
            )

        self.parser.add_argument('-c',
                                 '--certificate',
                                 metavar='N',
                                 type=str,
                                 nargs='?',
                                 help='Certificate path or certificate string')

        args = self.parser.parse_args()

        print(args.certificate)

    def _register_certificate(self):
        pass


if __name__ == '__main__':
    print("ERROR:")
    print("This is a submodule. You can not execute directly. Please import and extend CLI parser")
    exit(2)

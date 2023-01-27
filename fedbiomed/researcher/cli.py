# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Researcher CLI """

from fedbiomed.common.cli import CommonCLI
from fedbiomed.researcher.environ import environ

__intro__ = """
   __         _ _     _                          _ 
  / _|       | | |   (_)                        | |
 | |_ ___  __| | |__  _  ___  _ __ ___   ___  __| |
 |  _/ _ \/ _` | '_ \| |/ _ \| '_ ` _ \ / _ \/ _` |
 | ||  __/ (_| | |_) | | (_) | | | | | |  __/ (_| |    _ 
 |_| \___|\__,_|_.__/|_|\___/|_| |_| |_|\___|\__,_|   | |              
                     _ __ ___  ___  ___  __ _ _ __ ___| |__   ___ _ __ 
                    | '__/ _ \/ __|/ _ \/ _` | '__/ __| '_ \ / _ \ '__|
                    | | |  __/\__ \  __/ (_| | | | (__| | | |  __/ |   
                    |_|  \___||___/\___|\__,_|_|  \___|_| |_|\___|_|   
"""


class ResearcherCLI(CommonCLI):

    def __init__(self):
        super().__init__()
        self.description = f"{__intro__}: A CLI app for fedbiomed researchers."
        self._environ = environ

    def launch_cli(self):

        self.initialize_certificate_parser()
        self.initialize_create_configuration()

        print(__intro__)
        print('\t- 🆔 Your researcher ID:', environ['RESEARCHER_ID'], '\n')

        # Parse CLI arguments after the arguments are ready
        self.parse_args()


if __name__ == '__main__':
    ResearcherCLI().launch_cli()

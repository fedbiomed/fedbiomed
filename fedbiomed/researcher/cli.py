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


def main():
    cli = CommonCLI()
    CommonCLI.parser.description = f"{__intro__}: A CLI app for fedbiomed researchers."
    cli.initialize_certificate_parser()

    print(__intro__)
    print('\t- 🆔 Your node ID:', environ['NODE_ID'], '\n')


if __name__ == '__main__':
    main()

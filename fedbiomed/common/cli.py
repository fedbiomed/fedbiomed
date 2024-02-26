# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Common CLI Modules

This module includes common CLI methods and parser extension

"""

import argparse
import importlib
import os
import sys

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.certificate_manager import CertificateManager
from fedbiomed.common.constants import DB_FOLDER_NAME, MPSPDZ_certificate_prefix, ComponentType
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import get_existing_component_db_names, \
    get_all_existing_certificates, \
    get_method_spec, \
    ROOT_DIR
from fedbiomed.common.secagg_manager import SecaggBiprimeManager
from fedbiomed.common.config import Config
from fedbiomed.common.constants import CONFIG_FOLDER_NAME

RED = '\033[1;31m'  # red
YLW = '\033[1;33m'  # yellow
GRN = '\033[1;32m'  # green
NC = '\033[0m'  # no color
BOLD = '\033[1m'


class CLIArgumentParser:

    def __init__(self, subparser: argparse.ArgumentParser):

        self._subparser = subparser
        # Parser that is going to be add using subparser
        self._parser = None

    def default(self, args: argparse.Namespace = None) -> None:
        """Default function for subparser command"""
        
        self._parser.print_help()
        
        return None
        

class ConfigNameAction(ABC, argparse.Action):
    """Action for the argument config

    This action class gets the config file name and set environ object before
    executing any command.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #Sets environ by default if option string for config is not present.
        # The default is defined by the argument parser.
        if not set(self.option_strings).intersection(set(sys.argv)):
            self.set_environ(self.default)

    def __call__(self, parser, namespace, values, option_string=None):
        """When argument is called"""

        if not set(['--help', '-h']).intersection(set(sys.argv)):
            self.set_environ(values)

    @abstractmethod
    def import_environ(self) -> 'Environ':
        """Implements environ import


        Returns:
            Environ object
        """

    def set_environ(self, config_file: str):
        """Sets environ

        Args:
          config_file: Name of the config file that is activate
        """

        print(f'\n# {GRN}Using configuration file:{NC} {BOLD}{config_file}{NC} #')
        os.environ["CONFIG_FILE"] = config_file

        environ = self.import_environ()

        os.environ[f"FEDBIOMED_ACTIVE_{self._component.name}_ID"] = environ["ID"]

        # Sets environ for the CLI. This implementation is required for
        # the common CLI option that are present in fedbiomed.common.cli.CommonCLI
        self._this.set_environ(environ)

        # this may be changed on command line or in the config_node.ini
        logger.setLevel("DEBUG")


class ConfigurationParser(CLIArgumentParser):
    """Instantiates configuration parser"""

    def initialize(self):
        """Initializes argument parser for creating configuration file."""

        self._parser = self._subparser.add_parser(
            'configuration',
            help='The helper for generating or updating component configuration files, see `configuration -h`'
                 ' for more details')
        
        self._parser.set_defaults(func=self.default)

        # Common parser to register common arguments for create and refresh
        common_parser = argparse.ArgumentParser(add_help=False)
        common_parser.add_argument(
            '-r',
            '--root',
            metavar='ROOT_PATH_FEDBIOMED',
            type=str,
            nargs='?',
            default=None,
            help='Root directory for configuration and Fed-BioMed setup')

        # Add arguments
        common_parser.add_argument(
            '-n',
            '--name',
            metavar='CONFIGURATION_FILE_NAME',
            type=str,
            nargs='?',
            required=False,
            help='Name of configuration file')

        common_parser.add_argument(
            '-c',
            '--component',
            metavar='COMPONENT_TYPE[ NODE|RESEARCHER ]',
            type=str,
            nargs='?',
            required=True,
            help='Component type NODE or RESEARCHER')


        # Create sub parser under `configuration` command
        configuration_sub_parsers = self._parser.add_subparsers()

        create = configuration_sub_parsers.add_parser(
            'create',
            parents=[common_parser],
            help="Creates configuration file for the specified component if it does not exist. "
                 "If the configuration file exists, leave it unchanged"
        )

        refresh = configuration_sub_parsers.add_parser(
            'refresh',
            parents=[common_parser],
            help="Refreshes the configuration file by overwriting parameters without changing component ID"
        )

        create.add_argument(
            '-uc',
            '--use-current',
            action="store_true",
            help="Creates configuration only if there isn't an existing one"
        )

        create.add_argument(
            '-f',
            '--force',
            action="store_true",
            help='Force configuration create')

        create.set_defaults(func=self.create)
        refresh.set_defaults(func=self.refresh)

    def _create_config_instance(self, component, root, name):

        # TODO: this implementation is a temporary hack as it introduces a dependency of
        # fedbiomed.common to fedbiomed.node or fedbiomed.researcher
        # To be suppressed when redesigning the imports
        if component.lower() == "node":
            NodeConfig = importlib.import_module('fedbiomed.node.config').NodeConfig
            config = NodeConfig(root=root, name=name, auto_generate=False)
        elif component.lower() == "researcher":
            ResearcherConfig = importlib.import_module('fedbiomed.researcher.config').ResearcherConfig
            config = ResearcherConfig(root=root, name=name, auto_generate=False)
        else:
            print(f"Undefined component type {component}")
            exit(101)

        return config

    def _update_register_default_biprimes(self, config: Config):
        """Post common action after configuration is create

        Args:
          config: Config instance Node or Researcher
        """

        df_biprimes = config.get('mpspdz', 'allow_default_biprimes')
        biprimes_dir = os.path.normpath(
            os.path.join(config.root, CONFIG_FOLDER_NAME , config.get('mpspdz', 'default_biprimes_dir'))
        )
        # Update secure aggregation biprimes in component database
        print(
            "Updating secure aggregation default biprimes with:\n"
            f"ALLOW_DEFAULT_BIPRIMES : {df_biprimes}\n"
            f"DEFAULT_BIPRIMES_DIR   : {biprimes_dir}\n"
        )

        db_path = os.path.normpath(
            os.path.join(config.root, CONFIG_FOLDER_NAME, config.get('default', 'db'))
        )
        BPrimeManager = SecaggBiprimeManager(db_path)
        BPrimeManager.update_default_biprimes(df_biprimes, biprimes_dir)


    def create(self, args):
        """CLI Handler for creating configuration file and assets for given component

        TODO: This method doesn't yet concentrate all actions for creating configuration file for
            given component. Since, `environ` will be imported through component CLI, configuration
            file will be automatically created. In future, it might be useful to generate configuration
            files.
        """

        config = self._create_config_instance(args.component, args.root, args.name)

        # Overwrite force configuration file
        if config.is_config_existing() and args.force:
            print("Overwriting existing configuration file")
            config.generate(force=True)

        # Use exisintg one (do nothing)
        elif config.is_config_existing() and not args.force:
            if not args.use_current:
                print(f"Configuration file \"{config.path}\" is alreay existing for name \"{config.name}\". "
                      "Please use --force option to overwrite")
                exit(101)
            # Generate wont do anything
            config.generate()
        else:
            logger.info(f"Generation new configuration file \"{config.name}\"")
            config.generate()

        self._update_register_default_biprimes(config)

    def refresh(self, args):
        """Refreshes configuration file """

        config = self._create_config_instance(args.component, args.root, args.name)
        print("Refreshing configuration file using current environment variables. This operation will overwrite"
              "existing configuration file without changing component id.")

        # Refresh
        config.refresh()
        print("Configuration has been updated!")



class CommonCLI:

    _arg_parsers_classes: List[type] = []
    _arg_parsers: Dict[str, CLIArgumentParser] = {}

    def __init__(self) -> None:
        self._parser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog='fedbiomed_run',
            formatter_class=argparse.RawTextHelpFormatter
        )

        self._subparsers = self._parser.add_subparsers()
        self._certificate_manager: CertificateManager = CertificateManager()
        self._environ = None
        self._description: str = ''
        self._args = None

        # Initialize configuration parser
        self.configuration_parser = ConfigurationParser(self._subparsers)

    @property
    def parser(self) -> argparse.ArgumentParser:
        """Gets parser for CLI

        Returns:
            Main argument parser object
        """
        return self._parser

    @property
    def subparsers(self):
        """Gets subparsers of common cli

        Returns:
          Subparsers of CLI parser
        """
        return self._subparsers

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
    def config_action(this: 'CommonCLI', component: ComponentType):
        """Returns CLI argument action for config file name"""
        return ConfigNameAction

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

    def initialize_optional(self):
        """Initializes optional subparser

        Optional subparsers are not going to be visible for the CLI that are
        inherited from CommonCLI class as long as `intialize_optional` method
        is not executed.
        """

        self.configuration_parser.initialize()
        self.initialize_magic_dev_environment_parsers()

    def initialize(self):
        """Initializes parser classes and common parser for child clisses.

        This parser classes will be added by child classes.
        """

        for arg_parser in self._arg_parsers_classes:
            p = arg_parser(self._subparsers)
            p.initialize()
            self._arg_parsers.update({arg_parser.__name__ : p})

        self.initialize_certificate_parser()

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


    def initialize_certificate_parser(self):
        """Common arguments """

        # Add certificate sub parser (sub-command)
        certificate_parser = self._subparsers.add_parser(
            'certificate',
            help="Command to manage certificates in node and researcher components. " 
                 "Please see 'certificate --help' for more information.",
            prog="fedbiomed_run [ node | researcher ] [--config [CONFIG_FILE]] certificate",
        )
        
        def print_help(args):
            certificate_parser.print_help()

        certificate_parser.set_defaults(func=print_help)

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
            required=False,
            help="The path where certificates will be saved. By default it will overwrite existing certificate.")

        generate.add_argument(
            '-f',
            '--force',
            action="store_true",
            help="Forces to overwrite certificate files"
        )

    def _create_magic_dev_environment(self):
        """Creates development environment for MPSDPZ"""

        db_names = get_existing_component_db_names()
        certificates = get_all_existing_certificates()

        if len(certificates) <= 2:
            print(f"\n{RED}Error!{NC}")
            print(f"{BOLD}There is {len(certificates)} Fed-BioMed component(s) created.For 'certificate-dev-setup' "
                  f"you should have at least 2 components created{NC}\n")
            return

        for id_, db_name in db_names.items():
            print(f"Registering certificates for component {id_} ------------------")
            # Sets DB
            self._certificate_manager.set_db(
                os.path.join(
                    ROOT_DIR,
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

    def _generate_certificate(self, args: argparse.Namespace):
        """Generates certificate using Certificate Manager

        Args:
            args: Arguments that are passed after `certificate generate` command
        """

        if not args.force and (os.path.isfile(f"{args.path}/{MPSPDZ_certificate_prefix}.key") or
                               os.path.isfile(f"{args.path}/{MPSPDZ_certificate_prefix}.key")):

            CommonCLI.error(f"Certificate is already existing in {MPSPDZ_certificate_prefix}. \n "
                            f"Please use -f | --force option to overwrite existing certificate.")

        path = os.path.join(self._environ["CERT_DIR"], f"cert_{self._environ['ID']}") if not args.path else args.path

        try:
            CertificateManager.generate_self_signed_ssl_certificate(
                certificate_folder=path,
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
        self._certificate_manager.set_db(db_path=self._environ["DB_PATH"])

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
        print(f"{GRN}Listing registered certificates...{NC}")

        self._certificate_manager.set_db(db_path=self._environ["DB_PATH"])
        self._certificate_manager.list(verbose=True)

    def _delete_certificate(self, args: argparse.Namespace):

        self._certificate_manager.set_db(db_path=self._environ["DB_PATH"])
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
            print(f" 3- Run: \"scripts/fedbiomed_run [node | researcher] certificate register -pk [PATH WHERE "
                  f"CERTIFICATE IS SAVED] -pi {self._environ['ID']}  --ip {self._environ['MPSPDZ_IP']} "
                  f"--port {self._environ['MPSPDZ_PORT']}\" ")
            print("    Examples commands to use for VPN/docker mode:")
            print("      ./scripts/fedbiomed_run node certificate register -pk ./etc/cert-secagg "
                  f"-pi {self._environ['ID']} --ip {self._environ['MPSPDZ_IP']} --port {self._environ['MPSPDZ_PORT']}")
            print("      ./scripts/fedbiomed_run researcher certificate register -pk ./etc/cert-secagg "
                  f"-pi {self._environ['ID']} --ip {self._environ['MPSPDZ_IP']} --port {self._environ['MPSPDZ_PORT']}")
        pass

    def parse_args(self, args_ = None):
        """Parse arguments after adding the arguments

        !!! warning "Attention"
                Please make sure this method is called after all necessary arguments are set
        """
        args, unknown_args = self._parser.parse_known_args(args_)
        if hasattr(args, 'func'):
            specs = get_method_spec(args.func)
            if specs:
                 # If default function has 2 arguments
                if len(specs) > 1:
                    return args.func(args, unknown_args)
                else:
                    # Run parser_args to raise error for unrecognized arguments
                    if unknown_args:
                        args = self._parser.parse_args(args_)
                    args.func(args)
            else:
                 # Raise for unrecognized arguments
                if unknown_args:
                    self._parser.parse_args(args_)
                args.func()
        else:
            args = self._parser.print_help()


if __name__ == '__main__':
    cli = CommonCLI()
    # Initialize only development magic parser
    cli.initialize_magic_dev_environment_parsers()
    cli.initialize_optional()
    cli.parse_args()

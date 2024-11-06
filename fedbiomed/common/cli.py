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
from typing import Dict, List

from fedbiomed.common.certificate_manager import CertificateManager
from fedbiomed.common.constants import DB_FOLDER_NAME, ComponentType
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import (
    ROOT_DIR,
    get_all_existing_certificates,
    get_existing_component_db_names,
    get_method_spec,
    read_file
)

RED = "\033[1;31m"  # red
YLW = "\033[1;33m"  # yellow
GRN = "\033[1;32m"  # green
NC = "\033[0m"  # no color
BOLD = "\033[1m"


class CLIArgumentParser:

    def __init__(self, subparser: argparse.ArgumentParser, parser = None):

        self._subparser = subparser
        # Parser that is going to be add using subparser
        self._parser = None

        self._main_parser = parser

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

        # Sets environ by default if option string for config is not present.
        # The default is defined by the argument parser.
        if (
            not set(self.option_strings).intersection(set(sys.argv))
            and not set(["--help", "-h"]).intersection(set(sys.argv))
            and len(sys.argv) > 2
        ):
            self.set_environ()
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """When argument is called"""

        if not set(["--help", "-h"]).intersection(set(sys.argv)):
            self.set_environ(values)

        setattr(namespace, self.dest, values)

    @abstractmethod
    def import_environ(self, config_file: str | None = None) -> "Environ":
        """Implements environ import


        Returns:
            Environ object
        """

    def set_environ(self, config_file: str | None = None):
        """Sets environ

        Args:
          config_file: Name of the config file that is activate
        """

        print(f"\n# {GRN}Using component located at:{NC} {BOLD}{config_file}{NC} #")
        environ = self.import_environ(config_file)

        os.environ[f"FEDBIOMED_ACTIVE_{self._component.name}_ID"] = environ["ID"]

        # Sets environ for the CLI. This implementation is required for
        # the common CLI option that are present in fedbiomed.common.cli.CommonCLI
        self._this.set_environ(environ)

        # this may be changed on command line or in the config_node.ini
        logger.setLevel("DEBUG")


class ComponentParser(CLIArgumentParser):
    """Instantiates configuration parser"""

    def initialize(self):
        """Initializes argument parser for creating configuration file."""

        self._parser = self._subparser.add_parser(
            "component",
            help="The helper for generating or updating component configuration files, see `configuration -h`"
            " for more details",
        )

        self._parser.set_defaults(func=self.default)

        # Common parser to register common arguments for create and refresh
        common_parser = argparse.ArgumentParser(add_help=False)
        common_parser.add_argument(
            "-r",
            "--root",
            metavar="ROOT_PATH_FEDBIOMED",
            type=str,
            nargs="?",
            required=True,
            help="Root directory for configuration and Fed-BioMed setup",
        )

        common_parser.add_argument(
            "-c",
            "--component",
            metavar="COMPONENT_TYPE[ NODE|RESEARCHER ]",
            type=str,
            nargs="?",
            required=True,
            help="Component type NODE or RESEARCHER",
        )

        # Create sub parser under `configuration` command
        component_sub_parsers = self._parser.add_subparsers()

        create = component_sub_parsers.add_parser(
            "create",
            parents=[common_parser],
            help="Creates configuration file for the specified component if it does not exist. "
            "If the configuration file exists, leave it unchanged",
        )

        refresh = component_sub_parsers.add_parser(
            "refresh",
            parents=[common_parser],
            help="Refreshes the configuration file by overwriting parameters without changing component ID",
        )

        create.add_argument(
            "-eo",
            "--exist-ok",
            action="store_true",
            help="Creates configuration only if there isn't an existing one",
        )

        create.add_argument(
            "-f", "--force", action="store_true", help="Force configuration create"
        )

        create.set_defaults(func=self.create)
        refresh.set_defaults(func=self.refresh)

    def _get_component_instance(self, component):
        """ """
        if component.lower() == "node":
            config_node = importlib.import_module("fedbiomed.node.config")
            _component = config_node.node_component
        elif component.lower() == "researcher":
            config_researcher = importlib.import_module(
                "fedbiomed.researcher.config"
            )
            _component = config_researcher.researcher_component
        else:
            print(f"Undefined component type {component}")
            sys.exit(101)

        return _component

    def create(self, args):
        """CLI Handler for creating configuration file and assets for given component

        TODO: This method doesn't yet concentrate all actions for creating configuration file for
            given component. Since, `environ` will be imported through component CLI, configuration
            file will be automatically created. In future, it might be useful to generate configuration
            files.
        """

        component = self._get_component_instance(args.component)

        # Overwrite force configuration file
        if component.is_component_existing(args.root):
            if not args.exist_ok:
                CommonCLI.error(
                    "Component is already existing in given directory. To ignore "
                    "this error please execute component creation using `--exist-ok`"
                )
            else:
                CommonCLI.success(
                    "Component is already exsiting. Using existing component."
                )
                sys.exit(0)

        logger.info(f"Creating a new component in the directory {args.root}")
        component.create(args.root)

    def refresh(self, args):
        """Refreshes configuration file"""

        component = self._get_component_instance(args.component)
        if not component.is_component_existing(args.root):
            CommonCLI.error(
                "There is no component existing in the given directory."
                "Can not refresh non-existing component."
            )

        print(
            "Refreshing configuration file using current environment variables. "
            "This operation will overwrite existing configuration file without changing "
            "component id."
        )
        config = component.create(args.root)
        config.refresh()
        print("Configuration has been updated!")


class CommonCLI:

    _arg_parsers_classes: List[type] = []
    _arg_parsers: Dict[str, CLIArgumentParser] = {}

    def __init__(self) -> None:
        self._parser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog="fedbiomed_run", formatter_class=argparse.RawTextHelpFormatter
        )

        self._subparsers = self._parser.add_subparsers()
        self._certificate_manager: CertificateManager = CertificateManager()
        self._environ = None
        self._description: str = ""
        self._args = None

        # Initialize configuration parser
        self.configuration_parser = ComponentParser(self._subparsers)

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
    def config_action(this: "CommonCLI", component: ComponentType):
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
            p = arg_parser(self._subparsers, self._parser)
            p.initialize()
            self._arg_parsers.update({arg_parser.__name__: p})

        self.initialize_certificate_parser()

    def initialize_magic_dev_environment_parsers(self) -> None:
        """Initializes argument parser for the option to create development environment."""
        magic = self._subparsers.add_parser(
            "certificate-dev-setup",
            description="Prepares development environment by registering certificates"
                "of each component created in a single clone of Fed-BioMed. Parses configuration "
                "files ends with '.ini' that are created in 'etc' directory. This setup requires "
                "to have one 'researcher' and at least 2 nodes.",
            help="Prepares development environment by registering certificates of each "
                "component created in a single clone of Fed-BioMed.",
        )
        magic.set_defaults(func=self._create_magic_dev_environment)

    def initialize_certificate_parser(self):
        """Common arguments"""

        # Add certificate sub parser (sub-command)
        certificate_parser = self._subparsers.add_parser(
            "certificate",
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
            title="Subcommands",
        )

        register_parser = certificate_sub_parsers.add_parser(
            "register",
            help="Register certificate of specified party. Please run 'fedbiomed_run "
                "[COMPONENT SPECIFICATION] certificate register --help'",
        )  # command register

        list_parser = certificate_sub_parsers.add_parser(
            "list", help="Lists registered certificates"
        )  # command list
        delete_parser = certificate_sub_parsers.add_parser(
            "delete", help="Deletes specified certificate from database"
        )  # command delete

        # Command `certificate generate`
        generate = certificate_sub_parsers.add_parser(
            "generate",
            help="Generates certificate for given component/party if files don't exist yet. "
            "Overwrites existing certificate file if '--force' option is given. "
            "Uses an alternate directory if '--path DIRECTORY' is given",
        )

        # Command `certificate generate`
        prepare = certificate_sub_parsers.add_parser(
            "registration-instructions",
            help="Prepares certificate of current component to send other FL participant"
                "through trusted channel.",
        )

        register_parser.set_defaults(func=self._register_certificate)
        list_parser.set_defaults(func=self._list_certificates)
        delete_parser.set_defaults(func=self._delete_certificate)
        generate.set_defaults(func=self._generate_certificate)
        prepare.set_defaults(func=self._prepare_certificate_for_registration)

        # Add arguments
        register_parser.add_argument(
            "-pk",
            "--public-key",
            metavar="PUBLIC_KEY",
            type=str,
            nargs="?",
            required=True,
            help="Certificate/key that will be registered",
        )

        register_parser.add_argument(
            "-pi",
            "--party-id",
            metavar="PUBLIC_ID",
            type=str,
            nargs="?",
            required=True,
            help="ID of the party to which the certificate is to be registered (component ID).",
        )

        register_parser.add_argument(
            "--upsert",
            action="store_true",
            help="Updates if certificate of given party id is already existing.",
        )

        generate.add_argument(
            "--path",
            type=str,
            nargs="?",
            required=False,
            help="The path where certificates will be saved. By default it will overwrite"
                "existing certificate.",
        )

        generate.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="Forces to overwrite certificate files",
        )

    def _create_magic_dev_environment(self, dummy: None):
        """Creates development environment with cert registration

        This option registers activate component certificates for authentication
        purposes.
        """

        db_names = get_existing_component_db_names()
        certificates = get_all_existing_certificates()

        if len(certificates) <= 2:
            print(f"\n{RED}Error!{NC}")
            print(
                f"{BOLD}There is {len(certificates)} Fed-BioMed component(s) created.For "
                f"'certificate-dev-setup' you should have at least 3 components created{NC}\n"
            )
            return

        for id_, db_name in db_names.items():
            print(f"Registering certificates for component {id_} ------------------")
            # Sets DB
            self._certificate_manager.set_db(
                os.path.join(ROOT_DIR, DB_FOLDER_NAME, f"{db_name}.json")
            )

            for certificate in certificates:

                if certificate["party_id"] == id_:
                    continue
                try:
                    self._certificate_manager.insert(**certificate, upsert=True)
                except FedbiomedError as e:
                    CommonCLI.error(
                        f"Can not register certificate for {certificate['party_id']}: {e}"
                    )

                print(f"Certificate of {certificate['party_id']} has been registered.")

    def _generate_certificate(self, args: argparse.Namespace):
        """Generates certificate using Certificate Manager

        Args:
            args: Arguments that are passed after `certificate generate` command
        """

        if not args.force and (
            os.path.isfile(f"{args.path}/certificate.key")
            or os.path.isfile(f"{args.path}/certificate.pem")
        ):

            CommonCLI.error(
                "Certificate is already existing in. \n "
                "Please use -f | --force option to overwrite existing certificate."
            )

        path = (
            os.path.join(self._environ["CERT_DIR"], f"cert_{self._environ['ID']}")
            if not args.path
            else args.path
        )

        try:
            CertificateManager.generate_self_signed_ssl_certificate(
                certificate_folder=path,
                certificate_name="FBM_certificate",
                component_id=self._environ["ID"],
            )
        except FedbiomedError as e:
            CommonCLI.error(f"Can not generate certificate. Please see: {e}")

        CommonCLI.success(
            f"Certificate has been successfully generated in : {args.path} \n"
        )

        print(
            f"Please make sure in {os.getenv('CONFIG_FILE', 'component')}, the section "
            "`public_key` and `private_key` has new generated certificate files : \n\n"
            f"{BOLD}Certificates are saved in {NC}\n"
            f"{path}/certificate.key \n"
            f"{path}/certificate.pem \n\n"
            f"{YLW}IMPORTANT:{NC}\n"
            f"{BOLD}Since the certificate is renewed please ask other parties "
            f"to register your new certificate.{NC}\n"
        )

    def _register_certificate(self, args: argparse.Namespace):
        """Registers certificate with given parameters

        Args:
            args: Parser arguments
        """
        self._certificate_manager.set_db(db_path=self._environ["DB_PATH"])

        try:
            self._certificate_manager.register_certificate(
                certificate_path=args.public_key,
                party_id=args.party_id,
                upsert=args.upsert,
            )
        except FedbiomedError as exp:
            print(exp)
            sys.exit(1)
        else:
            print(f"{GRN}Success!{NC}")
            print(
                f"{BOLD}Certificate has been successfully created for party: {args.party_id}.{NC}"
            )

    def _list_certificates(self, args: argparse.Namespace):
        """Lists saved certificates"""
        print(f"{GRN}Listing registered certificates...{NC}")

        self._certificate_manager.set_db(db_path=self._environ["DB_PATH"])
        self._certificate_manager.list(verbose=True)

    def _delete_certificate(self, args: argparse.Namespace):

        self._certificate_manager.set_db(db_path=self._environ["DB_PATH"])
        certificates = self._certificate_manager.list(verbose=False)
        options = [d["party_id"] for d in certificates]
        msg = "Select the certificate to delete:\n"
        msg += "\n".join([f"{i}) {d}" for i, d in enumerate(options, 1)])
        msg += "\nSelect: "

        while True:
            try:
                opt_idx = int(input(msg)) - 1
                assert opt_idx in range(len(certificates))

                party_id = certificates[opt_idx]["party_id"]
                self._certificate_manager.delete(party_id=party_id)
                CommonCLI.success(
                    f"Certificate for '{party_id}' has been successfully removed"
                )
                return
            except (ValueError, IndexError, AssertionError):
                CommonCLI.error("Invalid option. Please, try again.")

    def _prepare_certificate_for_registration(self, args: argparse.Namespace):
        """Prints instruction to registration of the certificate by the other parties"""

        certificate = read_file(
            self._environ.get("FEDBIOMED_CERTIFICATE_PEM", self._environ.get("SERVER_SSL_CERT")
        ))

        print("Hi There! \n\n")
        print("Please find following certificate to register \n")
        print(certificate)

        print(
            f"{BOLD}Please follow the instructions below to register this certificate:{NC}\n\n"
        )

        print(" 1- Copy certificate content into a file e.g 'Hospital1.pem'")
        print(" 2- Change your directory to 'fedbiomed' root")
        print(
            f" 3- Run: scripts/fedbiomed_run [node | researcher] certificate register"
            f"-pk [PATH WHERE CERTIFICATE IS SAVED] -pi {self._environ['ID']}"
        )
        print("    Examples commands to use for VPN/docker mode:")
        print(
            "      ./scripts/fedbiomed_run node certificate register -pk ./etc/cert-secagg "
            f"-pi {self._environ['ID']}"
        )
        print(
            "      ./scripts/fedbiomed_run researcher certificate register "
            f"-pk ./etc/cert-secagg -pi {self._environ['ID']}"
        )

    def parse_args(self, args_=None):
        """Parse arguments after adding the arguments

        !!! warning "Attention"
                Please make sure this method is called after all necessary arguments are set
        """
        args, unknown_args = self._parser.parse_known_args(args_)
        if hasattr(args, "func"):
            specs = get_method_spec(args.func)
            if specs:
                # If default function has 2 arguments
                if len(specs) > 1:
                    return args.func(args, unknown_args)

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
            self._parser.print_help()


if __name__ == "__main__":
    cli = CommonCLI()
    # Initialize only development magic parser
    cli.initialize_magic_dev_environment_parsers()
    cli.initialize_optional()
    cli.parse_args()

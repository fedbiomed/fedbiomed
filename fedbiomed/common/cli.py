# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0


"""Common CLI Modules

This module includes common CLI methods and parser extension

"""

import argparse
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List

from fedbiomed.common.certificate_manager import (
    CertificateManager,
    validate_registering_component,
)
from fedbiomed.common.config import Config
from fedbiomed.common.constants import (
    DB_FOLDER_NAME,
    ComponentType,
    __version__,
)
from fedbiomed.common.exceptions import FedbiomedCertificateError, FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import (
    ROOT_DIR,
    get_all_existing_certificates,
    get_existing_component_db_names,
    get_method_spec,
    read_file,
)

RED = "\033[1;31m"  # red
YLW = "\033[1;33m"  # yellow
GRN = "\033[1;32m"  # green
NC = "\033[0m"  # no color
BOLD = "\033[1m"


class CLIArgumentParser:
    def __init__(self, subparser: argparse.ArgumentParser, parser=None):
        self._subparser = subparser
        # Parser that is going to be add using subparser
        self._parser = None

        self._main_parser = parser

    def default(self, args: argparse.Namespace = None) -> None:
        """Default function for subparser command"""

        self._parser.print_help()

        return None


class ComponentDirectoryAction(ABC, argparse.Action):
    """Action for the argument config

    This action class gets the config file name and set config object before
    executing any command.
    """

    _component: ComponentType

    def __call__(self, parser, namespace, values: str, option_string=None) -> None:
        """When argument is called"""

        if values is None:
            values = self.default

        if not set(["--help", "-h"]).intersection(set(sys.argv)):
            self._create_config(values)

        setattr(namespace, self.dest, values)

    @abstractmethod
    def set_component(self, component_dir: str) -> None:
        """Implements configuration import

        Args:
            component_dir: Name of the config file for the component
        """

    def _create_config(self, component_dir: str):
        """Sets configuration
        Args:
           config_file: Name of the config file that is activated
        """
        print(f"\n# {GRN}Using component located at:{NC} {BOLD}{component_dir}{NC} #")

        cdir = os.path.abspath(component_dir)

        if not os.path.isdir(cdir) and "-y" not in sys.argv:
            print(
                f"{BOLD}Action Needed{NC}: Action execution for a component not existing. "
                f"The component directory is not existing in the path {cdir}. \n"
                "Do you want to create this component to continue: (y/N)"
            )
            x = input()

            if not x.lower() == "y":
                sys.exit("Operation is called.")
            else:
                print(f"{GRN}Creating component directory:{NC}{cdir}")

        self.set_component(component_dir)

        # this may be changed on command line or in the config_node.ini


class CommonCLI:
    _arg_parsers_classes: List[type] = []
    _arg_parsers: Dict[str, CLIArgumentParser] = {}

    config: Config

    def __init__(self) -> None:
        self._parser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog="fedbiomed", formatter_class=argparse.RawTextHelpFormatter
        )

        self._subparsers = self._parser.add_subparsers()
        self._certificate_manager: CertificateManager = CertificateManager()
        self._description: str = ""
        self._args = None
        self._path_action: ComponentDirectoryAction | None = None
        if os.environ.get("FBM_DEBUG", "").lower() in ("1", "true", "yes"):
            logger.setLevel("DEBUG")
        else:
            logger.setLevel("INFO")

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

    @staticmethod
    def config_action(this: "CommonCLI", component: ComponentType):
        """Returns CLI argument action for config file name"""
        return ComponentDirectoryAction

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

        self.initialize_magic_dev_environment_parsers()
        self.initialize_version()

    def initialize(self):
        """Initializes parser classes and common parser for child classes.

        This parser classes will be added by child classes.
        """

        self._parser.add_argument("-y", action="store_true")

        for arg_parser in self._arg_parsers_classes:
            p = arg_parser(self._subparsers, self._parser)
            p.initialize()
            self._arg_parsers.update({arg_parser.__name__: p})

        self.initialize_certificate_parser()

    def initialize_version(self):
        """Initializes argument parser for common options."""
        self._parser.add_argument(
            "--version",
            "-v",
            action="version",
            version=str(__version__),
            help="Print software version",
        )

    def initialize_magic_dev_environment_parsers(self) -> None:
        """Initializes argument parser for the option to create development environment."""
        magic = self._subparsers.add_parser(
            "certificate-dev-setup",
            description="Prepares development environment by registering certificates "
            "of each component created under the same Fed-BioMed installation. Parses "
            "configuration files ends with '.ini' that are created in 'etc' "
            "directory. This setup requires to have one 'researcher' and "
            "at least 2 nodes.",
            help="Prepares development environment by registering certificates of each "
            "component created under the same Fed-BioMed installation.",
        )
        magic.set_defaults(func=self._create_magic_dev_environment)

    def initialize_certificate_parser(self):
        """Common arguments"""

        # Add certificate sub parser (sub-command)
        certificate_parser = self._subparsers.add_parser(
            "certificate",
            help="Command to manage certificates in node and researcher components. "
            "Please see 'certificate --help' for more information.",
            prog="fedbiomed [ node | researcher ] [--path [COMPONENT_DIRECTORY]] certificate",
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
            help="Register certificate of specified component. Please run 'fedbiomed' "
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
            help="Generates the certificate of the current component, where its "
            "configuration expects it. Refuses to replace an existing one unless "
            "'--force' is given. Uses an alternate directory if '--path DIRECTORY' "
            "is given.\n"
            "Certificate are here referring to the public certificate and its associated private key "
            "(the latter should remain secret and not shared to other parties).",
        )

        # Command `certificate generate`
        prepare = certificate_sub_parsers.add_parser(
            "registration-instructions",
            help="Prepares certificate of current component to send other FL participant"
            " through trusted channel.",
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
            "-ci",
            "--component-id",
            metavar="PUBLIC_ID",
            type=str,
            nargs="?",
            required=False,
            help="ID of the component to which the certificate is to be registered. "
            "Optional when the certificate embeds its identity; required otherwise.",
        )

        register_parser.add_argument(
            "--upsert",
            action="store_true",
            help="Updates if certificate of given component id is already existing.",
        )

        generate.add_argument(
            "--path",
            type=str,
            nargs="?",
            required=False,
            help="The path to the RESEARCHER|NODE component, in which certificate will be saved."
            " Defaults to the directory the component configuration points at.",
        )

        generate.add_argument(
            "--force",
            action="store_true",
            help="Replaces an existing certificate. The previous private key is lost, "
            "so every party holding the old certificate has to register the new one.",
        )

        generate.add_argument(
            "--san",
            type=str,
            action="append",
            metavar="HOST",
            help="A further host name or IP address nodes reach this researcher at, "
            "repeatable. The configured server host and the loopback names are always "
            "included; give this for a researcher reachable under any other name.",
        )

    def _create_magic_dev_environment(self, dummy: None):
        """Registers, in every local component, the certificates it must trust.

        Performs locally what components otherwise exchange by hand: a researcher
        registers the node certificates, a node registers the researcher's.
        Certificates of a component's own type are skipped — a component never
        registers its own kind, and `certificate list` reports such entries as
        an inconsistency.
        """

        db_names = get_existing_component_db_names()
        certificates = get_all_existing_certificates()

        component_types = {c["component_id"]: c["component_type"] for c in certificates}
        researchers = [
            component_id
            for component_id, type_ in component_types.items()
            if type_ == ComponentType.RESEARCHER.name
        ]
        nodes = [
            component_id
            for component_id, type_ in component_types.items()
            if type_ == ComponentType.NODE.name
        ]

        # One researcher because a node pins exactly one; two nodes because that is
        # the smallest federation worth setting up. Nothing is written otherwise.
        if len(researchers) != 1 or len(nodes) < 2:
            CommonCLI.error(
                "`certificate-dev-setup` sets up one researcher and at least two "
                f"nodes. Found {len(researchers)} researcher(s) "
                f"({', '.join(researchers) or 'none'}) and {len(nodes)} node(s) "
                f"({', '.join(nodes) or 'none'})."
            )

        for id_, db_name in db_names.items():
            print(f"Registering certificates for component {id_} ------------------")
            # Sets DB
            self._certificate_manager.set_db(
                os.path.join(ROOT_DIR, DB_FOLDER_NAME, f"{db_name}.json")
            )

            for certificate in certificates:
                # A component does not register its own kind: expected, not an error
                if certificate["component_type"] == component_types[id_]:
                    continue

                # Anything the shared rule still rejects is a real anomaly — a
                # certificate whose TLS role contradicts the component type it is
                # declared as — so report it rather than skip it quietly.
                try:
                    validate_registering_component(
                        certificate["certificate"],
                        certificate["component_type"],
                        component_types[id_],
                    )
                except FedbiomedCertificateError as e:
                    CommonCLI.error(
                        "Can not register certificate for "
                        f"{certificate['component_id']}: {e}"
                    )

                try:
                    self._certificate_manager.insert(**certificate, upsert=True)
                except FedbiomedError as e:
                    CommonCLI.error(
                        "Can not register certificate for "
                        f"{certificate['component_id']}: {e}"
                    )

                print(
                    f"Certificate of {certificate['component_id']} has been registered."
                )

    def _generate_certificate(self, args: argparse.Namespace):
        """Generates the certificate and private key of the current component.

        Written under the name and in the directory the component's configuration
        already points at, so the result is the certificate it serves. Replacing
        an existing one requires `--force`: the previous private key is lost, and
        every party holding the old certificate has to register the new one.

        Args:
            args: Arguments that are passed after `certificate generate` command
        """
        configured = self.config.getpath("certificate", "public_key")
        path = args.path or os.path.dirname(configured)
        name = os.path.splitext(os.path.basename(configured))[0]

        # Issued for the researcher's server host, which is what nodes verify it
        # under, plus any name given. A node certificate is resolved by fingerprint,
        # never by name, so it is issued for none.
        is_researcher = self.config.COMPONENT_TYPE == ComponentType.RESEARCHER.name
        host = self.config.get("server", "host") if is_researcher else None
        san = [host, *(args.san or [])] if is_researcher else None

        if args.san and not is_researcher:
            CommonCLI.error(
                "'--san' applies to a researcher certificate only: a node "
                "certificate is never verified by name."
            )

        if is_researcher and not args.san:
            logger.info(
                f"No '--san' given: issuing the certificate for the server host "
                f"'{host}' read from {self.config.config_path}, plus the loopback names."
            )

        existing = [
            file
            for file in (f"{name}.key", f"{name}.pem")
            if os.path.isfile(os.path.join(path, file))
        ]
        if existing and not args.force:
            CommonCLI.error(
                f"Certificate already exists in {path}: {', '.join(existing)}. "
                "Use '--force' to replace it."
            )

        try:
            key, pem = CertificateManager.generate_self_signed_ssl_certificate(
                certificate_folder=path,
                certificate_name=name,
                component_id=self.config.get("default", "id"),
                san=san,
            )
        except FedbiomedError as e:
            CommonCLI.error(f"Can not generate certificate. Please see: {e}")

        CommonCLI.success(f"Certificate has been successfully generated in : {path} \n")

        print(
            f"{BOLD}Certificates are saved in {NC}\n"
            f"{key} \n"
            f"{pem} \n\n"
            f"{YLW}IMPORTANT:{NC}\n"
            f"{BOLD}Since the certificate is renewed please ask other parties "
            f"to register your new certificate.{NC}\n"
        )

    def _register_certificate(self, args: argparse.Namespace):
        """Registers certificate with given parameters

        Args:
            args: Parser arguments
        """
        self._certificate_manager.set_db(db_path=self.config.getpath("default", "db"))

        try:
            component_id = self._certificate_manager.register_certificate(
                certificate_path=args.public_key,
                component_id=args.component_id,
                upsert=args.upsert,
                registering_component_type=self.config.COMPONENT_TYPE,
            )
        except FedbiomedError as exp:
            print(exp)
            sys.exit(1)
        else:
            print(f"{GRN}Success!{NC}")
            print(
                f"{BOLD}Certificate has been successfully registered for component: "
                f"{component_id}.{NC}"
            )

    def _list_certificates(self, args: argparse.Namespace):
        """Lists saved certificates"""
        print(f"{GRN}Listing registered certificates...{NC}")

        self._certificate_manager.set_db(db_path=self.config.getpath("default", "db"))
        certificates = self._certificate_manager.list(verbose=True)

        # Registration enforces these invariants; entries predating the checks
        # may still violate them, so flag such leftovers to the user.
        component_type = self.config.COMPONENT_TYPE
        own = [
            d["component_id"]
            for d in certificates
            if d.get("component_type") == component_type
        ]
        if own:
            msg = (
                f"Inconsistency: certificate(s) of this component's own type "
                f"({component_type}) are registered: {', '.join(own)}. Components "
                "register "
                "each other's certificates, never their own type."
            )
            logger.warning(msg)
            logger.security_event(
                operation="certificate_registry_inconsistent",
                status="warning",
                reason="own_component_type_registered",
                component_ids=own,
                component_type=component_type,
                detail=msg,
            )
        if component_type == ComponentType.NODE.name and len(certificates) > 1:
            msg = (
                "Inconsistency: a node registers at most one certificate — its "
                f"researcher's — but {len(certificates)} are registered. Delete "
                "the extra entries."
            )
            logger.warning(msg)
            logger.security_event(
                operation="certificate_registry_inconsistent",
                status="warning",
                reason="multiple_certificates_on_node",
                component_ids=[d["component_id"] for d in certificates],
                component_type=component_type,
                detail=msg,
            )

    def _delete_certificate(self, args: argparse.Namespace):
        self._certificate_manager.set_db(db_path=self.config.getpath("default", "db"))
        certificates = self._certificate_manager.list(verbose=False)
        options = [d["component_id"] for d in certificates]
        msg = "Select the certificate to delete:\n"
        msg += "\n".join([f"{i}) {d}" for i, d in enumerate(options, 1)])
        msg += "\nSelect: "

        while True:
            try:
                opt_idx = int(input(msg)) - 1
                assert opt_idx in range(len(certificates))

                component_id = certificates[opt_idx]["component_id"]
                self._certificate_manager.delete(component_id=component_id)
                CommonCLI.success(
                    f"Certificate for '{component_id}' has been successfully removed"
                )
                return
            except (ValueError, IndexError, AssertionError):
                CommonCLI.error("Invalid option. Please, try again.")

    def _prepare_certificate_for_registration(self, args: argparse.Namespace):
        """Prints this component's certificate and how other components register it."""

        certificate = read_file(self.config.getpath("certificate", "public_key"))

        print("Hi There! \n\n")
        print("Please find following certificate to register \n")
        print(certificate)

        print(
            f"{BOLD}Please follow the instructions below to register this certificate:{NC}\n\n"
        )

        # A component registers the certificates of the other kind, never of its own,
        # so these instructions are for the opposite component to follow.
        registers_on = (
            ComponentType.RESEARCHER.name
            if self.config.COMPONENT_TYPE == ComponentType.NODE.name
            else ComponentType.NODE.name
        ).lower()

        print(" 1- Copy certificate content into a file e.g 'Hospital1.pem'")
        print(f" 2- On each {registers_on}, change your directory to 'fedbiomed' root")
        print(
            f" 3- Run: fedbiomed {registers_on} certificate register "
            "-pk [PATH WHERE CERTIFICATE IS SAVED]"
        )
        print(
            f"\n{BOLD}The component id ({self.config.get('default', 'id')}) is read "
            f"from the certificate, so `-ci` is not needed.{NC}"
        )

    def parse_args(self, args_=None):
        """Parse arguments after adding the arguments

        !!! warning "Attention"
                Please make sure this method is called after all necessary arguments are set
        """
        args, unknown_args = self._parser.parse_known_args(args_)
        if hasattr(args, "func"):
            if self._path_action is not None and getattr(self, "config", None) is None:
                self._path_action._create_config(self._path_action.default)
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

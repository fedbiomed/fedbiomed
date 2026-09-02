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
    CERT_PURPOSE_CLIENT,
    CERT_PURPOSE_SERVER,
    CertificateManager,
)
from fedbiomed.common.config import Config
from fedbiomed.common.constants import (
    ComponentType,
    __version__,
)
from fedbiomed.common.exceptions import FedbiomedError
from fedbiomed.common.logger import logger
from fedbiomed.common.utils import (
    get_all_existing_certificates,
    get_all_existing_config_files,
    get_component_config,
    get_existing_component_db_paths,
    get_method_spec,
    read_file,
)

RED = "\033[1;31m"  # red
YLW = "\033[1;33m"  # yellow
GRN = "\033[1;32m"  # green
NC = "\033[0m"  # no color
BOLD = "\033[1m"

# TLS role each component acts in: a node dials the researcher, which serves.
COMPONENT_PURPOSE = {
    ComponentType.NODE.name: CERT_PURPOSE_CLIENT,
    ComponentType.RESEARCHER.name: CERT_PURPOSE_SERVER,
}


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
            "of each component found directly under the given directory. This setup "
            "requires to have one 'researcher' and at least 2 nodes.",
            help="Prepares development environment by registering certificates of each "
            "component found directly under the given directory.",
        )
        magic.add_argument(
            "--path",
            "-p",
            nargs="?",
            default=".",
            metavar="COMPONENTS_DIRECTORY",
            help="The directory the components are located in, absolute or relative "
            "to the path where CLI is executed. Only its first level is inspected. "
            "Defaults to the path where CLI is executed.",
        )
        magic.add_argument(
            "--prune",
            action="store_true",
            help="Deletes every certificate already registered in each component "
            "before registering the ones found under the path, so a component that "
            "left the federation stops being trusted. Without it, existing "
            "registrations are left as they are.",
        )
        magic.add_argument(
            "--enable-mutual-authentication",
            action="store_true",
            help="Sets `[authentication] mutual_authentication` to True in the "
            "configuration of every component, once their certificates are "
            "registered. Components already running have to be restarted to pick it up.",
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
            "repeatable. The configured server host is always included, and naming any "
            "loopback form issues the certificate for all of them.",
        )

    def _create_magic_dev_environment(self, args: argparse.Namespace):
        """Registers, in every local component, the certificates it must trust.

        Performs locally what components otherwise exchange by hand: a researcher
        registers the node certificates, a node registers the researcher's.
        Certificates of a component's own type are skipped — a component never
        registers its own kind, and registration would refuse them.

        `--prune` clears each component's registrations before writing the new ones,
        so the trust stores describe the components currently under the path and
        nothing else. Without it, what is already registered is kept.

        Args:
            args: Arguments that are passed after `certificate-dev-setup` command
        """

        path = os.path.abspath(args.path)
        db_paths = get_existing_component_db_paths(path)
        certificates = get_all_existing_certificates(path)

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
                f"({', '.join(nodes) or 'none'}) in {path}."
            )

        # Components are known by the directory they were created in, so the log
        # reads in those terms rather than in ids
        component_dirs = {
            id_: os.path.basename(os.path.dirname(os.path.dirname(db_path)))
            for id_, db_path in db_paths.items()
        }

        # A registration the run refused to replace, by the component holding it
        outdated = []

        for id_, db_path in db_paths.items():
            print(f"{NC}{BOLD}# Trust store of {component_dirs[id_]} ({id_}){NC}")
            # Sets DB
            self._certificate_manager.set_db(db_path)

            # Rebuilt from the components found under the path: whoever is no longer
            # there stops being trusted, which keeping the registrations would hide.
            if args.prune:
                for registered in self._certificate_manager.list():
                    stale = registered["component_id"]
                    self._certificate_manager.delete(component_id=stale)
                    print(
                        f"Certificate of {component_dirs.get(stale, stale)} has been "
                        "deleted."
                    )

            for certificate in certificates:
                # A component does not register its own kind: expected, not an error
                if certificate["component_type"] == component_types[id_]:
                    continue

                # Falls back to the id: a missing label must not fail a correct
                # registration
                owner = component_dirs.get(
                    certificate["component_id"], certificate["component_id"]
                )

                # Nothing survives a prune, so this is what a run without one keeps:
                # a registration that no longer matches the served certificate breaks
                # the handshake, so it is called out rather than kept quietly.
                registered = self._certificate_manager.get(certificate["component_id"])
                if registered:
                    if registered["certificate"] == certificate["certificate"]:
                        print(f"Certificate of {owner} is already registered.")
                    else:
                        print(
                            f"{YLW}Certificate of {owner} is registered but differs "
                            f"from the one it serves.{NC}"
                        )
                        outdated.append(f"{owner} on {component_dirs[id_]}")
                    continue

                # Registered through the same entry point as `certificate register`,
                # so this setup cannot write a state that command would refuse.
                try:
                    self._certificate_manager.register(
                        certificate=certificate["certificate"],
                        component_id=certificate["component_id"],
                        registering_purpose=COMPONENT_PURPOSE[component_types[id_]],
                    )
                except FedbiomedError as e:
                    CommonCLI.error(
                        "Can not register certificate for "
                        f"{certificate['component_id']}: {e}"
                    )

                print(f"Certificate of {owner} has been registered.")

        if outdated:
            CommonCLI.error(
                "Registrations that no longer match the certificate the component "
                f"serves were kept, so the federation in {path} is not ready: "
                f"{', '.join(outdated)}. Run with `--prune` to rebuild them."
            )

        # Enforced only once the certificates are in place: a component that
        # demands mutual authentication without them refuses every connection.
        if args.enable_mutual_authentication:
            for config_path in get_all_existing_config_files(path):
                config = get_component_config(config_path)
                if not config.has_section("authentication"):
                    config.add_section("authentication")

                # Rewriting would only cost the file its comments, which
                # configparser drops
                if config.getboolean(
                    "authentication", "mutual_authentication", fallback=False
                ):
                    continue

                config.set("authentication", "mutual_authentication", "True")
                with open(config_path, "w", encoding="UTF-8") as file_:
                    config.write(file_)

            print("Mutual authentication enforced in every component.")

        CommonCLI.success(
            f"Federation of 1 researcher and {len(nodes)} nodes has been set up "
            f"in {path}"
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
                f"'{host}' read from {self.config.config_path}, and no other name."
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
                purpose=COMPONENT_PURPOSE[self.config.COMPONENT_TYPE],
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
                registering_purpose=COMPONENT_PURPOSE[self.config.COMPONENT_TYPE],
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
        self._certificate_manager.list(verbose=True)

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
                # A node pins its certificate at startup; the researcher re-reads
                if self.config.COMPONENT_TYPE == ComponentType.NODE.name:
                    print(
                        f"{YLW}A running node keeps using the certificate it read "
                        "when it started: restart the node for this deletion to take "
                        f"effect.{NC}"
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

        print(" 1- Copy certificate content into a file e.g 'example_node.pem'")
        print(f" 2- On each {registers_on}, run:")
        print(
            f"      fedbiomed {registers_on} certificate register "
            "-pk [PATH WHERE CERTIFICATE IS SAVED]"
        )
        print(
            f"    from the directory the component lives in (default 'fbm-"
            f"{registers_on}'), or add '--path [COMPONENT_DIRECTORY]' right after "
            f"'fedbiomed {registers_on}' to point at it."
        )
        print(
            "    Add '--upsert' when a certificate of this component is already "
            "registered there, e.g. after a renewal."
        )
        component_id = self.config.get("default", "id")
        # `-ci` is optional only while the certificate is one Fed-BioMed issued: its
        # `CN=` counts as a component id under `O=Fed-BioMed`, which a CA does not set.
        print(
            f"\n{BOLD}`-ci` is only needed for a certificate issued outside "
            "Fed-BioMed, which embeds no component id. Register such a one with "
            f"`-ci {component_id}`.{NC}"
        )
        print(
            f"{BOLD}Registering does not enable mutual authentication by itself: "
            "set `[authentication] mutual_authentication = True` in the "
            f"configuration of this component and of every {registers_on}, then "
            f"restart them.{NC}"
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

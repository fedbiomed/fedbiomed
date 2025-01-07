# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""Researcher CLI """
import os
import subprocess
import importlib

from typing import List, Dict

from fedbiomed.common.constants import NOTEBOOKS_FOLDER_NAME
from fedbiomed.common.cli import (
    CommonCLI,
    CLIArgumentParser,
    ComponentDirectoryAction
)
from fedbiomed.common.constants import ComponentType


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


class ResearcherControl(CLIArgumentParser):

    def initialize(self):

        start = self._subparser.add_parser(
            "start", help="Starts Jupyter (server) Notebook for researcher API. The default"
                          "directory will be  notebook directory.")

        start.add_argument(
            "--directory",
            "-dir",
            type=str,
            nargs="?",
            required=False,
            help="The directory where jupyter notebook will be started.")
        start.set_defaults(func=self.start)


    def start(self, args):
        """Starts jupyter notebook"""

        options = ['--NotebookApp.use_redirect_file=false']

        component_path = os.path.join(os.getcwd(), args.path)

        if args.directory:
            nb_start_dir = args.directory
        else:
            nb_start_dir = os.path.join(component_path, NOTEBOOKS_FOLDER_NAME)

        options.append(f"--notebook-dir={nb_start_dir}")

        current_env = os.environ.copy()
        #comp_root = os.environ.get("FBM_RESEARCHER_COMPONENT_ROOT", None)
        command = ["jupyter", "notebook"]
        command = [*command, *options]
        process = subprocess.Popen(command, env=current_env)

        try:
            process.wait()
        except KeyboardInterrupt:
            try:
                process.terminate()
            except Exception:
                pass
            process.wait()


class ResearcherCLI(CommonCLI):
    """Researcher CLI"""

    _arg_parsers_classes: List[type] = [
        ResearcherControl,
    ]
    _arg_parsers: Dict[str, CLIArgumentParser] = {}


    def __init__(self):
        super().__init__()
        self.description = f"{__intro__}\nA CLI app for fedbiomed researchers."
        self.initialize()

    def initialize(self):
        """Initializes Researcher CLI"""


        class ComponentDirectoryActionResearcher(ComponentDirectoryAction):
            _this = self
            _component = ComponentType.RESEARCHER

            def set_component(self, component_dir: str) -> None:
                """Import configuration

                Args:
                    config_name: Name of the config file for the component
                """
                os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = os.path.abspath(
                        component_dir
                )
                module = importlib.import_module("fedbiomed.researcher.config")
                self._this.config = module.config

        super().initialize()

        self._parser.add_argument(
            "--path",
            "-p",
            nargs="?",
            action=ComponentDirectoryActionResearcher,
            default="fbm-researcher",
            help="Name of the config file that the CLI will be activated for. Default "
                 "is 'config_researcher.ini'."
        )
